import os
import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset
from models.common import se3
from utils.pcd import compute_corrs_nn, lift_pcd, sample_pcd, me_quantize_pcd
from utils.misc import sorted_alphanumeric
from torchvision.transforms import Compose
from torchvision.transforms import ColorJitter
from augmentations import random_guided_erase
import MinkowskiEngine as ME
from plyfile import PlyData


FLAG_DEBUG_VIZ = False

class mink_dataset(Dataset):

    def __init__(self, root, name, split, model_points, scene_points, voxel_size=2., augs_rgb=False, augs_erase=False, oracle=None, seed=None, corr_th=2., obj_split='all', fixed_sampling=False, filter_corrs=False):
        self.root = root
        self.name = name
        self.split = split
        self.transforms = None
        self.path_split = os.path.join(self.root, self.split)
        self.model_points = model_points
        self.scene_points = scene_points
        self.voxel_size = voxel_size
        self.corr_th = corr_th
        self.max_corrs = 1000
        self.augs_rgb = augs_rgb
        self.augs_erase = augs_erase
        self.obj_ids = self.get_object_split(obj_split)
        self.oracle = oracle

        self.seed = seed
        self.filter_corrs = filter_corrs
        self.fixed_sampling = fixed_sampling

        self.t_rgb = self.get_augs_rgb()
        self.t_scene_erase = random_guided_erase(erase_size=0.1, prob=0.5)
        
        if self.seed is not None:
            print('SETTING SEED: ', self.seed)
            os.environ['PYTHONHASHSEED'] = str(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = True

        
        if self.oracle is not None:
            oracle_path = os.path.join(self.root, f'{self.split}_bbox_{self.oracle}.json')  
            print("Loading oracle bounding boxes from ", oracle_path)
            
            with open(oracle_path) as f:
                self.oracle_boxes = json.load(f)

        # Get bbox annotations
        with open(os.path.join(root, 'models', 'models_info.json')) as file_json:
            self.gt_bbox = json.load(file_json)

        # Watch out: this does not include plane or background channel
        self.n_classes = len(self.obj_ids)
        self.obj_models = {}
        
        for i, obj_id in enumerate(self.obj_ids):
            model = self.get_obj_pcd(obj_id)
            self.obj_models[obj_id] = model

        self.images = list()
        self.obj_list = list() # this is needed for custom sampling

        for folder in sorted_alphanumeric(os.listdir(self.path_split)):

            with open(os.path.join(self.path_split, folder, 'scene_gt.json')) as f:
                gt = json.load(f)
            with open(os.path.join(self.path_split, folder, 'scene_gt_info.json')) as f:
                meta_gt = json.load(f)  

            if os.path.isdir(os.path.join(self.path_split, folder)):
                    
                for file_rgb in sorted_alphanumeric(os.listdir(os.path.join(self.path_split, folder, 'rgb'))):
                    # get annotation of current image to get present objects
                    img_id = str(int(os.path.splitext(file_rgb)[0]))
                    cur_gt, cur_meta_gt = gt[img_id], meta_gt[img_id]
                    
                    # cycle over object in current image
                    for i, (obj_gt, obj_meta_gt) in enumerate(zip(cur_gt, cur_meta_gt)):
                        obj_id = obj_gt['obj_id']
                        obj_visib = obj_meta_gt['visib_fract']

                        # object present and visible, add to dataset!
                        if obj_id in self.obj_ids and obj_visib > 0.:

                            img_path = os.path.join(self.path_split, folder, 'rgb', file_rgb)
                            self.images.append((img_path, folder, obj_id, i))
                            self.obj_list.append(obj_id)

        self.obj_list = torch.tensor(self.obj_list)
 
    def __getitem__(self, index):

        # Get image
        img_path, folder, obj_id, annot_idx = self.images[index]
        image = Image.open(img_path)
        image = np.asarray(image.convert('RGB'))

        # Get pointers
        pointer_rgb, name_image = os.path.split(img_path)
        image_id = os.path.splitext(name_image)[0]
        part_path = os.path.split(pointer_rgb)[0]
        part_id = os.path.split(part_path)[1]
        depth_path = os.path.join(part_path, 'depth_hf', image_id + '.png')
        mask_path = os.path.join(part_path, 'mask_segm', image_id + '.png')

        # Get annotations
        pointer_annot = os.path.join(part_path, 'scene_gt.json')
        pointer_camera = os.path.join(part_path, 'scene_camera.json')

        with open(pointer_annot) as file_json:
            annot = json.load(file_json)

        with open(pointer_camera) as file_json:
            camera_annot = json.load(file_json)

        with open(os.path.join(self.root, 'models', 'models_info.json')) as f:
            info = json.load(f)

        # get camera data and depth at true scale
        camera = camera_annot[str(int(image_id))]
        cam_K, cam_depth = np.asarray(camera['cam_K']), np.asarray(camera['depth_scale'])
        target_depth = np.asarray(Image.open(depth_path)).astype(np.float64) * cam_depth
        mask = np.asarray(Image.open(mask_path))

        target_depth = np.expand_dims(target_depth,axis=2)
        mask = np.expand_dims(mask,axis=2)

        filter = np.ones(target_depth.shape)
        if self.oracle is not None:
            part_id, img_id, obj_id = int(folder), int(image_id), int(obj_id)
            instance_id = f'{part_id:06d}_{img_id:06d}_{obj_id:02d}'
            if instance_id in self.oracle_boxes.keys():
                x,y,w,h = self.oracle_boxes[instance_id]
                filter = np.zeros(target_depth.shape)
                filter[y:y+h,x:x+w] = 1
            else:
                print('Warning: {} not present in oracle'.format(instance_id))

        target_depth = np.concatenate((target_depth,image,mask,filter), axis=2)

        # get object pcd and pose
        obj_pcd, gt_pose = self._get_target_regr(name_image, annot, obj_id, annot_idx)
        
        # depth lifting to get point cloud
        scene_pcd = lift_pcd(torch.tensor(target_depth), torch.tensor(cam_K))
        
        scene_pcd = sample_pcd(scene_pcd, n_points=self.scene_points)
        obj_pcd = sample_pcd(obj_pcd, n_points=self.model_points)

        if self.augs_rgb:
            obj_pcd[:,3:6] = self.t_rgb(obj_pcd[:,3:6].unsqueeze(0)).squeeze(0)

        if self.augs_erase:
            scene_pcd = self.t_scene_erase(scene_pcd, info[str(obj_id)]['diameter'], obj_id)
        
        # quantize pcds and update translation target
        
        obj_coords, obj_feats = me_quantize_pcd(obj_pcd, self.voxel_size, True)
        scene_coords, scene_feats = me_quantize_pcd(scene_pcd, self.voxel_size, True)
        
        gt_pose[:3,3] = gt_pose[:3,3] / self.voxel_size
        diameter = info[str(obj_id)]['diameter'] / self.voxel_size

        # compute correspondencies
        pts_allowed = torch.nonzero(scene_feats[:, -2] == obj_id).squeeze(1)
        if pts_allowed.shape[0] > 0:
            corrs = self.get_corrs(obj_coords, info[str(obj_id)], gt_pose, scene_coords, pts_allowed)        
        else:
            corrs = torch.zeros((0,2))

        if len(corrs.shape) == 1:
            corrs = torch.zeros((0,2))

        # remove mask channel, no more needed
        scene_filter = torch.nonzero(scene_feats[:,-1])
        scene_feats = scene_feats[:,:3]

        return scene_coords, scene_feats, scene_filter, obj_coords, obj_feats, corrs, gt_pose, cam_K, part_id, image_id, obj_id, diameter

    @staticmethod
    def get_symmetries_info(root):

        with open(os.path.join(root,'models','models_info.json')) as f:
            models_info  = json.load(f)
        
        symm_info = {}
        for k,info in models_info.items():

            is_symmetric = 'symmetries_discrete' in info.keys()
            symm_info[int(k)] = is_symmetric
        
        return symm_info

    def get_corrs(self, obj, obj_info, pose, scene, pts_allowed):

        # initial correspondences
        rot_obj = se3.torch_transform(pose, obj)
        corrs = compute_corrs_nn(rot_obj, scene, pts_allowed, pos_threshold=self.corr_th, max_corr=self.max_corrs)
        
        return corrs

    def __len__(self):
        return len(self.images)

    def _get_target_regr(self, name_image, annot, obj_id, annot_idx):

        # Get scene and ground truth
        img_id = os.path.splitext(name_image)[0]
        obj_annot = annot[str(int(img_id))][annot_idx]
        
        assert obj_id == obj_annot['obj_id']

        # Get bbox rotation and translation
        r = np.array(obj_annot['cam_R_m2c'])
        r = np.reshape(r, (3, 3))
        t = np.array(obj_annot['cam_t_m2c']) 
        t = np.expand_dims(t,axis=1)

        # get object pose
        transform_gt = torch.tensor(np.concatenate((r,t),axis=1))
        obj_pcd = torch.tensor(self.obj_models[obj_id])

        return obj_pcd, transform_gt

    def get_obj_pcd(self, obj_id):
        
        if self.name == 'lmo' or self.name == 'lm':
            pcd = mink_dataset.get_lm_obj_pcd(self.root, obj_id)
        elif self.name == 'ycbv':
            pcd = mink_dataset.get_ycbv_obj_pcd(self.root, obj_id)
        else:
            raise RuntimeError('Unknown dataset from path {}'.format(self.root))
        
        return pcd

    @staticmethod    
    def get_ycbv_obj_pcd(root, obj_id):

        pcd = PlyData.read(os.path.join(root,'models','obj_{:06d}.ply'.format(obj_id)))
        image = Image.open(os.path.join(root, 'models','obj_{:06d}.png'.format(obj_id))).convert('RGB')
        image = np.asarray(image).astype(float)

        w,h = image.shape[:2]

        xs = pcd['vertex']['x']
        ys = pcd['vertex']['y']
        zs = pcd['vertex']['z']

        tu = pcd['vertex']['texture_u']
        tv = pcd['vertex']['texture_v']
        
        raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
        faces = np.stack([vert for vert in raw_vertexs],axis=0)

        norm_u = (tu * w).astype(int)
        norm_v = ((1-tv) * h).astype(int)

        xyz = np.stack((xs,ys,zs), axis=1)
        rgb = image[norm_v, norm_u] / 255.

        pcd = np.concatenate((xyz, rgb), axis=1)

        return pcd

    @staticmethod
    def get_lm_obj_pcd(root, obj_id):
        
        ply_path = os.path.join(root, 'models', 'obj_{:06d}.ply'.format(obj_id))
        pcd = o3d.io.read_triangle_mesh(ply_path)
        
        faces = np.asarray(pcd.triangles) 
        xyz = np.asarray(pcd.vertices)
        rgb = np.asarray(pcd.vertex_colors)

        pcd_data = np.concatenate((xyz, rgb), axis=1)
    
        return pcd_data

    def get_object_split(self, obj_split_name):

        with open(os.path.join(self.root, 'object_splits.json')) as f:
            obj_splits = json.load(f)

        if obj_split_name in obj_splits.keys():
            obj_list = [int(obj_id) for obj_id in obj_splits[obj_split_name]]
            return obj_list
        else:
            raise RuntimeError('Object split {} not present in dataset {} at {}'.format(obj_split_name, self.name, self.root))

    def get_augs_rgb(self):

        # brightness change and small jittering
        return Compose([
            ColorJitter(brightness=.3, contrast=.0, saturation=.0, hue=.0),
            ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.1)
        ])

        

def ME_collation_fn(data):

    scene_coords, scene_feats, scene_filter, obj_coords, obj_feats, corrs, gt_pose, cam_K, part_id, image_id, obj_id, diameter = list(zip(*data))

    scene_coords_b, scene_feats_b = ME.utils.sparse_collate(scene_coords, scene_feats)
    obj_coords_b, obj_feats_b = ME.utils.sparse_collate(obj_coords, obj_feats)

    corrs_feats = [torch.ones((corr.shape[0],1)) for corr in corrs]
    assert len(corrs_feats) == len(corrs)

    try:
        corrs_coords, corrs_feats = ME.utils.sparse_collate(corrs, corrs_feats)
    except IndexError:
        for i,corr in enumerate(corrs):
            print(corr.shape)
            np.save('tmp/corr_{}.npy'.format(i),corr.cpu().numpy())

    gt_pose = torch.stack(gt_pose, dim=0)
    part_id = torch.tensor([int(idx) for idx in part_id])
    image_id = torch.tensor([int(idx) for idx in image_id])
    obj_id = torch.tensor(obj_id)
    diameters = torch.tensor(diameter)
    cam_K = torch.stack([torch.tensor(cam_i) for cam_i in cam_K], dim=0)

    return scene_coords_b, scene_feats_b, scene_filter, obj_coords_b, obj_feats_b, corrs_coords, corrs_feats, gt_pose, cam_K, part_id, image_id, obj_id, diameters