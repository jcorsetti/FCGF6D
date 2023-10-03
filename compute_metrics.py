import argparse
import csv
import json
import os
import sys

import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from utils.misc import boolean_string
from utils.pcd import lift_pcd
from PIL import Image
from tqdm import tqdm
from os import readlink
from utils import project_points
from utils.metrics import rte, rre, mae, rmse, cal_auc
from datasets import mink_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument('--exp', type=str, default='exp00', help='Experiment name')
    parser.add_argument('--dataset', type=str, default='lmo', help='Path of the data')
    parser.add_argument('--split', type=str, default='test', help='Name of partition')
    parser.add_argument('--checkpoint', type=int, help='Name of the checkpoint file')
    parser.add_argument('--exp_root', type=str, default=readlink('exp_data'), help='Root to models folder for storing experiments')
    parser.add_argument('--obj_split', type=str, default=None, help='Objects on which to evaluate. One of lm, lmo, lm-only')
    parser.add_argument('--oracle', type=str, default=None, help='If true, use ground truth detection boxes for testing')
    parser.add_argument('--solver', type=str, default='ransac', help='Solver type for pose, one of [ransac, teaser]')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    #icp args
    parser.add_argument('--icp', type=boolean_string, default=False, help='If true, use ICP to refine pose')
    parser.add_argument('--icp_method', type=str, default='plane', help='Plane or point. Plane requires to compute the normal')
    parser.add_argument('--icp_iters', type=int, default=50, help='Number of iterations')
    parser.add_argument('--icp_th', type=float, default=2., help='ICP threshold')
    parser.add_argument('--icp_fitness', type=float, default=1e-5, help='ICP threshold')

    args = parser.parse_args()
    args.path = readlink('data_{}'.format(args.dataset))

    return args

def apply_icp(obj_pcd, pred_r, pred_t, depth_path, args, box):

    pred_pose = np.zeros((4,4))
    pred_pose[:3,:3] = pred_r
    pred_pose[:3,3] = pred_t
    pred_pose[3,3] = 1.
    
    img_code = os.path.splitext(os.path.split(depth_path)[1])[0]
    part_path = os.path.join(*depth_path.split('/')[:-2])
    
    img_path = os.path.join('/',part_path,'rgb',img_code+'.png')
    with open(os.path.join('/',part_path, 'scene_camera.json')) as f:
        camera = json.load(f)[str(int(img_code))]

    cam_K, cam_depth = camera['cam_K'], camera['depth_scale']
    target_depth = np.asarray(Image.open(depth_path)).astype(float) * cam_depth
    target_rgb = np.asarray(Image.open(img_path)).astype(float)
    target_depth = np.expand_dims(target_depth,axis=2)

    filter = np.ones(target_depth.shape)
    if box is not None:
        x,y,w,h = box
        filter = np.zeros(target_depth.shape)
        filter[y:y+h,x:x+w] = 1

    target_depth = np.concatenate([target_depth,target_rgb, filter],axis=-1)
    depth_pcd = lift_pcd(torch.tensor(target_depth), torch.tensor(cam_K), include_feats=True).numpy()
    filter = depth_pcd[:,-1].astype(np.int8)
    depth_pcd = depth_pcd[filter == 1, :]
    depth = o3d.t.geometry.PointCloud()
    depth.point.positions = o3d.core.Tensor(depth_pcd[:,:3])
    
    obj = o3d.t.geometry.PointCloud()
    obj.point.positions = o3d.core.Tensor(obj_pcd[:,:3])
    
    obj = obj.cuda(0)
    depth = depth.cuda(0)

    voxel_sizes = o3d.cuda.pybind.utility.DoubleVector([args.icp_th, args.icp_th/2., args.icp_th/4])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        o3d.cuda.pybind.t.pipelines.registration.ICPConvergenceCriteria(args.icp_fitness, args.icp_fitness, args.icp_iters),
        o3d.cuda.pybind.t.pipelines.registration.ICPConvergenceCriteria(args.icp_fitness*0.1, args.icp_fitness*0.1, args.icp_iters),
        o3d.cuda.pybind.t.pipelines.registration.ICPConvergenceCriteria(args.icp_fitness*0.01, args.icp_fitness*0.01, args.icp_iters)
    ]
    max_correspondence_distances = o3d.cuda.pybind.utility.DoubleVector([args.icp_th, args.icp_th/2., args.icp_th/4])

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    if args.icp_method == 'plane':
        estimation = o3d.cuda.pybind.t.pipelines.registration.TransformationEstimationPointToPlane()
        obj.estimate_normals(radius=2, max_nn=30)
        depth.estimate_normals(radius=2, max_nn=30)
    else:
        estimation = o3d.cuda.pybind.t.pipelines.registration.TransformationEstimationPointToPoint()


    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    try:
        result_pose = o3d.t.pipelines.registration.multi_scale_icp(
            obj, depth, voxel_sizes, criteria_list, max_correspondence_distances, pred_pose, estimation
        )

        result_pose = result_pose.transformation.numpy()
        result_r = result_pose[:3,:3]
        result_t = result_pose[:3,3]

    except:
        print("Open3d error, using initial pose.")
        result_r, result_t = pred_r, pred_t


    return result_r, result_t

class PrintAndLog:
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        sys.stdout = self
        sys.stdout = self
    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout

def get_files(args):

    basedir = os.path.join(args.exp_root, args.exp, 'results')
    filename = '{}_epoch={:04d}'.format(args.split, args.checkpoint)

    if 'obj_split' in args:
        if args.obj_split is not None:
            filename += '_' + args.obj_split
        else:
            args.obj_split = 'all'

    filename += '_' + args.solver
    
    if args.seed is not None:
        filename += '_{}'.format(args.seed)
    
    if args.normalize:
        filename += '_norm'

    if args.oracle is not None:
        filename += f'_{args.oracle}'
    
    if args.add_rgb:
        filename += '_extrargb'

    test_file = os.path.join(basedir, f'pred_{filename}.csv')
    perf_file = os.path.join(basedir, f'performances_{filename}.json')
    if args.icp:
        perf_log = os.path.join(basedir, f'scores_{filename}_icp_{args.icp_iters}_{args.icp_method}_{args.icp_th}_{args.icp_fitness}.txt')
    else:
        perf_log = os.path.join(basedir, f'scores_{filename}.txt')

    return test_file, perf_file, perf_log

def perf_from_csv(file, obj_list):

    obj_occs = {obj_id:0 for obj_id in obj_list}
    poses = {}

    with open(file) as f:

        reader = list(csv.reader(f, delimiter=','))

        for i_r, row in enumerate(reader):

            if i_r == 0:  # Ignore first row of csv
               continue

            part_id, img_id, obj_id = int(row[0]),int(row[1]),int(row[2])

            if int(obj_id) in obj_list:
                obj_occs[obj_id] += 1

                r = np.resize(np.asarray(row[4].split(), dtype=np.float64), (3,3))
                t = np.asarray(row[5].split(), dtype=np.float64)

                instance_id = '{:06d}_{:06d}_{:02d}'.format(part_id, img_id, obj_id)
                poses[instance_id] = {
                    'r': r,
                    't': t
                }

    return poses, obj_occs

def cal_adds_pvn3d(pred_RT, gt_RT, p3ds):
    
    pred_RT = pred_RT.to(torch.double)
    gt_RT = gt_RT.to(torch.double)
    p3ds = p3ds.to(torch.double)

    N, _ = p3ds.size()
    pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    pd = pd.view(1, N, 3).repeat(N, 1, 1)
    gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    gt = gt.view(N, 1, 3).repeat(1, N, 1)
    dis = torch.norm(pd - gt, dim=2)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis).item()

def get_dict_stats(dict):
    '''
    Get mean and std of dictionary with numerical values
    '''

    values = np.asarray(list(dict.values()))
    return np.mean(values), np.std(values)


def main():

    # Parse input arguments
    args = parse_args()

    model_root = os.path.join(args.path, 'models_eval')
    test_file, perf_file, perf_log = get_files(args)

    if args.oracle is not None:
        oracle_path = os.path.join(args.path, f'{args.split}_bbox_{args.oracle}.json')  
        print("Loading oracle bounding boxes from ", oracle_path)
        with open(oracle_path) as f:
            oracle_boxes = json.load(f)

    with open(os.path.join(args.path, 'object_splits.json')) as f:
        object_lists = json.load(f)
    object_list = [int(obj_id) for obj_id in object_lists[args.obj_split]]

    # Reading object list
    with open(os.path.join(model_root, 'models_info.json')) as f:
        models_info = json.load(f)
    obj_list = [int(obj_id) for obj_id in models_info.keys() if int(obj_id) in object_list]
    sym_obj_list = [int(obj_id) for obj_id, obj_info in models_info.items() if (('symmetries_discrete' in obj_info or 'symmetries_continuous' in obj_info) and int(obj_id) in object_list)]

    pcd_models = {}
    rgb_pcd_models = {}
    diameters = {}
    obj_add_s_list = {} # ADD(S) - ADD for not symmetric, ADDS for symmetric
    obj_adds_list = {}  # ADDS   - ADDS for all!
    
    for obj_id in obj_list:
        if args.dataset == 'ycbv':
            rgb_pcd_models[obj_id] = mink_dataset.get_ycbv_obj_pcd(args.path, obj_id)
        if args.dataset == 'lmo':
            rgb_pcd_models[obj_id] = mink_dataset.get_lm_obj_pcd(args.path, obj_id)
        pcd_models[obj_id] = rgb_pcd_models[obj_id][:,:3]
        diameters[obj_id] = models_info[str(obj_id)]['diameter']
        obj_adds_list[obj_id] = []
        obj_add_s_list[obj_id] = []

    # Reading camera intrinsics matrix
    with open(os.path.join(args.path, 'camera.json')) as f:
        gt_camera = json.load(f)
    K = np.array([
        [gt_camera['fx'], 0.0, gt_camera['cx']],
        [0.0, gt_camera['fy'], gt_camera['cy']],
        [0.0, 0.0, 1.0]])

    gts_file = os.path.join('eval_gts', 'gt_{}_{}.csv'.format(args.split, args.dataset))

    perfs = {
        'exp' : args.exp,
        'split' : args.split, 
        'part' : [],
        'img' : [],
        'obj' : [],
        'gt_rot' : [],
        'gt_trans' : [],
        'pred_rot' : [],
        'pred_trans' : [],
        'add' : [],
        'add(s)' : [],
        'adds': [],
        'rep' : [],
        'rre' : [],
        'rte' : [],
        'mae_t' : [],
        'rmse_t' : [],
        'mae_r' : [],
        'rmse_r' : []
    }

    success_count_add_s = {k:0 for k in obj_list}
    success_count_add = {k:0 for k in obj_list}
    success_count_rep = {k:0 for k in obj_list}
    success_count_rre = {k:0 for k in obj_list}
    success_count_rte = {k:0 for k in obj_list}
    obj_mae_t = {k:0 for k in obj_list}
    obj_rmse_t = {k:0 for k in obj_list}
    obj_mae_r = {k:0 for k in obj_list}
    obj_rmse_r = {k:0 for k in obj_list}

    pred_poses, num_presents = perf_from_csv(test_file, obj_list)
    gt_poses, tot_occurrencies = perf_from_csv(gts_file, obj_list)
    
    some_missing = False
    for obj_id in obj_list:
        missing = tot_occurrencies[obj_id] - num_presents[obj_id]
        if missing > 0:
            print("  Obj {}: instances present are {} / {}".format(obj_id, num_presents[obj_id], tot_occurrencies[obj_id]))
            some_missing = True
    if some_missing:
        print("Warning: above objects have missing instances. Either you are using a detector with some false negatives, or the evaluation process was interrupted.")

    for gt_instance_id in tqdm(gt_poses.keys()):

        part_id, img_id, obj_id = gt_instance_id.split('_')
        part_id, img_id, obj_id = int(part_id), int(img_id), int(obj_id)    

        if gt_instance_id in pred_poses.keys():

            # check for false negative
            if args.oracle is not None:
                if gt_instance_id not in oracle_boxes:
                    # 1 meter of distance is an automatic fail
                    obj_adds_list[obj_id].append(1.)
                    obj_add_s_list[obj_id].append(1.)
                    continue

            obj_idx = obj_list.index(obj_id)
            test_r = pred_poses[gt_instance_id]['r']
            test_t = pred_poses[gt_instance_id]['t']
            gt_r = gt_poses[gt_instance_id]['r']
            gt_t = gt_poses[gt_instance_id]['t']

            if args.icp:

                box = None
                if args.oracle is not None:
                    box = oracle_boxes[gt_instance_id]

                img_path = os.path.join(args.path, args.split, f'{int(part_id):06d}', 'depth_hf', f'{int(img_id):06d}.png')
                test_r, test_t = apply_icp(rgb_pcd_models[obj_id], test_r, test_t, img_path, args, box=box)
            
            model_pred = np.dot(np.asarray(pcd_models[obj_id]), test_r.T) + test_t 
            model_gt = np.dot(np.asarray(pcd_models[obj_id]), gt_r.T) + gt_t

            # ADD computation
            add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

            # ADD-S computation
            kdt = KDTree(model_gt, metric='euclidean')
            distance, _ = kdt.query(model_pred, k=1)
            adds = np.mean(distance)
            
            #adds = cal_adds_pvn3d(pred_rt, gt_rt, torch.tensor(pcd_models[obj_id]))

            if obj_id in sym_obj_list:
                add_s = adds
            else:
                add_s = add

            # pvn3d auc function requires ADD(S) in meters
            obj_adds_list[obj_id].append(adds/1000.)
            obj_add_s_list[obj_id].append(add_s/1000.)

            # REP computation
            proj_pred = project_points(model_pred, K)
            proj_gt = project_points(model_gt, K)
            rep = np.linalg.norm(proj_pred - proj_gt, axis=1).mean()
            rot_err = rre(test_r, gt_r)
            trans_err = rte(test_t, gt_t)

            mae_t = mae(test_t, gt_t)
            rmse_t = rmse(test_t, gt_t)

            mae_r = mae(test_r.flatten(), gt_r.flatten())
            rmse_r = rmse(test_r.flatten(), gt_r.flatten())

            obj_mae_t[obj_id] += mae_t
            obj_rmse_t[obj_id] += rmse_t
            obj_mae_r[obj_id] += mae_r
            obj_rmse_r[obj_id] += rmse_r

            # Output infos to terminal
            perfs['part'].append(int(part_id))
            perfs['img'].append(int(img_id))
            perfs['obj'].append(int(obj_id))
            perfs['add'].append(round(add,2))
            perfs['adds'].append(round(adds,2))
            perfs['add(s)'].append(round(add_s,8))
            perfs['rep'].append(round(rep,2))
            perfs['rre'].append(round(rot_err,2))
            perfs['rte'].append(round(trans_err,2))
            
            perfs['mae_t'].append(round(mae_t,4))
            perfs['rmse_t'].append(round(rmse_t,4))
            perfs['mae_r'].append(round(mae_r,4))
            perfs['rmse_r'].append(round(rmse_r,4))

            perfs['pred_rot'].append(test_r.flatten().tolist())
            perfs['pred_trans'].append(test_t.flatten().tolist())
            perfs['gt_rot'].append(gt_r.flatten().tolist())
            perfs['gt_trans'].append(gt_t.flatten().tolist())

            # ADD-S
            if add_s < 0.1 * diameters[obj_id]:
                success_count_add_s[obj_id] += 1
            # ADD
            if add < 0.1 * diameters[obj_id]:
                success_count_add[obj_id] += 1
            # REP
            if rep < 5:
                success_count_rep[obj_id] += 1
            # RRE
            if rot_err < 5:
                success_count_rre[obj_id] += 1
            # RTE
            if trans_err < 5:
                success_count_rte[obj_id] += 1
        
        else:
            # 1 meter of distance is an automatic fail
            obj_adds_list[obj_id].append(1.)
            obj_add_s_list[obj_id].append(1.)


    # compute mean for every metric
    for obj_id in obj_list:
        if tot_occurrencies[obj_id] > 0:
            success_count_add_s[obj_id] = success_count_add_s[obj_id] / tot_occurrencies[obj_id]
            success_count_add[obj_id] = success_count_add[obj_id] / tot_occurrencies[obj_id]
            success_count_rep[obj_id] = success_count_rep[obj_id] / tot_occurrencies[obj_id]
            success_count_rre[obj_id] = success_count_rre[obj_id] / tot_occurrencies[obj_id]
            success_count_rte[obj_id] = success_count_rte[obj_id] / tot_occurrencies[obj_id]
        else:
            success_count_add_s[obj_id] = 0.
            success_count_add[obj_id] = 0.
            success_count_rep[obj_id] = 0.
            success_count_rre[obj_id] = 0.
            success_count_rte[obj_id] = 0.

        if num_presents[obj_id] > 0:
            obj_mae_r[obj_id] = obj_mae_r[obj_id] / num_presents[obj_id]
            obj_rmse_r[obj_id] = obj_rmse_r[obj_id] / num_presents[obj_id]
            obj_mae_t[obj_id] = obj_mae_t[obj_id] / num_presents[obj_id]
            obj_rmse_t[obj_id] = obj_rmse_t[obj_id] / num_presents[obj_id]
            
        else:
            obj_mae_r[obj_id] = 0.
            obj_rmse_r[obj_id] = 0.
            obj_mae_t[obj_id] = 0.
            obj_rmse_t[obj_id] = 0.
            

    with PrintAndLog(perf_log):
        print('-----ADD-S SCORE-----')
        for obj_id in obj_list:
            print('Obj {}: {}'.format(obj_id,success_count_add_s[obj_id]))
        mean, std = get_dict_stats(success_count_add_s)
        print('tot. avg.: {:.4f}'.format(mean))
        print('std.: {:.4f}'.format(std))

        print('-----ADD-S AUC-----')
        obj_adds_auc = np.zeros(len(obj_list))
        for obj_idx, obj_id in enumerate(obj_list):
            obj_adds_auc[obj_idx] = cal_auc(obj_adds_list[obj_id]) / 100.
            print('Obj {}: {}'.format(obj_id,obj_adds_auc[obj_idx]))
        print('tot. avg.: {:.4f}'.format(np.mean(obj_adds_auc)))
        print('std.: {:.4f}'.format(np.std(obj_adds_auc)))

        print('-----ADD(S) AUC-----')
        obj_add_s_auc = np.zeros(len(obj_list))
        for obj_idx, obj_id in enumerate(obj_list):
            obj_add_s_auc[obj_idx] = cal_auc(obj_add_s_list[obj_id]) / 100.
            print('Obj {}: {}'.format(obj_id,obj_add_s_auc[obj_idx]))
        print('tot. avg.: {:.4f}'.format(np.mean(obj_add_s_auc)))
        print('std.: {:.4f}'.format(np.std(obj_add_s_auc)))

        print('-----ADD SCORE-----')
        for obj_idx, obj_id in enumerate(obj_list):
            print('Obj {}: {}'.format(obj_id,success_count_add[obj_id]))
        mean, std = get_dict_stats(success_count_add)
        print('tot. avg.: {:.4f}'.format(mean))
        print('std.: {:.4f}'.format(std))

        print('-----REP SCORE-----')
        for obj_idx, obj_id in enumerate(obj_list):
            print('Obj {}: {}'.format(obj_id,success_count_rep[obj_id]))
        mean, std = get_dict_stats(success_count_rep)
        print('tot. avg.: {:.4f}'.format(mean))
        print('std.: {:.4f}'.format(std))

        print('-----RRE SCORE-----')
        for obj_idx, obj_id in enumerate(obj_list):
            print('Obj {}: {}'.format(obj_id,success_count_rre[obj_id]))
        mean, std = get_dict_stats(success_count_rre)
        print('tot. avg.: {:.4f}'.format(mean))
        print('std.: {:.4f}'.format(std))

        print('-----RTE SCORE-----')
        for obj_idx, obj_id in enumerate(obj_list):
            print('Obj {}: {}'.format(obj_id,success_count_rte[obj_id]))
        mean, std = get_dict_stats(success_count_rte)
        print('tot. avg.: {:.4f}'.format(mean))
        print('std.: {:.4f}'.format(std))

    perfs['adds_auc'] = {k:v for k,v in zip(obj_list,obj_adds_auc.tolist())}
    perfs['add(s)_auc'] = {k:v for k,v in zip(obj_list,obj_add_s_auc.tolist())}
    
    with open(perf_file,'w') as f:
        json.dump(perfs, f) 

if __name__ == '__main__':
    main()
