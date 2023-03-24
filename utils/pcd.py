import torch
import pandas
import numpy as np
from lib.csrc.fps import fps_utils
from utils.misc import np_normalize
from point_cloud_utils import mesh_mean_and_gaussian_curvatures
import MinkowskiEngine as ME
from scipy.optimize import linear_sum_assignment

def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')

def random_3d_rotation():
    # Generate rotation
    
    angles = np.zeros(3)
    angle_idx = np.random.choice([0,1,2],size=1)
    angles[angle_idx] = np.random.uniform() * np.pi * 0.5 
    
    cosx = np.cos(angles[0])
    cosy = np.cos(angles[1])
    cosz = np.cos(angles[2])
    sinx = np.sin(angles[0])
    siny = np.sin(angles[1])
    sinz = np.sin(angles[2])
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz
    return R_ab

def pcd_outlier_removal(pcd : torch.tensor, k : int = 5, dist : float = 2) -> torch.Tensor:

    dist_all = pdist(pcd, pcd)
    dist_topk = torch.topk(dist_all,dim=1,k=k,largest=False).values
    dist_near = torch.all(dist_topk <= dist, dim=1)
    near_idxs = torch.nonzero(dist_near).flatten() 
    return near_idxs

def get_pcd_curvature(vertex: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    
    curv, _ = mesh_mean_and_gaussian_curvatures(vertex, faces)
    curv[np.isnan(curv)] = 0        
    # take absolute value and normalize curvature
    curv = np_normalize(np.sqrt(np.abs(curv)))        

    return curv


def get_scene_mesh(depth: torch.Tensor) -> torch.Tensor:

    h,w = depth.shape[:2]
    vert_idxs = torch.arange(0,h*w)
    vert_idxs = vert_idxs.reshape(depth.shape[:2])

    idx0 = vert_idxs[:-1,:-1]
    idx1 = vert_idxs[:-1,:-1] + 1
    idx2 = vert_idxs[:-1,:-1] + 640
    idx3 = vert_idxs[:-1,:-1] + 641

    idx0,idx1,idx2,idx3 = idx0.flatten(),idx1.flatten(),idx2.flatten(),idx3.flatten()
    meshes1 = torch.stack((idx0, idx1, idx2),dim=1)
    meshes2 = torch.stack((idx1, idx2, idx3),dim=1)
    meshes = torch.cat((meshes1, meshes2),dim=0)
    return meshes

def get_pcd_bbox(pcd : torch.Tensor, margin : float =0. ) -> torch.Tensor:

    min_x, max_x = torch.min(pcd[:, 0]) - margin, torch.max(pcd[:, 0]) + margin
    min_y, max_y = torch.min(pcd[:, 1]) - margin, torch.max(pcd[:, 1]) + margin
    min_z, max_z = torch.min(pcd[:, 2]) - margin, torch.max(pcd[:, 2]) + margin

    bbox_3d = torch.tensor([
        [min_x,min_y,min_z],
        [max_x,min_y,min_z],
        [min_x,max_y,min_z],
        [max_x,max_y,min_z],
        [min_x,min_y,max_z],
        [max_x,min_y,max_z],
        [min_x,max_y,max_z],
        [max_x,max_y,max_z]
    ])

    return bbox_3d

def crop_pcd(pcd : torch.Tensor, bbox : torch.Tensor) -> torch.Tensor:

    max_x, min_x = torch.max(bbox[:,0]), torch.min(bbox[:,0])
    max_y, min_y = torch.max(bbox[:,1]), torch.min(bbox[:,1])
    max_z, min_z = torch.max(bbox[:,2]), torch.min(bbox[:,2])

    idx_x = torch.bitwise_and(pcd[:,0] <= max_x, pcd[:,0] >= min_x)
    idx_y = torch.bitwise_and(pcd[:,1] <= max_y, pcd[:,1] >= min_y)
    idx_z = torch.bitwise_and(pcd[:,2] <= max_z, pcd[:,2] >= min_z)

    mask = torch.all(torch.stack((idx_x, idx_y, idx_z),dim=1),dim=1)

    return mask

def transform_pcd(pcd, r, t):
    '''
    Rotates a batch of (B,N,3) point clouds
    '''
    pcd = pcd.to(torch.double).transpose(1,2)
    r = r.to(torch.double)
    t = t.to(torch.double)
    rotated = torch.bmm(r, pcd) + t.unsqueeze(2)
    
    return rotated.transpose(1,2).to(torch.float)

def np_transform_pcd(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed

def project_points(v, k):
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)
    
    assert len(v.shape) == 2, '  wrong dimension, expexted shape 2.'
    assert v.shape[1] == 3, ' expected 3d points, got ' + str(v.shape[0]) + ' ' + str(v.shape[1]) +'d points instead.' 

    p = np.matmul(k, v.T)
    p[0] = p[0] / (p[2] + 1e-12)
    p[1] = p[1] / (p[2] + 1e-12)
    
    return p[:2].T

def me_quantize_pcd(pcd, voxel_size, extract_features=True):
    '''
    Quantizes with Minkowski Engine a (N,D) point cloud.
    If extract_features is True, the indexes (D-3..D) are treated as features
    '''

    coords = pcd[:,:3]
    if extract_features:
        features = pcd[:,3:]
    else:
        features = torch.ones((pcd.shape[0],1), device=pcd.device)
    
    pcd_coords, pcd_feats = ME.utils.sparse_quantize(
        features=features,
        coordinates=coords,
        quantization_size=voxel_size
    )

    return pcd_coords, pcd_feats.to(torch.float32)

def lift_pcd(depth, camera, include_feats=True):

    '''
    Given a depth image and relative camera, lifts the depth to a point cloud.
    If depth has 4 channel, the last 3 are used as RGB and an RGB point cloud is produced in output.
    Image size is implicitly given as depth image size.
    '''

    image_h, image_w, n_channels = depth.shape

    d = depth[:,:,0]

    # make coordinate grid
    xs = torch.linspace(0, image_w-1, steps=image_w)
    ys = torch.linspace(0, image_h-1, steps=image_h)
    ymap, xmap = torch.meshgrid(xs,ys, indexing='xy')

    ymap = ymap.to(d.device).flatten().to(torch.float32)
    xmap = xmap.to(d.device).flatten().to(torch.float32)

    # get camera info
    fx = camera[0]
    fy = camera[4]
    cx = camera[2]
    cy = camera[5]
    pt2 = d.flatten()
    # perform lifting
    pt0 = (ymap - cx) * pt2 / fx
    pt1 = (xmap - cy) * pt2 / fy
    pcd_depth = torch.stack((pt0, pt1, pt2),dim=1) 
    
    if include_feats:
        r = (depth[:,:,1]).clone().flatten()
        g = (depth[:,:,2]).clone().flatten()
        b = (depth[:,:,3]).clone().flatten()
        rgb = torch.stack((r,g,b),dim=1) / 255.
        if n_channels == 5:
            filt = (depth[:,:,4]).clone().flatten().unsqueeze(1)
            pcd_depth = torch.cat((pcd_depth, rgb, filt),dim=1)
        elif n_channels == 6:
            filt = (depth[:,:,5]).clone().flatten().unsqueeze(1)
            mask = (depth[:,:,4]).clone().flatten().unsqueeze(1)
            pcd_depth = torch.cat((pcd_depth, rgb, mask, filt),dim=1)
        else:
            raise RuntimeError("Unrecognized {} channels".format(n_channels))

    return pcd_depth

def sample_pcd(pcd : torch.Tensor, n_points : int) -> torch.Tensor:
    
    '''
    Performs pcd subsampling
    '''

    pcd_points = pcd.shape[0]

    if pcd_points > n_points:
        uniform_dist = torch.ones(pcd_points, dtype=torch.float).to(pcd.device)
        inds_choosen = torch.multinomial(uniform_dist, n_points, replacement=False)
        pcd = pcd[inds_choosen]

    return pcd

def sample_pcd_fps(pcd, ratio):
    
    num_points = int(pcd.shape[0]/ratio)
    fps_points = fps_utils.farthest_point_sampling(pcd, num_points, False)
    return fps_points

def compute_corrs_nn(pcd0, pcd1, pcd1_idxs, pos_threshold=2., max_corr=1000):
    '''
    Compute correspondences between two point clouds (N,3) and (M,3) tensors.
    Positive match is given by nearest point within the threshold. 
    Returns paired list of correspondences
    '''
    pcd0, pcd1 = pcd0.clone().to(torch.float), pcd1.clone().to(torch.float)

    dist = torch.cdist(pcd0, pcd1[pcd1_idxs], p=2.0)
    pcd0_idxs = torch.arange(pcd0.shape[0])
    min_dist = torch.amin(dist, dim=1)
    pcd1_subidxs = torch.argmin(dist, dim=1)
    valid_corr = torch.nonzero(min_dist < pos_threshold)
    pcd1_idxs = pcd1_idxs[pcd1_subidxs]
    final_corrs = torch.stack((pcd0_idxs[valid_corr.squeeze(1)], pcd1_idxs[valid_corr.squeeze(1)]),dim=1)

    if final_corrs.shape[0] > max_corr:
        uniform_dist = torch.ones(final_corrs.shape[0], dtype=float).to(final_corrs.device)
        choosen = torch.multinomial(uniform_dist, max_corr, replacement=False)
        final_corrs = final_corrs[choosen]
    
    return final_corrs


def compute_corrs_hungarian(pcd0, pcd1, pos_threshold=2., max_corr=1000):
    
    '''
    As above, but computes correspondences by solving the linear assignment problem
    '''

    pcd0, pcd1 = pcd0.clone().to(torch.float), pcd1.clone().to(torch.float)

    dist = torch.cdist(pcd0, pcd1)
    min0_dist = torch.amin(dist, dim=1)
    min1_dist = torch.amin(dist, dim=0)
    corr0 = torch.nonzero(min0_dist <= pos_threshold).squeeze(1)
    corr1 = torch.nonzero(min1_dist <= pos_threshold).squeeze(1)

    filt_pcd0, filt_pcd1 = pcd0[corr0], pcd1[corr1]
    filt_pcd0_idxs = torch.arange(0,pcd0.shape[0])[corr0]
    filt_pcd1_idxs = torch.arange(0,pcd1.shape[0])[corr1]

    id0, id1 = linear_sum_assignment(torch.cdist(filt_pcd0, filt_pcd1))
    corrs = torch.tensor(np.stack((id0,id1),axis=1))
    final_corrs = torch.stack((filt_pcd0_idxs[corrs[:,0]], filt_pcd1_idxs[corrs[:,1]]),dim=1)

    if final_corrs.shape[0] > max_corr:
        uniform_dist = torch.ones(final_corrs.shape[0], dtype=float).to(final_corrs.device)
        choosen = torch.multinomial(uniform_dist, max_corr, replacement=False)
        final_corrs = final_corrs[choosen]

    return final_corrs