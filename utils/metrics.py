import math
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import teaserpp_python
from .pcd import compute_corrs_hungarian
import torch

def mae(v1, v2):
    '''
    Mean Absolute Error between two np.arrays of shape [N]
    '''

    assert v1.shape == v2.shape, 'Must be of the same shape, found {} and {}'.format(v1.shape, v2.shape)
    assert (len(v1.shape) == 1), 'Tensor must have 1 dimension, found {}'.format(v1.shape)
    
    loss = np.sum(np.abs(v1-v2))
    
    return loss

def rmse(v1, v2):
    '''
    Root Mean Square Error between two np.arrays of shape [N]
    '''

    assert v1.shape == v2.shape, 'Must be of the same shape, found {} and {}'.format(v1.shape, v2.shape)
    assert (len(v1.shape) == 1), 'Tensor must have 1 dimension, found {}'.format(v1.shape)
    
    loss = np.sqrt(np.sum(np.square(v1-v2)))
    
    return loss


def rre(R_est, R_gt):
    '''
    Rotation error from BOP toolkit:
    https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py, line 187
    '''
    assert (R_est.shape == R_gt.shape == (3, 3))
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg]
    return error


def rte(t_est, t_gt):
    '''
    Translation error from BOP toolkit:
    https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py, line 205
    '''
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

def cal_auc(add_dis, max_dis=0.1):
    '''
    From PVN3D repo https://github.com/ethnhe/PVN3D/blob/d7c2084e82687e4f5fdd383cf69d3df7c3ee379e/pvn3d/lib/utils/basic_utils.py#L32
    Warning: requires list of ADD(S) in meters!
    '''
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100

def VOCap(rec, prec):
    '''
    From PVN3D repo https://github.com/ethnhe/PVN3D/blob/d7c2084e82687e4f5fdd383cf69d3df7c3ee379e/pvn3d/lib/utils/basic_utils.py#L597
    '''
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def register_pcd(depth, obj, depth_feats, obj_feats, solver='ransac', icp=False):
    '''
    depth: torch.tensor(N,3)
    obj: torch.tensor(M,3)
    
    depth_feats: torch.tensor(N,D)
    obj_feats: torch.tensor(M,D)

    solver: One of [teaser, ransac]
    icp: boolean, if True applies ICP refinement
    '''

    depth = depth.cpu().detach().numpy()
    obj = obj.cpu().detach().numpy()
    depth_feats = depth_feats.cpu().detach().numpy()
    obj_feats = obj_feats.cpu().detach().numpy()

    if solver == 'ransac':
        pred_pose = open3d_ransac(depth, obj, depth_feats, obj_feats)
    elif solver == 'teaser':
        pred_pose = teaser_solver(depth, obj, depth_feats, obj_feats)
    else:
        raise RuntimeError(f"Solver {solver} not implemented.")

    return pred_pose

def teaser_solver(source, target, source_f, target_f) -> np.array:

    # get teaser solver 
    teaser_solver = get_teaser_solver(5.)
    
    # compute corrs with hungarian algs
    if source_f.shape[0] > 0 and target_f.shape[0] > 0:
        corrs = compute_corrs_hungarian(torch.tensor(source_f),torch.tensor(target_f), pos_threshold=6.)
        
        f_corr_t = target[corrs[:,1].numpy(),:].T
        f_corr_s = source[corrs[:,0].numpy(),:].T

        if f_corr_t.shape[1] > 0 and f_corr_s.shape[1] > 0:

            #solve and get solution
            teaser_solver.solve(f_corr_s, f_corr_t)
            solution = teaser_solver.getSolution()

            # re-build pose matrix in 4x4 format as required by ICP
            R_teaser = solution.rotation
            t_teaser = solution.translation
            pred_pose = np.concatenate((R_teaser,np.expand_dims(t_teaser,axis=1)), axis=1)
            pred_pose = np.concatenate((pred_pose, np.asarray([[0.,0.,0.,1.]])),axis=0)

        else:
            pred_pose = np.eye(4)

    else:
        pred_pose = np.eye(4)

    return pred_pose

def open3d_ransac(source, target, source_f, target_f):

    distance_threshold = 2.

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)

    pcd1_feats = o3d.pipelines.registration.Feature()
    pcd1_feats.data = source_f.T

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target)

    pcd2_feats = o3d.pipelines.registration.Feature()
    pcd2_feats.data = target_f.T

    #o3d.utility.random.seed(42)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1, pcd2, pcd1_feats, pcd2_feats, True, 2.,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    return result.transformation

def mutual_NNs(input0, input1):
	'''
	input0: descriptors of point cloud 0 -> N x D (N number of points point cloud 0, D descriptor dimension)
	input1: descriptors of point cloud 1 -> M x D (M number of points point cloud 1, D descriptor dimension)

	mutual_nn: boolean array flagging the mutual nearest neighbours wrt point cloud 0 -> N x 1
	'''

	input1_desc_tree = cKDTree(input1)
	_, nn1_inds = input1_desc_tree.query(input0)

	input0_desc_tree = cKDTree(input0)
	_, nn0_inds = input0_desc_tree.query(input1)

	mutual_nn = list(range(input0.shape[0])) == nn0_inds[nn1_inds]

	# input1[nn1_inds] -> to align the nearest neighbours of input1 wrt input0
	# input1[nn1_inds][mutual_nn] -> to pick the mutual nearest neighbours of input1
	# input0[mutual_nn] -> to pick the mutual nearest neighbours of input0

	return mutual_nn, nn1_inds

def get_teaser_solver(noise_bound):
    
    '''
    Fromhttps://github.com/MIT-SPARK/TEASER-plusplus/blob/master/examples/teaser_python_fpfh_icp/helpers.py
    '''

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver