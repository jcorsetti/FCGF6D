import argparse
import os
from utils.misc import boolean_string
from os import readlink

def parse_test_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, default='exp00', help='Experiment name')
    parser.add_argument('--exp_root', type=str, default=readlink('exp_data'), help='Root to models folder for storing experiments')
    parser.add_argument('--oracle', type=str, default=None, help='Eventual oracle for detection')
    parser.add_argument('--icp', type=boolean_string, default=False, help='If true, use ICP to refine pose')
    parser.add_argument('--solver', type=str, default='ransac', help='Solver type for pose, one of [ransac, teaser]')
    parser.add_argument('--seed', type=int, default=1, help='Set seed')

    # Train
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--model_points', type=int, default=4096, help='Number of sampled points per object model')
    parser.add_argument('--depth_points', type=int, default=50000, help='Number of sampled points per depth point cloud')
    parser.add_argument('--voxel_size', type=float, default=2., help='Minkowski quantization sampling size, in mm')
    parser.add_argument('--obj', type=str, default='all', help='Object test splits')

    # Data
    parser.add_argument('--dataset', type=str, default='lmo', help='Name of dataset')
    parser.add_argument('--split', type=str, default='test', help='Data split where to perform inference')
    parser.add_argument('--save_results', type=boolean_string, default=False, help='Visualize segmentations and projections results')
    parser.add_argument('--save_only', type=boolean_string, default=False, help='Only saves pcds without doing registration')

    # Model
    parser.add_argument('--checkpoint', type=str, help='Name of the checkpoint file')
    parser.add_argument('--arch', type=str, default='mink34', help='Type of net used for point cloud feature extraction')
    parser.add_argument('--dim_features', type=int, default=32, help='Dimension of features vector in output')
    parser.add_argument('--first_kernel', type=int, default=5, help='Dimension of first minknet kernel')


    args = parser.parse_args()
    args.path = os.readlink('data_{}'.format(args.dataset))

    return args


def parse_train_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, default='exp00', help='Experiment name')
    parser.add_argument('--device', default='cuda', help='Computational device')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--exp_root', type=str, default=readlink('exp_data'), help='Root to models folder for storing experiments')
    parser.add_argument('--profile', type=boolean_string, default=False, help='Use PL Advanced Profiler')
    parser.add_argument('--oracle', type=str, default=None, help='Eventual oracle for detection')

    # Train
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Manual epoch number (useful on restarts)')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--freq_train', type=int, default=1, help='Frequence at which log training')
    parser.add_argument('--freq_valid', type=int, default=10, help='Frequence at which log validation')
    parser.add_argument('--freq_save', type=int, default=10, help='Frequence at which save the model')
    parser.add_argument('--flag_resume', action='store_true', help='Flag to resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Name of the checkpoint file')
    
    # Generalization
    parser.add_argument('--train_obj', type=str, default='all', help='Training split. One of lm, lmo, lm-only')
    parser.add_argument('--valid_obj', type=str, default='all', help='Validation split. One of lm, lmo, lm-only')

    # Data
    parser.add_argument('--dataset', type=str, default='lmo', help='Name of dataset')
    parser.add_argument('--split_train', type=str, default='train_pbr',help='Data split where to perform training')
    parser.add_argument('--split_valid', type=str, default='test',help='Data split where to perform validation')
    parser.add_argument('--model_points', type=int, default=4096, help='Number of sampled points per object model')
    parser.add_argument('--depth_points', type=int, default=50000, help='Number ofsampled points per depth point cloud')
    parser.add_argument('--voxel_size', type=float, default=2., help='Minkowski quantization sampling size, in mm')
    parser.add_argument('--corr_th', type=float, default=2., help='Threshold for correspondences')
    
    # Augs
    parser.add_argument('--aug_erase', type=boolean_string, default=False, help='If True, applies guided random erasing on scene')
    parser.add_argument('--aug_rgb', type=boolean_string, default=False, help='If True, apply augmentations to RGB data')
     
    # Model
    parser.add_argument('--arch', type=str, default='mink34', help='Type of net used for point cloud feature extraction')
    parser.add_argument('--loss', type=str, default='hc_kernel', help='Type of loss for metric learning')
    parser.add_argument('--dim_features', type=int, default=32, help='Dimension of features vector in output')
    parser.add_argument('--first_kernel', type=int, default=5, help='Dimension of first kernel')

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='Scheduler type')
    parser.add_argument('--step', type=int, default=None, help='Step for the lr decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor of lr decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--w_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--mu1', type=float, default=1., help='Weight of positive loss')
    parser.add_argument('--mu2', type=float, default=0.6, help='Weight of negative object loss')
    parser.add_argument('--mu3', type=float, default=0.4, help='Weight of negative scene loss')

    parser.add_argument('--pos_margin', type=float, default=0.1, help='Distance margin for positive features')
    parser.add_argument('--neg_margin', type=float, default=10., help='Distance margin for negative features')
    parser.add_argument('--kernel_th_object', type=float, default=0.1, help='Diameter scale for negative object loss')
    parser.add_argument('--kernel_th_scene', type=float, default=0.1, help='Diameter scale for negative scene loss')
    
    args = parser.parse_args()
    args.path = os.readlink('data_{}'.format(args.dataset))

    return args

