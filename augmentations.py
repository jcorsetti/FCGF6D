import torch
import random
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter
from utils.pcd import random_3d_rotation
from models.common import se3

class random_scene_rotation:
    '''
    Applies a random rotation only to some specific points in the pcd
    '''

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pcd, gt_pose, obj_id):
        
        if np.random.random() <= self.prob:

            still_idxs = torch.nonzero(pcd[:, -1] != obj_id)
            rotate_idxs = torch.nonzero(pcd[:, -1] == obj_id)

            rotate_pcd = pcd[rotate_idxs, :3].squeeze(1)
            rotate_feats = pcd[rotate_idxs, 3:].squeeze(1) # take features

            canon_points = se3.torch_transform(se3.torch_inverse(gt_pose).squeeze(0), rotate_pcd)

            R = torch.tensor(random_3d_rotation())
            gt_pose[:3,:3] = R @ gt_pose[:3,:3]

            # concatenate back features
            rotated_pts = torch.cat((se3.torch_transform(gt_pose, canon_points), rotate_feats), dim=1)
            pcd = torch.cat((pcd[still_idxs].squeeze(1), rotated_pts), dim=0)

        return pcd, gt_pose


class random_guided_erase:
    """
    Guided random erasing. The range depends on the given diameter, and the points from which the center is selected are given.
    """
    
    def __init__(self, prob=0.5, erase_size=0.30):
        self.erase_size = erase_size # size of range around random point to erase, relative to the pcd diameter
        self.prob = prob
    
    def __call__(self, pcd, diameter, obj_id):

        # take random point on the list of allowed

        if np.random.random() <= self.prob:

            pts = torch.nonzero(pcd[:, -1] == obj_id).squeeze(1)
            # may happen due to sampling
            if pts.shape[0] > 0:

                erasing_center = pcd[pts[np.random.randint(0, pts.shape)],:3]
                xyz = pcd[:,:3]
                
                dists = torch.abs(xyz - erasing_center).sum(1)
                min_dist = self.erase_size * diameter
                retained = torch.nonzero(dists > min_dist).squeeze(1)
                pcd = pcd[retained]
        
        return pcd

class random_erase:
    """ 
    Random erasing of a point cloud based on fraction of diameter size 
    """
    def __init__(self, prob=0.5, erase_size=0.10):
        self.erase_size = erase_size # size of range around random point to erase, relative to the pcd diameter
        self.prob = prob

    def __call__(self, pcd):
        
        if np.random.random() <= self.prob:
            pcd = pcd.numpy()

            n_points = pcd.shape[0]

            erasing_center = np.random.randint(0, n_points)
            pcd_diameter = np.linalg.norm(np.amax(pcd,axis=0) - np.amin(pcd,axis=0),ord=2)
            
            dists = np.linalg.norm(pcd[:,:3] - pcd[erasing_center,:3],axis=1, ord=2)
            min_dist = self.erase_size * pcd_diameter
            retained = np.nonzero(dists > min_dist)
            pcd = pcd[retained]

        return torch.tensor(pcd)

class random_jitter:
    """ generate perturbations """
    def __init__(self, noise_std=0.01, clip=0.05):
        self.noise_std = noise_std
        self.clip = clip

    def __call__(self, pcd):

        noise = np.clip(np.random.normal(0.0, scale=self.noise_std, size=(pcd.shape[0], 3)), a_min=-self.clip, a_max=self.clip)
        pcd[:,:3] += noise  # Add noise to xyz

        return pcd

class shuffle_points:
    """Shuffles the order of the points"""
    def __call__(self, pcd):

        
        idxs = torch.randperm(pcd.shape[0])
        rand_pcd = pcd[idxs]

        return rand_pcd

class to_tensor(object):
    def __call__(self, sample):
        # return (F.to_tensor(sample[0]), sample[1])
        return(
            torch.tensor(np.asarray(sample[0]).transpose(2, 0, 1), dtype=torch.float32).div(255),
            torch.tensor(np.asarray(sample[1]), dtype=torch.long),
            torch.tensor(np.asarray(sample[2]), dtype=torch.float32))

class resize(object):
    def __init__(self, size_out):
        assert isinstance(size_out, (int, tuple))
        self.size_out = size_out
    
    def __call__(self, sample):
        # TODO Check if this works for int
        height_out, width_out = self.size_out
        
        return (
            F.resize(sample[0], (height_out, width_out), interpolation=F.InterpolationMode.BILINEAR),
            F.resize(sample[1], (height_out, width_out), interpolation=F.InterpolationMode.NEAREST),
            F.resize(sample[2], (height_out, width_out), interpolation=F.InterpolationMode.BILINEAR)
        )

class random_blur(object):
    def __init__(self, prob=.5, ksize=7):
        self.prob = prob
        self.ksize = ksize
    def __call__(self, sample):
        if random.random() < self.prob:
            return (
                F.gaussian_blur(sample[0], kernel_size=self.ksize), 
                sample[1], 
                sample[2])
        else:
            return sample

class random_brightness(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return (
                ColorJitter(brightness=.25, contrast=.0, saturation=.0, hue=.00)(sample[0]),
                sample[1],
                sample[2])
        else:
            return sample

class color_jitter(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return (
                ColorJitter(brightness=.125, contrast=.5, saturation=.5, hue=.05)(sample[0]),
                sample[1],
                sample[2])
        else:
            return sample

class normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        return (
            F.normalize(sample[0], mean=self.mean, std=self.std),
            sample[1],
            sample[2]
        )