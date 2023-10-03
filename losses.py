import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils.pcd import pdist

class hc_kernel_loss(nn.Module):
    '''
    Modification of Hardest Contrastive Loss as implemented in FCGF (https://arxiv.org/abs/1411.4038)
    Negatives are sampled from the same point clouds instead of the other one
    '''

    def __init__(self, th_kernel_obj, th_kernel_scene, th_pos=0.2, th_neg=10.):
        super(hc_kernel_loss, self).__init__()
        
        self.th_scale_obj = th_kernel_obj
        self.th_scale_scene = th_kernel_scene
        self.threshold_pos = th_pos
        self.threshold_neg = th_neg
        self.scene_subsample = 10000

    def forward(self, obj, scene, positive_pairs, obj_diameters, obj_ids):
        
        pos_loss, pos_dist = torch.tensor(0.,requires_grad=True),torch.tensor(0.,requires_grad=True)
        neg_loss_o, neg_dist_o = torch.tensor(0.,requires_grad=True),torch.tensor(0.,requires_grad=True)
        neg_loss_s, neg_dist_s = torch.tensor(0.,requires_grad=True),torch.tensor(0.,requires_grad=True)

        # first dimension is batch: get batch size
        obj_list = torch.unique(positive_pairs.C[:,0]).cpu().numpy()

        BS = len(obj_list)

        for i in obj_list:
            
            # get current pcd of object and scene
            obj_F_i, obj_C_i = obj.features_at(i), obj.coordinates_at(i)
            scene_F_i, scene_C_i = scene.features_at(i), scene.coordinates_at(i)
            corr_i = positive_pairs.coordinates_at(i)
            
            diameter_i = obj_diameters[i]
            obj_id = int(obj_ids[i].item())
            
            # and correspondences as well
            pos_ind_obj = corr_i[:, 0].long()
            pos_ind_scene = corr_i[:, 1].long()
            pos_loss_i, pos_dist_i = self.positive_loss(obj_F_i, scene_F_i, pos_ind_obj, pos_ind_scene)

            # scene is subsampled to reduce memory occupancy
            neg_loss_s_i, neg_dist_s_i = self.negative_loss(scene_C_i, scene_F_i, pos_ind_scene, diameter_i, self.th_scale_scene, subsample=self.scene_subsample)
            
            neg_loss_o_i, neg_dist_o_i = self.negative_loss(obj_C_i, obj_F_i, pos_ind_obj, diameter_i, self.th_scale_obj, subsample=None)

            # accumulate losses and distances in batch
            pos_loss = pos_loss + pos_loss_i
            neg_loss_o = neg_loss_o + neg_loss_o_i
            neg_loss_s = neg_loss_s + neg_loss_s_i
            pos_dist = pos_dist + pos_dist_i
            neg_dist_o = neg_dist_o + neg_dist_o_i
            neg_dist_s = neg_dist_s + neg_dist_s_i

        losses = {
            'positive' : pos_loss/BS if BS > 0 else torch.tensor(0.,requires_grad=True),
            'negative obj' : neg_loss_o/BS if BS > 0 else torch.tensor(0.,requires_grad=True),
            'negative scene' : neg_loss_s/BS if BS > 0 else torch.tensor(0.,requires_grad=True)
        }

        distances = {
            'positive' : pos_dist/BS if BS > 0 else 0.,
            'negative obj' : neg_dist_o/BS if BS > 0 else 0.,
            'negative scene' : neg_dist_s/BS if BS > 0 else 0.
        }

        return losses, distances

    def positive_loss(self, feats0, feats1, idxs0, idxs1):
        
        feats0, feats1 = feats0[idxs0], feats1[idxs1]
        #get features and compute positive loss
        pos_dist = torch.sqrt((feats0 - feats1).pow(2).sum(1))
        pos_loss = F.relu(pos_dist - self.threshold_pos).pow(2)

        return pos_loss.mean(0), pos_dist.mean(0)

    def negative_loss(self, pcd_coords, pcd_feats, pos_idxs, obj_diameter, th_kernel, subsample=None):

        # correspondences used as anchors
        sub_coords, sub_feats = pcd_coords[pos_idxs], pcd_feats[pos_idxs]
        n_points = pcd_feats.shape[0]
        
        pcd_min_F, pcd_min_C = pcd_feats.clone(), pcd_coords.clone()

        if subsample is not None:

            if n_points > subsample:
                uniform_dist = torch.ones(n_points, dtype=float)
                choosen_idxs = torch.multinomial(uniform_dist, subsample, replacement=False)
                pcd_min_F = pcd_min_F[choosen_idxs]
                pcd_min_C = pcd_min_C[choosen_idxs]

        # compute diameter of point cloud, used as scale for the kernel size

        s2s_geom_dist = pdist(sub_coords, pcd_min_C, 'L2')
        s2s_feat_dist = pdist(sub_feats, pcd_min_F, 'L2')

        # put infinite distance between the points geometrically near to avoid selection!
        s2s_feat_dist = s2s_feat_dist + 1e6 * F.relu(th_kernel*obj_diameter - s2s_geom_dist)
        
        # select points near in the feature space
        neg_dist = torch.amin(s2s_feat_dist, dim=1)
        neg_loss = F.relu(self.threshold_neg - neg_dist).pow(2)

        # return both negative loss and distances
        return neg_loss.mean(), neg_dist.mean()
