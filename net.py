import torch
from models.minkunet import MinkUNet14, MinkUNet34, MinkUNet50, MinkUNet101
from torch.nn.functional import normalize
import MinkowskiEngine as ME
import numpy as np

class Metric6DNet(torch.nn.Module):
    def __init__(self, arch, in_channels, out_channels, first_kernel, normalize=False, D=3):
        
        self.normalize = normalize
        super().__init__()


        if arch == 'mink14':
            self.obj_net =  MinkUNet14(in_channels, out_channels, first_kernel, D)
            self.scene_net = MinkUNet14(in_channels, out_channels, first_kernel, D)

        elif arch == 'mink34':
            self.obj_net =  MinkUNet34(in_channels, out_channels, first_kernel, D)
            self.scene_net = MinkUNet34(in_channels, out_channels, first_kernel, D)

        elif arch == 'mink50':
            self.obj_net =  MinkUNet50(in_channels, out_channels, first_kernel, D)
            self.scene_net = MinkUNet50(in_channels, out_channels, first_kernel, D)

        elif arch == 'mink101':
            self.obj_net =  MinkUNet101(in_channels, out_channels, first_kernel, D)
            self.scene_net = MinkUNet101(in_channels, out_channels, first_kernel, D)

        else:
            raise RuntimeError('Architecture {} not supported.'.format(arch))


    def forward(self, xs):

        obj, scene = xs
        obj_feats = self.obj_net(obj)
        scene_feats = self.scene_net(scene)

        if self.normalize:
            obj_feats = ME.SparseTensor(obj_feats.F/torch.norm(obj_feats.F, p=2,dim=1,keepdim=True), obj_feats.C)
            scene_feats = ME.SparseTensor(scene_feats.F/torch.norm(scene_feats.F+1e-12, p=2,dim=1,keepdim=True), scene_feats.C)

        return (obj_feats, scene_feats)