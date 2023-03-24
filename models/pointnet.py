import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm


class my_pn_feat(nn.Module):
    def __init__(self, stn_type='qnet', ftn_type=None):
        super().__init__()

        self.stn_type = stn_type
        self.ftn_type = ftn_type

        if self.stn_type == 'tnet':
            self.stn = tnet(3)
        elif self.stn_type == 'qnet':
            self.stn = qnet(3)
        else:
            self.stn = None

        if self.ftn_type == 'tnet':
            self.ftn = tnet(64)
        else:
            self.ftn = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True)  # TODO In the original implementation no activation is used here
        )
    
    # expects BS,3,N
    def forward(self, verts):

        # Spatial transformer
        if self.stn_type == 'tnet':
            trans_stn = self.stn(verts)
            verts = torch.bmm(trans_stn, verts)
        elif self.stn_type == 'qnet':
            quat = self.stn(verts)
            angle_axis = tgm.quaternion_to_angle_axis(quat)
            _trans_stn = tgm.angle_axis_to_rotation_matrix(angle_axis)
            trans_stn = _trans_stn[:, :3, :3]
            verts = torch.bmm(trans_stn, verts)
        else:
            trans_stn = None

        x = verts
        #x = torch.cat((verts, feats), dim=1)
        n_verts = x.shape[2]

        x = self.conv1(x)

        x_skip = self.conv2(x)

        # Feature transformer
        if self.ftn_type == 'tnet':
            trans_ftn = self.ftn(x_skip)
            x_skip = torch.bmm(trans_ftn, x_skip)
        else:
            trans_ftn = None

        x = self.conv3(x_skip)
        x = self.conv4(x)
        x_feat = self.conv5(x)  # (bs, 1024, n_verts)

        return x_feat


class pn_1layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        return x


class tnet(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.ch, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            #nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self._init_last_layer()

    def _init_last_layer(self):
        self.fc3 = nn.Linear(256, self.ch ** 2, bias=True)


        # torch.nn.init.zeros_(self.fc3.bias)

        # torch.nn.init.zeros_(self.fc3.weight)
        # torch.nn.init.zeros_(self.fc3.bias)
        # self.fc3.bias.data = torch.eye(self.dim_in).view(self.dim_in ** 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        mx, _ = torch.max(x, 2, keepdim=True)
        x = mx.view(-1, 1024).contiguous()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self._forward_last_layer(x)
        return x

    def _forward_last_layer(self, x):
        x = self.fc3(x)
        x = x + torch.eye(self.ch, device='cuda').view(1, self.ch ** 2).repeat(x.size()[0], 1)  # TODO Is the sum of identity necessary?
        x = x.view(-1, self.ch, self.ch).contiguous()
        return x


class qnet(tnet):
    def _init_last_layer(self):
        self.fc3 = nn.Linear(256, 4, bias=True)
        torch.nn.init.zeros_(self.fc3.bias)

    def _forward_last_layer(self, x):
        quat = self.fc3(x)
        quat = quat + torch.tensor([1, 0, 0, 0], device='cuda').repeat(quat.size()[0], 1)
        quat = F.normalize(quat, p=2, dim=1)
        return quat


class pn_feat(nn.Module):
    def __init__(self, args, flag_global_feat=True):
        super().__init__()

        self.args = args
        self.stn_type = args.stn_type
        self.ftn_type = args.ftn_type
        self.flag_global_feat = flag_global_feat

        if self.stn_type == 'tnet':
            self.stn = tnet(3)
        elif self.stn_type == 'qnet':
            self.stn = qnet(3)
        else:
            self.stn = None

        if self.ftn_type == 'tnet':
            self.ftn = tnet(64)
        else:
            self.ftn = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, args.dim, 1, bias=False),
            nn.BatchNorm1d(args.dim),
            # nn.ReLU(inplace=True)  # TODO In the original implementation no activation is used here
        )

    def forward(self, verts, feats):

        # Spatial transformer
        if self.stn_type == 'tnet':
            trans_stn = self.stn(verts)
            verts = torch.bmm(trans_stn, verts)
        elif self.stn_type == 'qnet':
            quat = self.stn(verts)
            angle_axis = tgm.quaternion_to_angle_axis(quat)
            _trans_stn = tgm.angle_axis_to_rotation_matrix(angle_axis)
            trans_stn = _trans_stn[:, :3, :3]
            verts = torch.bmm(trans_stn, verts)
            # _verts = tgm.convert_points_to_homogeneous( \
            #     verts.contiguous().transpose(2, 1)).contiguous().transpose(2, 1)
            # _verts = torch.bmm(_trans, _verts)
            # verts = tgm.convert_points_from_homogeneous( \
            #     _verts.contiguous().transpose(2, 1)).contiguous().transpose(2, 1)
        else:
            trans_stn = None

        x = torch.cat((verts, feats), dim=1)
        n_verts = x.shape[2]

        x = self.conv1(x)
        x_skip = self.conv2(x)

        # Feature transformer
        if self.ftn_type == 'tnet':
            trans_ftn = self.ftn(x_skip)
            x_skip = torch.bmm(trans_ftn, x_skip)
        else:
            trans_ftn = None

        x = self.conv3(x_skip)
        x = self.conv4(x)
        x_feat = self.conv5(x)  # (bs, 1024, n_verts)

        # Pooling
        # mx, _ = torch.max(x_feat, 2, keepdim=True)  # https://github.com/fxia22/pointnet.pytorch
        # x = mx.view(-1, 1024)
        x = F.adaptive_max_pool1d(x_feat, 1).squeeze(2)  # From DGCNN implementation
        # TODO Add mean pooling

        if self.flag_global_feat:
            x_class = x
            x_segm = None
        else:
            x_class = x
            x_segm = x.view(-1, self.args.dim, 1).repeat(1, 1, n_verts)
            x_segm = torch.cat([x_skip, x_segm], dim=1)

        return x_class, x_segm, trans_stn, trans_ftn


class pn_class(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.feat = pn_feat(args, flag_global_feat=True)

        self.fc1 = nn.Sequential(
            nn.Linear(args.dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=args.p_drop)  # TODO
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.Dropout(p=args.p_drop),  # TODO
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=args.p_drop)  # TODO
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, args.n_classes, bias=True)
        )

    def forward(self, x):
        verts = x[:, :3, :]
        feats = x[:, 3:, :]

        x_class, x_segm, trans_stn, trans_ftn = self.feat(verts, feats)

        x = self.fc1(x_class)
        y = self.fc2(x)
        x = self.fc3(y)

        return x, y, trans_stn, trans_ftn


class pn_segm(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.feat = pn_feat(args, flag_global_feat=False)

        self.conv1 = nn.Sequential(
            nn.Conv1d(args.dim + 64, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, args.n_classes, 1, bias=True)
        )

    def forward(self, x):
        verts = x[:, :3, :]
        feats = x[:, 3:, :]
        x_class, x_segm, trans_stn, trans_ftn = self.feat(verts, feats)
        x = self.conv1(x_segm)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return None, x, trans_stn, trans_ftn


# Positional encoding
# TODO Concatenate the coordinates themselves to the tensor
# L = 10
# _verts = torch.zeros((verts.shape[0], 6 * L, verts.shape[2]), device='cuda')
# for i in range(3):
#     p = verts[:, i, :]
#     for j in range(L):
#         _verts[:, L * i + j, :] = torch.sin(torch.tensor(np.power(2, j) * np.pi, device='cuda') * p)
#     for j in range(10):
#         _verts[:, 30 + L * i + j] = torch.cos(torch.tensor(np.power(2, j) * np.pi, device='cuda') * p)
# xin = torch.cat((_verts, feats), dim=1)