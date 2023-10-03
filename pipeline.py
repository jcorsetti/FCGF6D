import os
import torch
import math
import numpy as np
from net import Metric6DNet
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from utils.trainer import get_parameters
from datasets import mink_dataset, ME_collation_fn
from losses import hc_kernel_loss
import MinkowskiEngine as ME
from utils.metrics import register_pcd
from torch.utils.data import WeightedRandomSampler
from torch.nn.functional import softmax

class MinkowskiMetricLearning(LightningModule):
    r"""
    This class is a PyTorch Lightning system and contain the core of the major steps made during the training of a NN
    """

    def __init__(self, args, test_model=False):
        r"""
        This functions setup the model and the NN loss
        """
        super().__init__()

        self.args = args

        self.model = self.get_model()

        if not test_model:

            self.loss_fn = self.get_loss_fn()
            
            self.args.loss_weights = {
                'positive' : args.mu1,
                'negative obj' : args.mu2,
                'negative scene' : args.mu3,
                'consistency obj' : args.mu4,
                'consistency scene' : args.mu5
            }

            # Log input arguments
        for arg, value in vars(self.args).items():
            print('{:s} = {:s}'.format(arg, str(value)))
 
    def get_callbacks(self):

        args=self.args
        cpt_callback = ModelCheckpoint(
            dirpath=args.ckpt_out,
            every_n_epochs=args.freq_save,
            save_top_k=-1,
            filename='{epoch:04d}'
        )

        bar_callback = TQDMProgressBar(refresh_rate=50)

        return [cpt_callback, bar_callback]

    def get_logger(self):

        args = self.args
        
        wb_logger = WandbLogger(
            save_dir=args.logs_out,
            project='6Dpose',
            name=args.exp_name,
            offline=True
        )

        self.wb_logger = wb_logger

        return [wb_logger]

    def get_loss_fn(self):

        args = self.args

        if args.loss == 'hc_kernel':
            loss_fn = hc_kernel_loss(args.kernel_th_object, args.kernel_th_scene, args.pos_margin, args.neg_margin)
        else:
            raise RuntimeError('Loss {} not implemented!'.format(args.loss))
    
        return loss_fn

    def get_model(self):
        
        in_feats = 3

        model = Metric6DNet(self.args.arch, in_channels=in_feats, out_channels=self.args.dim_features, first_kernel=self.args.first_kernel, normalize=self.args.normalize, D=3)
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

        return model

    def configure_optimizers(self):
        """
        This functions setup the optimizer and the scheduler
        """
        parameters = get_parameters(self.model)

        if self.args.optim_type == 'SGD':
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.w_decay,
                nesterov=False)
        elif self.args.optim_type == 'Adam':
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.args.lr,
                weight_decay=self.args.w_decay)
        else:
            raise RuntimeError('Optimizer type {} not implemented!'.format(self.args.optim_type))

        self.optimizer = optimizer
        self.args.step = self.args.n_epochs        

        if self.args.scheduler_type == 'step':
            # Learning rate is reduced after 50%, 75% and 90% of samples like in Segdriven original implementation
            part_milestones = [math.ceil(self.args.n_epochs * step) for step in [0.5,0.75,0.9]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=part_milestones,
                gamma=self.args.gamma)
                
        elif self.args.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.step - 1,
                eta_min=self.args.gamma * self.args.lr)

        elif self.args.scheduler_type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.args.gamma
                )
        else:
            raise RuntimeError('Scheduler type {} not implemented!'.format(self.args.scheduler_type))
        
        return [self.optimizer], [scheduler]
    
    def reduce_losses(self, losses):

        weights = self.args.loss_weights

        w_losses = {}
        final_loss = 0.

        for k in losses.keys():
            w_loss = losses[k] * weights[k]
            final_loss = final_loss + w_loss     
            w_losses[k] = w_loss

        return final_loss, w_losses
        
    def forward_batch(self, batch):

        scene_coords = batch[0]
        scene_feats = batch[1]
        scene_filter = batch[2]
        obj_coords = batch[3]
        obj_feats = batch[4]
        corrs_coords = batch[5]
        corrs_feats = batch[6]
        diameters = batch[-1]

        sparse_scene = ME.SparseTensor(scene_feats, scene_coords)
        sparse_obj = ME.SparseTensor(obj_feats, obj_coords)
        sparse_corrs = ME.SparseTensor(corrs_feats, corrs_coords)
 
        out_obj, out_scene = self.model((sparse_obj, sparse_scene))

        return out_obj, out_scene, scene_filter, sparse_corrs, diameters

#################### TRAINING ##########################

    def on_train_start(self) -> None:

        if self.global_rank == 0:
            self.wb_logger.watch(self.model, log_freq=100)
            for k, v in vars(self.args).items():
                self.wb_logger.experiment.config.update({k:v})

        return super().on_train_start()

    def training_step(self, batch, batch_idx):

        obj_ids = batch[11]

        out_obj, out_scene, _, sparse_corrs, diameters = self.forward_batch(batch)

        losses, distances = self.loss_fn(out_obj, out_scene, sparse_corrs, diameters, obj_ids)

        loss, w_losses = self.reduce_losses(losses)
        
        self.structured_log(loss, w_losses, distances, sparse_corrs.shape[0], prefix='train')
        
        return loss

    def on_train_end(self):

        return super().on_train_end()

################## VALIDATION ##########################

    def validation_step(self, batch, batch_idx):

        obj_ids = batch[11]

        out_obj, out_scene, _, sparse_corrs, diameters = self.forward_batch(batch)

        losses, distances = self.loss_fn(out_obj, out_scene, sparse_corrs, diameters, obj_ids)

        loss, w_losses = self.reduce_losses(losses)
        
        self.structured_log(loss, w_losses, distances, sparse_corrs.shape[0], prefix='valid')

        return loss

    def structured_log(self, total_loss, losses, distances, num_corrs, prefix):

        all_metrics = {}
        for k,v in losses.items():
            all_metrics['{}_loss/{}'.format(prefix, k)] = v
        for k,v in distances.items():
            all_metrics['{}_distance/{}'.format(prefix, k)] = v
        all_metrics['{}_loss/total'.format(prefix)] = total_loss
        all_metrics['{}_correspondences'.format(prefix)] = float(num_corrs)

        self.log_dict(
            all_metrics, 
            on_step=False, 
            on_epoch=True, 
            logger=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=self.args.bs
        )

################ TEST ################################
    
    def get_pred_filename(self):
        checkpoint_name = int(os.path.splitext(self.args.checkpoint)[0])
        filename = 'pred_{}_epoch={:04d}_{}_{}_{}'.format(self.args.split, checkpoint_name, self.args.obj, self.args.solver, self.args.seed)
        
        if self.args.oracle is not None:
            filename += f'_{self.args.oracle}'

        filename += '.csv'

        return filename

    def on_test_start(self):

        res_filename = self.get_pred_filename()
        self.args.results_file = os.path.join(self.args.results_out, res_filename)
        
        print('Logging results in {}'.format(self.args.results_file))
        
        with open(self.args.results_file,'w') as f:
            f.write('scene_id,im_id,obj_id,score,R,t,time\n')
        
        return super().on_test_start()
    
    def test_step(self, batch, batch_idx):

        scene_coords = batch[0]
        scene_feats = batch[1]
        obj_coords = batch[3]
        obj_feats = batch[4]
        gt_poses = batch[7]
        part_ids = batch[9]
        image_ids  = batch[10]
        obj_ids =  batch[11]

        #out_obj, out_scene, scene_filter, sparse_corrs, sym_lists, diameters
        out_obj, out_scene, scene_filter, sparse_corrs, _ = self.forward_batch(batch)
        obj_list = torch.unique(out_obj.C[:,0]).cpu().numpy()

        for i_b in obj_list:
            
            obj_coords, obj_feats = out_obj.coordinates_at(i_b), out_obj.features_at(i_b)
            scene_coords, scene_feats = out_scene.coordinates_at(i_b), out_scene.features_at(i_b)

            part_id, img_id, obj_id = part_ids[i_b].item(), image_ids[i_b].item(), obj_ids[i_b].item()

            gt_pose = gt_poses[i_b]
            path = os.path.join(self.args.results_out, 'pcds')
            instance_id = '{:06d}_{:04d}_{:02d}'.format(int(part_id), int(img_id), int(obj_id))
    
            # if using oracle, compute object bbox and crop scene with it
            if self.args.oracle is not None:
                
                filter_i = scene_filter[i_b].squeeze(1)
                scene_coords = scene_coords[filter_i,:]
                scene_feats = scene_feats[filter_i,:]                

            if scene_feats.shape[0] > 1 and obj_feats.shape[0] > 1:

                pred_pose = register_pcd(obj_coords, scene_coords, obj_feats, scene_feats, solver=self.args.solver)
                pred_pose = np.copy(pred_pose)

            else:
                pred_pose = np.eye(4)

            if self.args.save_results:

                corrs = sparse_corrs.coordinates_at(i_b)
                np.save('{}/obj_pcd_{}.npy'.format(path, instance_id), obj_coords.cpu().numpy())
                np.save('{}/obj_feats_{}.npy'.format(path, instance_id), obj_feats.cpu().numpy())
                np.save('{}/scene_pcd_{}.npy'.format(path, instance_id), scene_coords.cpu().numpy())
                np.save('{}/scene_feats_{}.npy'.format(path, instance_id), scene_feats.cpu().numpy())
                np.save('{}/gt_pose_{}.npy'.format(path, instance_id), gt_pose.cpu().numpy())
                np.save('{}/corr_{}.npy'.format(path, instance_id), corrs.cpu().numpy())
                np.save('{}/pred_pose_{}.npy'.format(path, instance_id), pred_pose)
            
            pred_pose[:3,3] = pred_pose[:3,3] * self.args.voxel_size
            pred_r_i, pred_t_i = pred_pose[:3,:3], pred_pose[:3,3]
            
            with open(self.args.results_file,'a')  as f:
                f.write('{},{},{},1.0,'.format(part_id, img_id, obj_id))
                f.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f},'.format(
                    pred_r_i[0][0], pred_r_i[0][1], pred_r_i[0][2], pred_r_i[1][0], pred_r_i[1][1], pred_r_i[1][2], pred_r_i[2][0], pred_r_i[2][1], pred_r_i[2][2]))
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(pred_t_i[0], pred_t_i[1], pred_t_i[2]))

###################### DATALOADERS ######################

    def get_train_dataloader(self):
        
        args = self.args
        
        dataset = mink_dataset(
            root=args.path,
            name=args.dataset,
            split=args.split_train,
            model_points=args.model_points,
            voxel_size=args.voxel_size,
            scene_points=args.depth_points,
            obj_split=args.train_obj,
            augs_erase=args.aug_erase,
            augs_rgb=args.aug_rgb,
            corr_th = args.corr_th
        )


        # Log train/valid stats
        print('TRAIN samples: {:d}'.format(len(dataset)))

        # Get the training data loader
        loader_train = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.bs,
            num_workers=args.n_workers,
            collate_fn=ME_collation_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

        return loader_train
    
    def get_valid_dataloader(self):

        args = self.args
        # Get the validation dataset
        dataset_valid = mink_dataset(
            root=args.path,
            name=args.dataset,
            split=args.split_valid,
            model_points=args.model_points,
            voxel_size=args.voxel_size,
            scene_points=args.depth_points,
            obj_split=args.valid_obj,
            oracle=args.oracle,
            corr_th = args.corr_th
        )
        print('VALID samples: {:d}'.format(len(dataset_valid)))

        # Get the validation data loader
        loader_valid = torch.utils.data.DataLoader(
            dataset=dataset_valid,
            batch_size=args.bs,
            num_workers=args.n_workers,
            collate_fn=ME_collation_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=True)

        return loader_valid

    def get_test_dataloader(self):

        args = self.args
        # Get the validation dataset
        dataset_test = mink_dataset(
            root=args.path,
            name=args.dataset,
            split=args.split,
            model_points=args.model_points,
            voxel_size=args.voxel_size,
            scene_points=args.depth_points,
            obj_split=args.obj,
            seed=args.seed,
            corr_th=2.,
            oracle=args.oracle
        )
        print('TEST samples: {:d}'.format(len(dataset_test)))

        # Get the validation data loader
        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=args.bs,
            num_workers=args.n_workers,
            collate_fn=ME_collation_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        return loader_test

    def forward(self, x):
        return self.model(x)