from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from pipeline import MinkowskiMetricLearning
from utils.parsing import parse_train_args as parse_args
from utils.trainer import init_plugins, init_storage_folders
from os.path import join, exists
import torch

def get_ckpt_path(args):
    '''
    Get checkpoint path from folder of previous experiment
    '''
    ckpt_file = 'epoch={:04d}.ckpt'.format(int(args.checkpoint))
    ckpt_path = join(args.exp_root, args.exp, 'models', ckpt_file)

    if not exists(ckpt_path):
        raise RuntimeError('Checkpoint {} does not exist.'.format(ckpt_path))

    print('Resuming from checkpoint at {}'.format(ckpt_path))

    return ckpt_path
    
def run_pipeline(args):

    args.mode = 'train'
    checkpoint_out, logs_out, results_out = init_storage_folders(args, 2)    
    print('Running on ', torch.cuda.device_count(), ' GPUs.')
    
    args.logs_out = logs_out
    args.ckpt_out = checkpoint_out
    args.results_out = results_out
    
    if 'slurm' in args:
        args.sanity_steps = 2 # default lightning sanity steps
        args.strategy = 'ddp_find_unused_parameters_false'
        args.n_workers = args.slurm.cpus_per_task
    else:
        args.sanity_steps = 0 # don't when debugging
        args.num_nodes=1
        args.strategy = None
        args.n_workers = 0


    system = MinkowskiMetricLearning(args)

    if args.profile:
        profiler = AdvancedProfiler(
            dirpath=args.ckpt_out,
            filename='profiler.out'
        )
    else: 
        profiler = False

    trainer = Trainer(
        profiler=profiler,
        logger = system.get_logger(),
        plugins=init_plugins(args),
        enable_checkpointing=True,
        num_sanity_val_steps=args.sanity_steps,
        callbacks=system.get_callbacks(),
        accelerator='gpu',
        strategy=args.strategy,
        log_every_n_steps=10,
        # NB: needs devices to run on local machine
        auto_select_gpus=True,
        num_nodes=args.num_nodes,
        check_val_every_n_epoch=args.freq_valid,
        max_epochs=args.n_epochs
    )

    train_data = system.get_train_dataloader()
    valid_data = system.get_valid_dataloader()
    
    ckpt_path = None
    if args.flag_resume:
        ckpt_path = get_ckpt_path(args)

    trainer.fit(
        system, 
        train_dataloaders=train_data, 
        val_dataloaders=valid_data,
        ckpt_path=ckpt_path
    )

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
