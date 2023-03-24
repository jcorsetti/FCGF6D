import torch
from os.path import join
from utils.parsing import parse_test_args as parse_args
from pytorch_lightning import Trainer
from pipeline import MinkowskiMetricLearning

def run_test(args):

    args.mode = 'test'
    checkpoints_output_folder = join(args.exp_root, args.exp, "models")
    logs_output_folder = join(args.exp_root, args.exp, "runs")
    results_output_folder = join(args.exp_root, args.exp, "results")
    args.output_dir = join(checkpoints_output_folder,'test')
    args.logs_out = logs_output_folder
    args.ckpt_out = checkpoints_output_folder
    args.results_out = results_output_folder

    if 'slurm' in args:
        args.strategy = 'ddp_find_unused_parameters_false'
        args.n_workers = args.slurm.cpus_per_task
    else:
        args.num_nodes=1
        args.strategy=None
        args.n_workers = 8

    print('Running on ', torch.cuda.device_count(), ' GPUs.')
    system = MinkowskiMetricLearning(args, test_model=True)

    trainer = Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        # NB: needs devices to run on local machine
        auto_select_gpus=True,
        log_every_n_steps=10,
        num_nodes=args.num_nodes
    )

    test_data = system.get_test_dataloader()
    
    ckpt_file = 'epoch={:04d}.ckpt'.format(int(args.checkpoint))
    trained_model = MinkowskiMetricLearning.load_from_checkpoint(join(args.ckpt_out, ckpt_file), args=args, test_model=True)

    trainer.test(model=trained_model, dataloaders=test_data)

if __name__ == '__main__':
    args = parse_args()
    run_test(args)
