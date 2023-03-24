from time import sleep
from .cluster import SLURMClusterSettings
from datetime import datetime
from os import makedirs, listdir
from os.path import isdir, join

def get_parameters(models):
    r"""
    This function get all the parameter recursive exploring the dictionary, list or module in input if necessary
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_learning_rate(optimizer):
    r"""
    This function return the first learning rate found inside the given optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_new_exp_version(args):
    
    old_exps = [int(file.split('_')[-1][1:]) for file in listdir(args.exp_root) if file.startswith(args.exp + '_V')]
    
    # first exp of this name
    if len(old_exps) == 0:
        version = 0
    # not first exp, set flag_resume
    elif args.flag_resume:
        version = max(old_exps)
    # not first exp, no flag_resume
    else:
        version = max(old_exps) +1

    new_exp = '{}_V{}'.format(args.exp, version)

    return new_exp


def init_storage_folders(args, creation_wait_timeout):
    
    r"""
    This function init the directory store where store the results

    This include the directory where storing the checkpoints and logs
    """

    if not args.flag_resume:

        if 'slurm' not in args:
            args.exp_name = get_new_exp_version(args)

        checkpoints_output_folder = join(args.exp_root, args.exp_name, "models")
        logs_output_folder = join(args.exp_root, args.exp_name, "runs")
        results_output_folder = join(args.exp_root, args.exp_name, "results")


        if 'slurm' not in args:
            makedirs(checkpoints_output_folder, exist_ok=False)
            makedirs(logs_output_folder, exist_ok=False)
            makedirs(results_output_folder, exist_ok=False)
            makedirs(join(results_output_folder,'pcds'), exist_ok=False)

        elif not (args.slurm.global_rank != 0 or args.slurm.local_rank != 0):
            makedirs(checkpoints_output_folder, exist_ok=False)
            makedirs(logs_output_folder, exist_ok=False)
            makedirs(results_output_folder, exist_ok=False)
            makedirs(join(results_output_folder,'pcds'), exist_ok=False)

            # Wait the master node create the directories, before proceeding
            if (args.slurm.global_rank != 0 or args.slurm.local_rank != 0):
                start_wait_time = datetime.now()
                time_expired = True
                while (datetime.now() - start_wait_time).total_seconds() < creation_wait_timeout:
                    if isdir(checkpoints_output_folder) and isdir(logs_output_folder) and isdir(results_output_folder):
                        time_expired = False
                        break
                    sleep(0.2)
                if time_expired:
                    raise TimeoutError("Time expire during the control of the creation of the directory")
                
                print("Output directories was correctly created by the master")
    else:
        args.exp_name = args.exp
        checkpoints_output_folder = join(args.exp_root, args.exp, "models")
        logs_output_folder = join(args.exp_root, args.exp, "runs")
        results_output_folder = join(args.exp_root, args.exp, "results")

    print("Checkpoint folder: {}".format(checkpoints_output_folder))
    print("Logs folder: {}".format(logs_output_folder))
    print("Results folder: {}".format(results_output_folder))

    return checkpoints_output_folder, logs_output_folder, results_output_folder

def init_plugins(args):
    r"""
    This function init the plugins that allow PyTorch Lighting to work in the various environments supported (SLURM, single local GPU, multiple local GPUs)
    """
    # Init the various plugins for the trainer
    plugins = []

    if 'slurm' in args:
        # SLURM plugin
        if args.slurm.is_slurm:
            slurm_cluster_settings = SLURMClusterSettings(args.slurm, args.exp)
            plugins.append(slurm_cluster_settings)    
    return plugins

