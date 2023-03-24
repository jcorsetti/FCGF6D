import os

from subprocess import check_output
from submitit import AutoExecutor, JobEnvironment

from utils.trainer import get_new_exp_version

def slurm_runner(args, fun_to_run):
    r"""
    This function schedule a SLURM job that will run the specify function
    """
    args.slurm.is_slurm = True

    class SLURM_Trainer(object):
        def __init__(self, args, function_to_call):
            self.args = args
            self.function_to_call = function_to_call

        def __call__(self):

            # find a common host name on all nodes
            cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
            stdout = check_output(cmd.split())
            host_name = stdout.decode().splitlines()[0]

            job_env = JobEnvironment()
            
            # Set the port
            # use the last 4 numbers in the job id as the id
            default_port = job_env.job_id[-4:]
            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

            self.args.distributed_strategy = "ddp"
            self.args.slurm.port = default_port

            self.args.slurm.host_name = host_name

            # distributed parameters
            self.args.slurm.world_size = int(os.getenv('SLURM_JOB_NUM_NODES')) * self.args.slurm.ngpus_per_node 
            
            print("Current GPU used: ", str(job_env.local_rank))
            self.args.slurm.node_rank = job_env.node
            self.args.slurm.local_rank = job_env.local_rank
            self.args.devices = -1
            self.args.slurm.global_rank = job_env.global_rank
            self.args.num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))

            self.function_to_call(self.args)

    # Training: if resuming get exp name, otherwise get new
    if 'freq_train' in args:
        if args.flag_resume:
            exp_name = args.exp
        else:
            exp_name = get_new_exp_version(args)
            args.exp_name = exp_name    
        exp_name = os.path.join(exp_name,'slurm')

    # not training: just add test
    else:
        exp_name = os.path.join(args.exp, 'slurm','test')
    
    output_dir = os.path.join(args.exp_root, exp_name)
    print("Output dir: ", output_dir)

    executor = AutoExecutor(folder=output_dir, slurm_max_num_timeout=30)

    executor.update_parameters(
        name=args.slurm.task_name,
        mem_gb=args.slurm.giga_per_gpu*args.slurm.ngpus_per_node,
        gpus_per_node=args.slurm.ngpus_per_node,
        tasks_per_node=args.slurm.ngpus_per_node,
        cpus_per_task=args.slurm.cpus_per_task,
        nodes=int(args.slurm.num_nodes),
        timeout_min=args.slurm.timeout,
        slurm_partition=args.slurm.partition,
    )

    if len(args.slurm.account):
        executor.update_parameters(
            slurm_additional_parameters={
                'account': args.slurm.account
            }
        )

    trainer = SLURM_Trainer(args, fun_to_run)
    job = executor.submit(trainer)
    
    print(f"Logging slurm in {output_dir}.")
    print(f"Submitted job_id: {job.job_id}")
