from dataclasses import dataclass

@dataclass
class SLURMConf:
    r"""
    This data class store all the setting needed to set up SLURM
    """
    # This are set automatically by the submission process
    is_slurm : bool = False # is SLURM active
    host_name: str = "" # the hostname of the SLURM master
    port: str = "" # the port of the SLURM master
    world_size: int = 1 # the number of SLURM processes
    global_rank: int = 0 # the global rank of the current process
    local_rank: int = 0 # the local rank of the current process
    node_rank: int = 0 # the node rank of the current process

    # This must be set by the user
    partition : str = "gpu-A40" # slurm partition
    timeout : int = 10080 # slurm timeout minimum
    num_nodes : str = 1 # the number of nodes to use
    ngpus_per_node: int = 2 # nGPUs per node
    cpus_per_task: int = 10 # nCPU for each task
    giga_per_gpu: int = 20 # giga to assign to each GPU

    account : str = "" # the optional account name to use
    task_name : str = "eval" # the name of the SLURM task
