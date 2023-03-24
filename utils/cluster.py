from pytorch_lightning.plugins.environments import ClusterEnvironment

from .slurm_config import SLURMConf
from typing import Optional
from os import environ

__all__ = [
    'SLURMClusterSettings',
]

class SLURMClusterSettings(ClusterEnvironment):
    r"""
    The class read from the arguments (mostly automatically set by the SLURM executor) the parameter that allow PyTorch to communicate with SLURM
    """

    def __init__(self, args : 'SLURMConf', exp_name : 'str'):
        self.__master_address = args.host_name
        self.__master_port = args.port
        self.__world_size = args.world_size
        self.__global_rank = args.global_rank
        self.__local_rank = args.local_rank
        self.__node_rank = args.node_rank
        self.__exp_name = exp_name

    @property
    def creates_processes_externally(self) -> 'bool':
        return True

    @property
    def main_address(self):
        environ["MASTER_ADDR"] = self.__master_address
        return self.__master_address

    @property
    def main_port(self) -> 'int':
        environ["MASTER_PORT"] = str(self.__master_port)
        return self.__master_port

    @staticmethod
    def detect() -> 'bool':
        """Returns ``True`` if the current process was launched on a SLURM cluster.
        It is possible to use the SLURM scheduler to request resources and then launch processes manually using a
        different environment. For this, the user can set the job name in SLURM to 'bash' (``SLURM_JOB_NAME=bash``).
        This will then avoid the detection of ``SLURMEnvironment`` and another environment can be detected
        automatically.
        """
        return "SLURM_NTASKS" in environ

    @staticmethod
    def job_name() -> 'Optional[str]':
        return self.__exp_name

    @staticmethod
    def job_id() -> 'Optional[int]':
        return None

    def world_size(self):
        return self.__world_size

    def set_world_size(self, size: int) -> None:
        print("Impossible set world size in a SLURM environment, must be specify using the settings. Ignored")

    def global_rank(self) -> 'int':
        return self.__global_rank
    
    def set_global_rank(self, rank: int) -> None:
        print("Impossible set global rank in a SLURM environment, must be specify using the settings. Ignored")

    def local_rank(self) -> 'int':
        return self.__local_rank
    
    def node_rank(self) -> 'int':
        return self.__node_rank
    
    def set_world_size(self, size: 'int') -> None:
        pass

    def set_global_rank(self, rank: 'int') -> None:
        pass
