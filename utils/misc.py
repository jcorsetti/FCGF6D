import re
import logging
import numpy as np


def np_normalize(v : np.ndarray) -> np.ndarray:
    
    assert len(v.shape) == 1, ' Works only with 1-dimensional arrays!'
    
    v = v.astype(float)
    vmax, vmin = np.max(v), np.min(v)

    return (v - vmax) / (vmin -vmax)

def sorted_alphanumeric(data):
    '''
    https://gist.github.com/SeanSyue/8c8ff717681e9ecffc8e43a686e68fd9
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_logger(path_log):
    '''
    https://www.toptal.com/python/in-depth-python-logging
    '''
    # Get logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    # Get formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Get file handler and add it to logger
    fh = logging.FileHandler(path_log, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Get console handler  and add it to logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger

def generate_random_rotation():
    # Generate rotation
    anglex = np.random.uniform() * np.pi * 2
    angley = np.random.uniform() * np.pi * 2
    anglez = np.random.uniform() * np.pi * 2

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz
    
    return R_ab

