import os
from typing import Union, List
from tensorflow.python.client import device_lib


def set_used_gpu(gpu_ids: Union[List[int], int]):
    """
    Specify for CUDA which GPU(s) to be used for computing.
    Set environment variable CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES.

    Parameters
    ----------
    gpu_ids : int or array-like of int
        if gpu_ids is a list, then it should be contains GPU IDs.
        if gpu_ids if int, then it should be equal GPU id.

    Raises
    ----------
    GPUNotAvailable: if GPU is not available.

    """
    all_gpu_available = [
        int(device.name.split(':')[-1])
        for device in device_lib.list_local_devices()
        if device.device_type == 'GPU'
    ]
    if len(all_gpu_available) == 0:
        raise GPUNotAvailable('GPU is not available.')

    gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
    available_gpu = []
    for gpu_id in gpu_ids:
        if gpu_id in all_gpu_available:
            available_gpu.append(gpu_id)
        else:
            print(f'Warning: GPU {gpu_id} is not available.')
    if len(available_gpu) == 0:
        raise GPUNotAvailable(f'GPU(s) {gpu_ids} is not available. Maybe used f{all_gpu_available}?')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_gpu)


def set_used_cpu():
    """
    Specifies for CUDA to use the GPU for computing.
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GPUNotAvailable(Exception):
    pass
