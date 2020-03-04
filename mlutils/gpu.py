import os
import ctypes
import sys
from typing import Union, List


class CUDAProvider:
    """
    Class for call CUDA functions.
    """

    def __init__(self):
        self._cuda = self._init_cuda()

    def _init_cuda(self):
        libname = self.platform_libname()
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            raise OSError('CUDA is not available on this device.')

        status = cuda.cuInit(0)
        self._check_error(status, 'cuInit')
        return cuda

    @classmethod
    def platform_libname(cls):
        """
        The method mapping this platform name with specific cuda library name.
        Returns
        -------
            str: library name

        Raises
        -------
            OSError: if platform is not linux/macos/win.
        """
        platform = sys.platform
        if 'linux' in platform:
            return 'libcuda.so'
        elif 'darwin' in platform:
            return 'libcuda.dylib'
        elif 'win' in platform:
            return 'cuda.dll'
        else:
            raise OSError(f'Unknown libname for {platform}')

    def get_count_devices(self):
        """
        Get count available CUDA GPU.
        Returns
        -------
            int: count available CUDA devices.
        """
        n_var = ctypes.c_int()
        status = self._cuda.cuDeviceGetCount(ctypes.byref(n_var))
        self._check_error(status, 'cuDeviceGetCount')
        return n_var.value

    def _check_error(self, status, func_name: str):
        if status != 0:
            result = ctypes.c_int()
            error_str = ctypes.c_char_p()
            self._cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise OSError(f"{func_name} failed with error code {result}: {error_str.value.decode()})")


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
    cuda = CUDAProvider()
    gpu_count = cuda.get_count_devices()
    if gpu_count == 0:
        raise GPUNotAvailable('GPU is not available.')

    gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
    available_gpu = []
    for gpu_id in gpu_ids:
        if gpu_id < gpu_count:
            available_gpu.append(gpu_id)
        else:
            print(f'Warning: GPU {gpu_id} is not available.')
    if len(available_gpu) == 0:
        raise GPUNotAvailable(f'GPU(s) {gpu_ids} is not available. Maybe used {list(range(gpu_count))}?')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in available_gpu)


def set_used_cpu():
    """
    Specifies for CUDA to use the GPU for computing.
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GPUNotAvailable(Exception):
    pass
