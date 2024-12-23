import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Queue
import threading
import ctypes

import numpy as np


def image4k():
    img = np.empty((3840, 2160, 3), dtype=np.uint8)
    img[990:1000, 990:1000, :] = 0
    return img


def modify_first(queue):
    while True:
        data = queue.get()
        if data is None:
            break
        # data = np.memmap('/dev/shm/shared_data', dtype=np.int8, mode='w+', shape=(10,))
        shm = shared_memory.SharedMemory(name=data)
        data = np.ndarray((10,), dtype=np.int8, buffer=shm.buf)
        data[0] = 1
        shm.close()


# Methods: copy-on-write, shared array, memory-mapped file, sharedmemory, ray?


def main():
    queue = mp.Queue()
    data = np.zeros((10,), dtype=np.int8)
    # shared_data = np.memmap('/dev/shm/shared_data', dtype=np.int8, mode='w+', shape=(10,))
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name='shared_data')
    shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(shared_data, data)
    # shared_array = mp.RawArray(ctypes.c_int8, 10)
    # shared_array_np = np.ndarray((10,), dtype=np.int8, buffer=shared_array)
    # np.copyto(shared_array_np, data)
    # data = [0] * 10
    # data = '00000000000'
    print(shared_data)
    # p = threading.Thread(target=modify_first, args=(queue,))
    p = mp.Process(target=modify_first, args=(queue,))
    p.start()
    queue.put('shared_data')
    queue.put(None)
    p.join()
    print(shared_data)
    shm.close()
    shm.unlink()


if __name__ == '__main__':
    main()