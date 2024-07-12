import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import time


def worker(x, queue):
    queue.put(x)


def test():
    queue = mp.SimpleQueue()
    tasks = range(10)

    for task in tasks:
        mp.Process(
            target=worker,
            args=(
                task,
                queue,
            ),
        ).start()

    for _ in tasks:
        print(queue.get())
