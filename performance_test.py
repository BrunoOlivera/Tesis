import numpy as np
import timeit
import time


def numpy_clip_list(seq_to_clip):
    for i in range(1_000_000):
        np.clip(seq_to_clip, 0, 680)


def numpy_clip_np(seq_to_clip):
    for i in range(1_000_000):
        np.clip(np.array(seq_to_clip), 0, 680)


def numpy_clip_max_min(seq_to_clip):
    # for i in range(2):
    #     np.max(0, np.min(seq_to_clip, 680))
    pass


def efficient_clip(seq_to_clip):
    seq_to_clip[seq_to_clip < 0] = 0
    seq_to_clip[seq_to_clip > 680] = 680


def a():
    for i in range(10_000_000):
        res = np.zeros_like((10, 100))


def b():
    res = np.empty_like((10, 100))
    for i in range(10_000_000):
        res[:] = 0


def c():
    res = np.empty_like((10, 100))
    for i in range(10_000_000):
        res.fill(0)


def d():
    res = np.empty_like((10, 100))
    for i in range(10_000_000):
        res * 0


# seq = [-1] * 1000 + [700] * 1000
seq = [-1, -1, -1, 700, 700, 700]
# seq = np.array([-1] * 10000 + [700] * 10000)

# print(f'numpy_clip_list: {timeit.timeit("numpy_clip_list(seq)", globals=globals()):.2f}s')
# print(f'numpy_clip_np: {timeit.timeit("numpy_clip_np(seq)", globals=globals()):.2f}s')
# print(f'numpy_clip_max_min: {timeit.timeit("numpy_clip_max_min(seq)", globals=globals()):.2f}s')
t_0 = time.time()
# print(f'{numpy_clip_list(seq)} t: {time.time()-t_0}s')
# print(f'{numpy_clip_list(seq)} t: {time.time()-t_1}s')
a()
print(f'a-> {time.time()-t_0}s')
t_1 = time.time()
b()
print(f'b-> {time.time()-t_1}s')
t_2 = time.time()
c()
print(f'c-> {time.time()-t_2}s')
t_3 = time.time()
d()
print(f'd-> {time.time()-t_3}s')
