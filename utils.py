import numpy as np

def crop_breakout(a):
    ans = a[:,4:78+1, 16:79+1]
    # print(ans.shape)
    return ans


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n