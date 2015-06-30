import numpy as np

def filter_sequence(seq, filter_var, eps):
    filtered = np.zeros(seq.shape)
    T = seq.shape[1]
    x = np.arange(-T, T+1)
    filter = np.exp(-0.5 * (1 / (filter_var ** 2)) * np.square(x))
    l = T - np.where(filter > eps)[0]
    filter = filter[T-l+1:T+l+2]
    for t in xrange(T):
        filter_start, filter_end = max(1, l-t+2), min(l*2+1, T+l-t+1) + 1
        T_start, T_end = max(1, t-l), min(T, t+l)
        f = filter[filter_start:filter_end]
        filtered[:,t] = np.sum(seq[:,T_start:T_end]*f, axis=1) / np.sum(f)
    return filtered
