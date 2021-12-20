import numpy as np


def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end-1, :, :] = disps

    return final_sample
