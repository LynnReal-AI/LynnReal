"""Inference schedule shared by the Base sampler."""
def shift_sigma(sigma,shift=3.):
    return shift*sigma/(1+(shift-1)*sigma)
