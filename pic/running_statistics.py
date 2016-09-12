import numba

@numba.jit(nopython=True)
def update_mean_estimate(new_sample, mean_estimate, samples_count):
    """Return updated mean estimate with new sample added

    samples_count includes the new one

    Based on Numerically stable mean:
    http://diego.assencio.com/?index=c34d06f4f4de2375658ed41f70177d59

    """
    return mean_estimate + (new_sample - mean_estimate) / samples_count


def std_from_means(mean, mean_squares):
    """Return standard deviation estimated from mean and mean of squares"""
    return abs(mean_squares - mean**2)**0.5
