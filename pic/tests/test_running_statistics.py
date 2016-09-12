from pic.running_statistics import update_mean_estimate, std_from_means
import numpy as np


def test_running_stats():
    samples = np.random.random(1000)
    mean = 0.0
    mean_sq = 0.0
    for i in range(samples.shape[0]):
        mean = update_mean_estimate(samples[i], mean, i+1)
        mean_sq = update_mean_estimate(samples[i]**2, mean_sq, i+1)
    std = std_from_means(mean, mean_sq)
    np.testing.assert_allclose(mean, np.mean(samples))
    np.testing.assert_allclose(std, np.std(samples))
