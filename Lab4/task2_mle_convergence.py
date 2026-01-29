import numpy as np
from visual import plot_mle_convergence

np.random.seed(0)

true_p = 0.7
sample_sizes = [10, 30, 50, 100, 500, 1000]
p_mle_values = []

for n in sample_sizes:
    data = np.random.binomial(1, true_p, n)
    p_mle_values.append(data.mean())

plot_mle_convergence(sample_sizes, p_mle_values, true_p)
