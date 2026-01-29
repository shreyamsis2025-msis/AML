import numpy as np
from visual import plot_mu_convergence

np.random.seed(0)

true_mu = 30.0
true_sigma = 2.0
sample_sizes = [10, 30, 50, 100, 500, 1000]

mu_estimates = []

for n in sample_sizes:
    data = np.random.normal(true_mu, true_sigma, size=n)
    
    # MLE of mean for Normal distribution
    mu_mle = data.mean()
    mu_estimates.append(mu_mle)

print("MLE estimates of mean:", mu_estimates)

plot_mu_convergence(sample_sizes, mu_estimates, true_mu)
