import numpy as np
from data import generate_coin_data
from likelihood import log_likelihood as likelihood
from visual import plot_log_likelihood

data = generate_coin_data(n=10, true_p=0.7, seed=1)

p_values = np.linspace(0.01, 0.99, 100)
ll_values = [likelihood(p, data) for p in p_values]

p_mle = data.mean()
print("MLE estimate of probability of heads:", p_mle)

plot_log_likelihood(p_values, ll_values, p_mle)
