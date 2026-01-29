import matplotlib.pyplot as plt
import numpy as np

def plot_log_likelihood(p_values, ll_values, p_mle=None):
    plt.figure()
    plt.plot(p_values, ll_values, label="Log Likelihood")

    if p_mle is not None:
        plt.axvline(p_mle, linestyle="--", label=f"MLE p = {p_mle:.3f}")

    plt.xlabel("Probability of Heads (p)")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood vs Probability of Heads")
    plt.legend()
    plt.show()


def plot_mle_convergence(sample_sizes, p_mle_values, true_p):
    plt.figure()
    plt.plot(sample_sizes, p_mle_values, marker="o", label="MLE Estimate")
    plt.axhline(true_p, linestyle="--", label="True p")

    plt.xlabel("Sample Size")
    plt.ylabel("MLE Estimate of p")
    plt.title("MLE Convergence with Increasing Sample Size")
    plt.legend()
    plt.show()

def plot_mu_convergence(sample_sizes, mu_estimates, true_mu):
    plt.figure()
    plt.plot(sample_sizes, mu_estimates, marker='o', label='MLE Estimate')
    plt.axhline(true_mu, linestyle='--', label='True μ')
    plt.xlabel('Sample Size')
    plt.ylabel('MLE of Mean')
    plt.title('MLE Convergence of Normal Mean')
    plt.legend()
    plt.show()
