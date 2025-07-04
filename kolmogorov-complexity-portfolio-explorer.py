import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyswarm import pso
from scipy.stats import genpareto
from scipy.optimize import dual_annealing

# 1. Define the objective function for portfolio optimization with Kolmogorov complexity penalty
def portfolio_risk(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def kolmogorov_penalty(weights, lambda_0, lambda_1):
    # L0 Penalty (sparsity)
    l0_penalty = lambda_0 * np.count_nonzero(weights)
    # L1 Penalty (regularization)
    l1_penalty = lambda_1 * np.sum(np.abs(weights))
    return l0_penalty + l1_penalty

def objective_function(weights, cov_matrix, lambda_0, lambda_1):
    risk = portfolio_risk(weights, cov_matrix)
    penalty = kolmogorov_penalty(weights, lambda_0, lambda_1)
    return risk + penalty

# 2. Simulated Annealing Optimization (SA)
def optimize_portfolio_sa(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1
    result_sa = dual_annealing(objective_function, bounds=list(zip(lb, ub)), maxiter=2000, maxfun=2000, args=(X, lambda_0, lambda_1))
    return result_sa.x

# 3. Particle Swarm Optimization (PSO)
def optimize_portfolio_pso(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1
    optimal_weights, _ = pso(objective_function, lb, ub, maxiter=2000, swarmsize=80, c1=2, c2=2, args=(X, lambda_0, lambda_1))
    return optimal_weights

# 4. Refined Tail Risk (99% VaR using EVT and GPD)
def gpd_tail_risk(returns, threshold=0.05):
    excess_returns = returns[returns > threshold] - threshold
    params = genpareto.fit(excess_returns)
    tail_risk = genpareto.ppf(0.99, *params)
    return tail_risk

def compute_tail_risk(X, optimized_weights):
    portfolio_returns = np.dot(X, optimized_weights)
    return gpd_tail_risk(portfolio_returns)

# 5. Visualization of Sparse Weights
def plot_sparse_weights(weights):
    plt.bar(range(len(weights)), weights)
    plt.xlabel('Assets')
    plt.ylabel('Weights')
    plt.title('Sparse Portfolio Weights')
    plt.show()

# Example Data (n_assets x n_samples)
np.random.seed(42)
n_assets = 5
n_samples = 1000
X = np.random.randn(n_samples, n_assets)
cov_matrix = np.cov(X.T)

# Hyperparameters for Kolmogorov complexity penalty
lambda_0 = 0.1  # L0 penalty strength (sparsity)
lambda_1 = 0.01  # L1 penalty strength (regularization)

# Optimization using PSO
optimized_weights_pso = optimize_portfolio_pso(X, lambda_0, lambda_1)
print(f"Optimized Portfolio Weights (PSO): {optimized_weights_pso}")

# Optimization using SA
optimized_weights_sa = optimize_portfolio_sa(X, lambda_0, lambda_1)
print(f"Optimized Portfolio Weights (SA): {optimized_weights_sa}")

# Compute refined tail risk (99% VaR)
tail_risk = compute_tail_risk(X, optimized_weights_sa)
print(f"Refined Tail Risk (99% VaR): {tail_risk}")

# Plot sparse portfolio weights
plot_sparse_weights(optimized_weights_sa)
