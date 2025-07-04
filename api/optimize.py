import numpy as np
from scipy.optimize import dual_annealing
from pyswarm import pso
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

# Portfolio optimization objective function with Kolmogorov penalties
def objective_function(w, cov_matrix, expected_returns, lambda_0, lambda_1):
    risk = np.dot(w.T, np.dot(cov_matrix, w))  # Portfolio risk
    return_ = np.dot(w.T, expected_returns)  # Portfolio return
    sparsity = lambda_0 * np.count_nonzero(w)  # L0 sparsity penalty
    magnitude = lambda_1 * np.sum(np.abs(w))  # L1 weight magnitude penalty
    return risk - return_ + sparsity + magnitude

# Optimization using Simulated Annealing (SA)
def optimize_portfolio_sa(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1
    result_sa = dual_annealing(objective_function, bounds=list(zip(lb, ub)), maxiter=2000, maxfun=2000, args=(X, lambda_0, lambda_1))
    return result_sa.x

# Optimization using Particle Swarm Optimization (PSO)
def optimize_portfolio_pso(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1
    optimal_weights, _ = pso(objective_function, lb, ub, maxiter=2000, swarmsize=80, c1=2, c2=2, args=(X, lambda_0, lambda_1))
    return optimal_weights

# Risk Metrics - VaR and CVaR
def compute_VaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    return np.percentile(portfolio_returns, alpha * 100)

def compute_CVaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    var = compute_VaR(weights, returns_data, alpha)
    return portfolio_returns[portfolio_returns <= var].mean()

# API Route to handle POST request for portfolio optimization
@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    data = request.json
    returns_data = np.array(data['returns_data'])
    lambda_0 = data['lambda_0']
    lambda_1 = data['lambda_1']

    # Calculate covariance matrix and expected returns
    cov_matrix = np.cov(returns_data.T)
    expected_returns = np.mean(returns_data, axis=0)

    # Run optimization using PSO or SA (default SA here)
    optimized_weights = optimize_portfolio_sa(returns_data, lambda_0, lambda_1)

    # Calculate risk metrics
    var = compute_VaR(optimized_weights, returns_data)
    cvar = compute_CVaR(optimized_weights, returns_data)

    # Return optimized portfolio weights and risk metrics
    return jsonify({
        'optimized_weights': optimized_weights.tolist(),
        'risk_metrics': {
            'VaR': var,
            'CVaR': cvar
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
