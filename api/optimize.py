import numpy as np
from flask import Flask, request, jsonify
from scipy.optimize import dual_annealing
from pyswarm import pso

from api.utils import compute_VaR, compute_CVaR

app = Flask(__name__)

# -----------------------------------
# Objective Function
# -----------------------------------
def objective_function(weights, *args):
    cov_matrix, expected_returns, lambda_0, lambda_1 = args
    risk = np.dot(weights.T, np.dot(cov_matrix, weights))
    return_ = np.dot(weights.T, expected_returns)
    sparsity = lambda_0 * np.count_nonzero(weights)
    magnitude = lambda_1 * np.sum(np.abs(weights))
    return risk - return_ + sparsity + magnitude

# -----------------------------------
# Simulated Annealing Optimizer
# -----------------------------------
def optimize_portfolio_sa(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1

    cov_matrix = np.cov(X.T)
    expected_returns = np.mean(X, axis=0)

    result_sa = dual_annealing(
        objective_function,
        bounds=list(zip(lb, ub)),
        maxiter=2000,
        maxfun=2000,
        args=(cov_matrix, expected_returns, lambda_0, lambda_1)
    )
    return result_sa.x

# -----------------------------------
# Particle Swarm Optimizer (Optional)
# -----------------------------------
def optimize_portfolio_pso(X, lambda_0, lambda_1):
    n_assets = X.shape[1]
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.1

    cov_matrix = np.cov(X.T)
    expected_returns = np.mean(X, axis=0)

    optimal_weights, _ = pso(
        objective_function,
        lb,
        ub,
        maxiter=2000,
        swarmsize=80,
        c1=2,
        c2=2,
        args=(cov_matrix, expected_returns, lambda_0, lambda_1)
    )
    return optimal_weights

# -----------------------------------
# API Endpoint
# -----------------------------------
@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    data = request.json

    returns_data = np.array(data['returns_data'])
    lambda_0 = data['lambda_0']
    lambda_1 = data['lambda_1']

    optimized_weights = optimize_portfolio_sa(returns_data, lambda_0, lambda_1)

    var = compute_VaR(optimized_weights, returns_data)
    cvar = compute_CVaR(optimized_weights, returns_data)

    return jsonify({
        'optimized_weights': optimized_weights.tolist(),
        'risk_metrics': {
            'VaR': var,
            'CVaR': cvar
        }
    })

# -----------------------------------
# Run
# -----------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
