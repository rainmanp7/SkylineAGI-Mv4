# Cache_utils.py start

import functools
import hashlib
import numpy as np
from functools import lru_cache
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

# Caching utilities
cache_conditions = {
    'X_train_hash': None,
    'y_train_hash': None,
    'hyperparameters_hash': None,
}

def compute_hash(data):
    """
    Computes a SHA256 hash of the given data.
    """
    try:
        return hashlib.sha256(str(data).encode()).hexdigest()
    except Exception as e:
        print(f"Error hashing data: {e}")
        return None

def cached_bayesian_fit(func):
    """
    Decorator for caching Bayesian optimization results.
    
    Features:
    - Caches results based on input data hash.
    - Invalidates cache when inputs change.
    - Improves computational efficiency.
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Compute hash for current inputs
        input_hash = compute_hash(args)
        hyperparameters_hash = compute_hash(kwargs.get("hyperparameters", {}))

        # Check cache conditions
        if input_hash in cache and cache_conditions['hyperparameters_hash'] == hyperparameters_hash:
            print("Using cached results...")
            return cache[input_hash]
        
        # Execute and cache result
        result = func(self, *args, **kwargs)
        cache[input_hash] = result
        cache_conditions['hyperparameters_hash'] = hyperparameters_hash
        
        return result
    
    def cache_clear():
        """Clear the entire cache."""
        cache.clear()
    
    wrapper.cache_clear = cache_clear
    return wrapper

# Main Hyperparameter Optimization Class
class BayesianHyperparameterTuning:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def _define_search_space(self):
        """
        Define the search space for Bayesian hyperparameter tuning.
        """
        return [
            Real(0.1, 10.0, name="kernel_constant"),  # ConstantKernel
            Real(0.1, 10.0, name="kernel_length_scale"),  # RBF kernel
            Integer(1, 100, name="n_restarts_optimizer")  # GPR hyperparameter
        ]

    @use_named_args(self._define_search_space())
    def _bayesian_objective(self, kernel_constant, kernel_length_scale, n_restarts_optimizer):
        """
        Objective function for Bayesian optimization.
        """
        kernel = C(kernel_constant) * RBF(kernel_length_scale)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        gpr.fit(self.X_train, self.y_train)
        y_pred = gpr.predict(self.X_val)
        score = r2_score(self.y_val, y_pred)
        return -score  # Minimize the negative R² score

    @cached_bayesian_fit
    def tune_hyperparameters(self, n_calls=50, random_state=42, hyperparameters=None):
        """
        Perform Bayesian hyperparameter tuning with caching.
        """
        search_space = self._define_search_space()
        res_gp = gp_minimize(self._bayesian_objective, search_space, n_calls=n_calls, random_state=random_state)
        best_hyperparameters = {
            "kernel_constant": res_gp.x[0],
            "kernel_length_scale": res_gp.x[1],
            "n_restarts_optimizer": res_gp.x[2]
        }
        best_score = -res_gp.fun
        return best_hyperparameters, best_score

    def evaluate_best_model(self, best_hyperparameters):
        """
        Evaluate the best model with the tuned hyperparameters.
        """
        kernel = C(best_hyperparameters["kernel_constant"]) * RBF(best_hyperparameters["kernel_length_scale"])
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=best_hyperparameters["n_restarts_optimizer"])
        gpr.fit(self.X_train, self.y_train)
        y_pred = gpr.predict(self.X_val)
        mse = mean_squared_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)
        return {"MSE": mse, "R2": r2}

# Example usage
if __name__ == "__main__":
    # Sample data
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 5)
    y_val = np.random.rand(20)

    # Initialize tuner
    tuner = BayesianHyperparameterTuning(X_train, y_train, X_val, y_val)

    # Perform tuning
    best_hyperparameters, best_score = tuner.tune_hyperparameters(n_calls=50, random_state=42)
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best R^2 Score:", best_score)

    # Evaluate model
    evaluation_metrics = tuner.evaluate_best_model(best_hyperparameters)
    print("Evaluation Metrics (MSE and R^2):", evaluation_metrics)

# Cache_utils.py end
