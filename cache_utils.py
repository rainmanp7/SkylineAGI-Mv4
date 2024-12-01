# Beginning of cache_utils.py
import hashlib
from functools import lru_cache
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Dictionary to store the previous hash of data and hyperparameters
cache_conditions = {
    'X_train_hash': None,
    'y_train_hash': None,
    'hyperparameters_hash': None,
}

# Function to compute hash of data
def compute_hash(data):
    """
    Computes a SHA256 hash of the given data.
    """
    try:
        return hashlib.sha256(str(data).encode()).hexdigest()
    except Exception as e:
        print(f"Error hashing data: {e}")
        return None

# Bayesian fit function with caching
@lru_cache(maxsize=512)
def cached_bayesian_fit(X_train, y_train, hyperparameters):
    """
    Performs Bayesian model fitting with caching.
    
    Parameters:
    - X_train: Training input data
    - y_train: Training output data
    - hyperparameters: Dictionary of model hyperparameters

    Returns:
    - model: Trained Bayesian model
    """
    print("Performing Bayesian fit with Gaussian Process...")
    
    try:
        # Extract hyperparameters
        kernel_constant = hyperparameters.get("kernel_constant", 1.0)
        kernel_length_scale = hyperparameters.get("kernel_length_scale", 1.0)
        
        # Define Gaussian Process kernel
        kernel = C(kernel_constant, (1e-3, 1e3)) * RBF(kernel_length_scale, (1e-2, 1e2))
        
        # Create and train the Gaussian Process model
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp_model.fit(X_train, y_train)
        
        print("Bayesian fit completed successfully.")
        return gp_model

    except Exception as e:
        print(f"Error during Bayesian fit: {e}")
        return None

# Function to invalidate cache if conditions change
def invalidate_cache_if_changed(current_X_train, current_y_train, current_hyperparameters):
    """
    Invalidates the cache if the hash of the input data or hyperparameters has changed.
    """
    try:
        current_X_train_hash = compute_hash(current_X_train)
        current_y_train_hash = compute_hash(current_y_train)
        current_hyperparameters_hash = compute_hash(current_hyperparameters)

        # Check if any condition has changed
        if (cache_conditions['X_train_hash'] != current_X_train_hash or
            cache_conditions['y_train_hash'] != current_y_train_hash or
            cache_conditions['hyperparameters_hash'] != current_hyperparameters_hash):

            # Clear the cached results
            cached_bayesian_fit.cache_clear()

            # Update cache conditions
            cache_conditions['X_train_hash'] = current_X_train_hash
            cache_conditions['y_train_hash'] = current_y_train_hash
            cache_conditions['hyperparameters_hash'] = current_hyperparameters_hash

            print("Cache invalidated due to data or parameter changes.")

    except Exception as e:
        print(f"Error in cache invalidation: {e}")
# End of cache_utils.py
