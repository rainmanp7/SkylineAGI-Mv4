
```python
# Beginning of optimization.py
# Last updated: Nov 23, 2024
# Purpose: Handles hyperparameter optimization using Bayesian methods and dynamic adjustment of the search space.
# modified and adjusted Nov 23 7:42 PM

import os
import json
from typing import Dict, Tuple
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from models import create_model  # Ensure this is implemented and matches your architecture.
from utils import load_data, compute_complexity_factor  # Ensure these utilities exist.

class HyperparameterOptimization:
    """
    Class for handling hyperparameter optimization tasks with dynamic search space adjustment.
    """

    def __init__(self, config_file: str = 'config.json'):
        """
        Initialize the optimization class and load configuration from the specified file.
        """
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration settings from a JSON file.
        
        Args:
            config_file (str): Path to the configuration file.
        
        Returns:
            dict: Configuration data.
        """
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON in '{config_file}'.")

    def adjust_search_space(self, current_space: Dict, performance: float, threshold: float = 0.1) -> Dict:
        """
        Adjusts the hyperparameter search space dynamically based on model performance.
        
        Args:
            current_space (dict): Current search space.
            performance (float): Model's current performance score.
            threshold (float): Performance threshold for adjustments.
        
        Returns:
            dict: Adjusted search space.
        """
        adjusted_space = current_space.copy()
        if performance > threshold:
            adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low * 0.1,
                                                   current_space['learning_rate'].high * 10,
                                                   prior='log-uniform')
        else:
            adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low,
                                                   current_space['learning_rate'].high * 0.1,
                                                   prior='log-uniform')
        return adjusted_space

    async def parallel_bayesian_optimization(self, X_train, y_train, X_val, y_val, n_iterations: int) -> Tuple[Dict, float, float]:
        """
        Performs Bayesian optimization in parallel to determine optimal hyperparameters.
        
        Args:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation data.
            y_val (numpy.ndarray): Validation labels.
            n_iterations (int): Number of iterations for optimization.
        
        Returns:
            tuple: Best parameters, best performance score, and quality score.
        """
        # Initial hyperparameter search space
        param_space = {
            'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(5, 15),
            'subsample': Real(0.5, 1.0),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
        }

        # Gaussian Process Regressor setup
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10, random_state=self.config['random_seed'])

        # Bayesian optimization using skopt
        optimizer = BayesSearchCV(estimator=gpr,
                                   search_spaces=param_space,
                                   n_iter=n_iterations,
                                   random_state=self.config['random_seed'],
                                   n_jobs=self.config['num_parallel_jobs'])

        # Perform optimization without eval_set
        optimizer.fit(X_train, y_train)

        # Best hyperparameters and scores
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Calculate complexity and quality scores
        complexity_factor = compute_complexity_factor(best_params)
        best_quality_score = 1 / complexity_factor

        return best_params, best_score, best_quality_score

    def perform_optimization(self, X_train, y_train, X_val, y_val):
        """
        Manages the overall optimization process, including adjusting the search space.
        
        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels.
        
        Returns:
            tuple: Best parameters, best performance score, and quality score.
        """
        
         # Ensure initial model setup
         model = create_model(self.config['model_type'], self.config['model_params'])

         # Initial Bayesian optimization
         best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(
             X_train, y_train, X_val, y_val,self.config['num_optimization_steps'])

         # Adjust search space based on results
         adjusted_search_space = self.adjust_search_space(param_space, best_score)

         # Re-run optimization with adjusted space
         best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(
             X_train, y_train, X_val, y_val,self.config['num_optimization_steps'])

         # Final model creation and saving
         best_model = create_model(self.config['model_type'], best_params)
         model_save_path = os.path.join(self.config['model_save_dir'], 'best_model.pkl')
         best_model.save(model_save_path)

         return best_params,best_score,best_quality_score


# End of optimization.py
```
