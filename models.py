# models.py
# memory added fitted Nov14
# modified Friday Dec6th 2024

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time
import memory_profiler
import logging
from collections import defaultdict
import torch.nn as nn
from memory_manager import MemoryManager
from attention_mechanism import MultiHeadAttention, ContextAwareAttention

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkylineModel(nn.Module):
    """
    Skyline Model with Multi-Head Attention and Context-Aware Attention.
    
    Args:
    - input_size (int): Input size of the model.
    - num_heads (int): Number of heads for Multi-Head Attention.
    - context_size (int): Context size for Context-Aware Attention.
    """
    
    def __init__(self, input_size: int, num_heads: int, context_size: int):
        super(SkylineModel, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size=input_size, num_heads=num_heads)
        self.context_aware_attention = ContextAwareAttention(input_size=input_size, context_size=context_size)
        # Add other layers, such as feedforward, residual connections, etc.

    def forward(self, x: Any, context: Any) -> Any:
        """
        Forward pass of the Skyline Model.
        
        Args:
        - x (Any): Input to the model.
        - context (Any): Context for Context-Aware Attention.
        
        Returns:
        - Any: Output of the model.
        """
        # Apply multi-head attention
        x = self.multi_head_attention(x)

        # Apply context-aware attention
        x = self.context_aware_attention(x, context)

        # Apply other layers
        #...

        return x

class BaseModel:
    """
    Base Model with fit and predict methods.
    """
    
    def __init__(self):
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the given data.
        
        Args:
        - X (np.ndarray): Features.
        - y (np.ndarray): Target variable.
        """
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the given data.
        
        Args:
        - X (np.ndarray): Features.
        
        Returns:
        - np.ndarray: Predictions.
        """
        raise NotImplementedError

@dataclass
class ModelMetrics:
    """
    Data class to hold model metrics.
    
    Attributes:
    - mae (float): Mean Absolute Error.
    - mse (float): Mean Squared Error.
    - r2 (float): R-Squared.
    - training_time (float): Time taken for training.
    - memory_usage (float): Memory usage during training.
    - prediction_latency (float): Latency of predictions.
    """
    mae: float
    mse: float
    r2: float
    training_time: float
    memory_usage: float
    prediction_latency: float

class ModelValidator:
    """
    Validator for models with metrics calculation and storage.
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def validate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_key: str
    ) -> ModelMetrics:
        """
        Validate a model and calculate its metrics.
        
        Args:
        - model (Any): Model to validate.
        - X_val (np.ndarray): Validation features.
        - y_val (np.ndarray): Validation target variable.
        - model_key (str): Key for the model in metrics history.
        
        Returns:
        - ModelMetrics: Calculated metrics for the model.
        """
        try:
            start_time = time.time()
            memory_usage = memory_profiler.memory_usage()
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, time.time() - start_time, memory_usage)
            metrics.prediction_latency = self._measure_prediction_latency(model, X_val)
            
            # Store metrics
            self.metrics_history[model_key].append(metrics)
            
            return metrics
        
        except Exception as e:
            logging.error(f"Validation failed for {model_key}: {str(e)}")
            raise

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        training_time: float,
        memory_usage: List[float]
    ) -> ModelMetrics:
        """
        Calculate MAE, MSE, R2, training time, and memory usage.
        
        Args:
        - y_true (np.ndarray): True values.
        - y_pred (np.ndarray): Predicted values.
        - training_time (float): Time taken for training.
        - memory_usage (List[float]): Memory usage during training.
        
        Returns:
        - ModelMetrics: Calculated metrics.
        """
        return ModelMetrics(
            mae=mean_absolute_error(y_true, y_pred),
            mse=mean_squared_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            training_time=training_time,
            memory_usage=max(memory_usage) - min(memory_usage),
            prediction_latency=0.0  # To be updated later
        )

    def _measure_prediction_latency(
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 100
    ) -> float:
        """
        Measure the latency of predictions.
        
        Args:
        - model (Any): Model to measure latency for.
        - X (np.ndarray): Features for predictions.
        - n_iterations (int): Number of iterations for latency measurement.
        
        Returns:
        - float: Average latency of predictions.
        """
        latencies = []
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X[:100])  # Use small batch for latency test
            latencies.append(time.time() - start_time)
        return np.mean(latencies)

@dataclass
class ExpandedModelMetrics(ModelMetrics):
    """
    Expanded model metrics with loss, accuracy, and feature importance.
    
    Attributes:
    - loss (float): Loss of the model.
    - accuracy (float): Accuracy of the model.
    - feature_importance (Dict[str, float]): Feature importance.
    """
    loss: float
    accuracy: float
    feature_importance: Dict[str, float]

class ExpandedModelValidator(ModelValidator):
    """
    Expanded model validator with additional metrics calculation.
    """
    
    def validate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_key: str
    ) -> ExpandedModelMetrics:
        """
        Validate a model and calculate its expanded metrics.
        
        Args:
        - model (Any): Model to validate.
        - X_val (np.ndarray): Validation features.
        - y_val (np.ndarray): Validation target variable.
        - model_key (str): Key for the model in metrics history.
        
        Returns:
        - ExpandedModelMetrics: Calculated expanded metrics.
        """
        # Call the original validate_model method
        metrics = super().validate_model(model, X_val, y_val, model_key)
        
        try:
            # Compute additional metrics
            loss, accuracy = model.evaluate(X_val, y_val)
            feature_importance = model.feature_importances_
            
            # Create the expanded metrics object
            expanded_metrics = ExpandedModelMetrics(
                mae=metrics.mae,
                mse=metrics.mse,
                r2=metrics.r2,
                training_time=metrics.training_time,
                memory_usage=metrics.memory_usage,
                prediction_latency=metrics.prediction_latency,
                loss=loss,
                accuracy=accuracy,
                feature_importance=feature_importance
            )
            
            # Store the expanded metrics
            self.metrics_history[model_key].append(expanded_metrics)
            return expanded_metrics
        
        except AttributeError as e:
            logging.warning(f"Could not compute expanded metrics: {str(e)}")
            return metrics
            
