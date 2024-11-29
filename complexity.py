# 9 Base tier implemented Nov9
# Beginning of complexity.py
# Nov9 RRL Memory Module 
# Quality change applied nov12

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
import logging

from .knowledge_base import TieredKnowledgeBase
from .assimilation_memory_module import AssimilationMemoryModule

class EnhancedModelSelector:
    def __init__(self, knowledge_base: TieredKnowledgeBase, assimilation_module: AssimilationMemoryModule):
        self.knowledge_base = knowledge_base
        self.assimilation_module = assimilation_module

# Quality change begin
class ModelConfig:
    model_class: Any
    default_params: Dict[str, Any]
    complexity_level: str
    suggested_iterations: int
    suggested_metric: Callable
    quality_score: float  # New attribute to store the quality score
# Quality change end.

class EnhancedModelSelector(ModelSelector):
    def __init__(self):
        super().__init__()
        # Updated to match the 9 tiers
        self.complexity_tiers = {
            'easy': (1, 3, mean_squared_error, 100),    # Simplest models, basic metric
            'simp': (4, 7, mean_squared_error, 200),    
            'norm': (8, 11, mean_absolute_error, 300),
            'mods': (12, 15, mean_absolute_error, 400),
            'hard': (16, 19, mean_absolute_error, 500),
            'para': (20, 23, r2_score, 600),
            'vice': (24, 27, r2_score, 700),
            'zeta': (28, 31, r2_score, 800),
            'tetris': (32, 35, r2_score, 1000)         # Most complex models, sophisticated metric
        }
        
        # Define model configurations for each complexity range
        self.model_configs = {
            'easy': ModelConfig(
                model_class=LinearRegression,
                default_params={},
                complexity_level='easy',
                suggested_iterations=100,
                suggested_metric=mean_squared_error
            ),
            'simp': ModelConfig(
                model_class=Ridge,
                default_params={'alpha': 1.0},
                complexity_level='simp',
                suggested_iterations=200,
                suggested_metric=mean_squared_error
            ),
            'norm': ModelConfig(
                model_class=Lasso,
                default_params={'alpha': 1.0},
                complexity_level='norm',
                suggested_iterations=300,
                suggested_metric=mean_absolute_error
            ),
            'mods': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 50},
                complexity_level='mods',
                suggested_iterations=400,
                suggested_metric=mean_absolute_error
            ),
            'hard': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 100},
                complexity_level='hard',
                suggested_iterations=500,
                suggested_metric=mean_absolute_error
            ),
            'para': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 100},
                complexity_level='para',
                suggested_iterations=600,
                suggested_metric=r2_score
            ),
            'vice': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 200},
                complexity_level='vice',
                suggested_iterations=700,
                suggested_metric=r2_score
            ),
            'zeta': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (100, 50)},
                complexity_level='zeta',
                suggested_iterations=800,
                suggested_metric=r2_score
            ),
            'tetris': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (200, 100, 50)},
                complexity_level='tetris',
                suggested_iterations=1000,
                suggested_metric=r2_score
            )
        }

    def _get_tier(self, complexity_factor: float) -> str:
        """Determine which tier a complexity factor belongs to."""
        for tier, (min_comp, max_comp, _, _) in self.complexity_tiers.items():
            if min_comp <= complexity_factor <= max_comp:
                return tier
        # Fallback to 'easy' if out of range
        return 'easy'

    def choose_model_and_config(
        self,
        complexity_factor: float,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Callable, int]:
        """
        Enhanced model selection based on the 9-tier complexity system.
        
        Args:
            complexity_factor: Float between 1 and 35
            custom_params: Optional custom parameters for the model
            
        Returns:
            Tuple[model, evaluation_metric, num_iterations]
        """
        try:
            # Ensure complexity factor is within bounds
            complexity_factor = max(1, min(35, complexity_factor))
            
            # Get appropriate tier
            tier = self._get_tier(complexity_factor)
            config = self.model_configs[tier]
            
            # Initialize model with appropriate parameters
            params = config.default_params.copy()
            if custom_params:
                params.update(custom_params)
            model = config.model_class(**params)
            
            # Get corresponding metric and iterations
            _, _, metric, iterations = self.complexity_tiers[tier]
            
            logging.info(
                f"Selected model configuration:\n"
                f"Tier: {tier}\n"
                f"Complexity Factor: {complexity_factor}\n"
                f"Model: {config.model_class.__name__}\n"
                f"Metric: {metric.__name__}\n"
                f"Iterations: {iterations}"
            )
            
            return model, metric, iterations
            
        except Exception as e:
            logging.error(f"Error in enhanced model selection: {str(e)}", exc_info=True)
            # Fallback to simplest configuration
            return (
                self.model_configs['easy'].model_class(),
                mean_squared_error,
                100
            )

    def get_tier_details(self, tier: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tier.
        
        Args:
            tier: The tier name (easy, simp, norm, etc.)
            
        Returns:
            Dictionary containing tier details
        """
        if tier in self.model_configs:
            config = self.model_configs[tier]
            min_comp, max_comp, metric, iterations = self.complexity_tiers[tier]
            return {
                'complexity_range': (min_comp, max_comp),
                'model_class': config.model_class.__name__,
                'default_params': config.default_params,
                'iterations': iterations,
                'metric': metric.__name__
            }
        return None
# End of complexity.py