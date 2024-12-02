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
# updated matching dataset. Dec 3rd.
class EnhancedModelSelector(ModelSelector):
    def __init__(self):
        super().__init__()
        # Updated to match the 9 tiers and the complexity ranges provided
        self.complexity_tiers = {
            # 1st Section
    'easy': (1111, 1389),
    'simp': (1390, 1668),
    'norm': (1669, 1947),
    # 2nd Section
    'mods': (1948, 2226),
    'hard': (2227, 2505),
    'para': (2506, 2784),
    # 3rd Section
    'vice': (2785, 3063),
    'zeta': (3064, 3342),
    'tetr': (3343, 3621),
    # 4th Section
    'eafv': (3622, 3900),
    'sipo': (3901, 4179),
    'nxxm': (4180, 4458),
    # 5th Section
    'mids': (4459, 4737),
    'haod': (4738, 5016),
    'parz': (5017, 5295),
    # 6th Section
    'viff': (5296, 5574),
    'zexa': (5575, 5853),
    'sip8': (5854, 6132),
    # 7th Section
    'nxVm': (6133, 6411),
    'Vids': (6412, 6690),
    'ha3d': (6691, 6969),
    # 8th Section
    'pfgz': (6970, 7248),
    'vpff': (7249, 7527),
    'z9xa': (7528, 7806),
    # 9th Section
    'Tipo': (7807, 8085),
    'nxNm': (8086, 8364),
    'mPd7': (8365, 9918)
        }
        
        # Define model configurations for each complexity range
        self.model_configs = {
            # 1st Section
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

            # 2nd Section
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

            # 3rd Section
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
            'tetr': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (200, 100, 50)},
                complexity_level='tetr',
                suggested_iterations=1000,
                suggested_metric=r2_score
            ),

            # 4th Section
            'eafv': ModelConfig(
                model_class=LinearRegression,
                default_params={},
                complexity_level='eafv',
                suggested_iterations=100,
                suggested_metric=mean_squared_error
            ),
            'sipo': ModelConfig(
                model_class=Ridge,
                default_params={'alpha': 1.0},
                complexity_level='sipo',
                suggested_iterations=200,
                suggested_metric=mean_squared_error
            ),
            'nxxm': ModelConfig(
                model_class=Lasso,
                default_params={'alpha': 1.0},
                complexity_level='nxxm',
                suggested_iterations=300,
                suggested_metric=mean_absolute_error
            ),

            # 5th Section
            'mids': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 50},
                complexity_level='mids',
                suggested_iterations=400,
                suggested_metric=mean_absolute_error
            ),
            'haod': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 100},
                complexity_level='haod',
                suggested_iterations=500,
                suggested_metric=mean_absolute_error
            ),
            'parz': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 100},
                complexity_level='parz',
                suggested_iterations=600,
                suggested_metric=r2_score
            ),

            # 6th Section
            'viff': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 200},
                complexity_level='viff',
                suggested_iterations=700,
                suggested_metric=r2_score
            ),
            'zexa': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (100, 50)},
                complexity_level='zexa',
                suggested_iterations=800,
                suggested_metric=r2_score
            ),
            'sip8': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (200, 100, 50)},
                complexity_level='sip8',
                suggested_iterations=1000,
                suggested_metric=r2_score
            ),

            # 7th Section
            'nxVm': ModelConfig(
                model_class=LinearRegression,
                default_params={},
                complexity_level='nxVm',
                suggested_iterations=100,
                suggested_metric=mean_squared_error
            ),
            'Vids': ModelConfig(
                model_class=Ridge,
                default_params={'alpha': 1.0},
                complexity_level='Vids',
                suggested_iterations=200,
                suggested_metric=mean_squared_error
            ),
            'ha3d': ModelConfig(
                model_class=Lasso,
                default_params={'alpha': 1.0},
                complexity_level='ha3d',
                suggested_iterations=300,
                suggested_metric=mean_absolute_error
            ),

            # 8th Section
            'pfgz': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 50},
                complexity_level='pfgz',
                suggested_iterations=400,
                suggested_metric=mean_absolute_error
            ),
            'vpff': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={'n_estimators': 100},
                complexity_level='vpff',
                suggested_iterations=500,
                suggested_metric=mean_absolute_error
            ),
            'z9xa': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 100},
                complexity_level='z9xa',
                suggested_iterations=600,
                suggested_metric=r2_score
            ),

            # 9th Section
            'Tipo': ModelConfig(
                model_class=GradientBoostingRegressor,
                default_params={'n_estimators': 200},
                complexity_level='Tipo',
                suggested_iterations=700,
                suggested_metric=r2_score
            ),
            'nxNm': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (100, 50)},
                complexity_level='nxNm',
                suggested_iterations=800,
                suggested_metric=r2_score
            ),
            'mPd7': ModelConfig(
                model_class=MLPRegressor,
                default_params={'hidden_layer_sizes': (200, 100, 50)},
                complexity_level='mPd7',
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
