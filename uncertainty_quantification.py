
# Created on Nov14 2024 9:40pm
# Uncertainty Code implementation.

import numpy as np
import scipy.stats as stats

class UncertaintyQuantification:
    def __init__(self, config=None):
        """
        Initialize Uncertainty Quantification module
        
        :param config: Configuration settings for uncertainty handling
        """
        self.config = config or {}
        self.epistemic_uncertainty = 0.0
        self.aleatoric_uncertainty = 0.0
        self.confidence_level = 0.0

    def estimate_epistemic(self, model_predictions, ensemble_predictions):
        """
        Estimate epistemic uncertainty using variance across ensemble predictions
        
        :param model_predictions: Predictions from a single model
        :param ensemble_predictions: Predictions from multiple models
        :return: Epistemic uncertainty score
        """
        try:
            epistemic_var = np.var(ensemble_predictions, axis=0)
            self.epistemic_uncertainty = np.mean(epistemic_var)
            return self.epistemic_uncertainty
        except Exception as e:
            print(f"Error in epistemic uncertainty estimation: {e}")
            return None

    def handle_aleatoric(self, data_variance):
        """
        Estimate aleatoric uncertainty based on data variance
        
        :param data_variance: Variance in the input data
        :return: Aleatoric uncertainty score
        """
        try:
            self.aleatoric_uncertainty = np.sqrt(data_variance)
            return self.aleatoric_uncertainty
        except Exception as e:
            print(f"Error in aleatoric uncertainty handling: {e}")
            return None

    def calibrate_confidence(self, predictions, true_labels):
        """
        Calibrate model confidence using prediction probabilities
        
        :param predictions: Model prediction probabilities
        :param true_labels: Actual labels
        :return: Confidence calibration metric
        """
        try:
            from sklearn.calibration import calibration_curve
            
            prob_true, prob_pred = calibration_curve(true_labels, predictions)
            self.confidence_level = 1 - np.mean(np.abs(prob_true - prob_pred))
            return self.confidence_level
        except Exception as e:
            print(f"Error in confidence calibration: {e}")
            return None

    def make_decision_with_uncertainty(self, predictions, uncertainty_threshold=0.5):
        """
        Make decisions while considering uncertainty
        
        :param predictions: Model predictions
        :param uncertainty_threshold: Threshold for uncertainty tolerance
        :return: Decision and uncertainty status
        """
        try:
            total_uncertainty = self.epistemic_uncertainty + self.aleatoric_uncertainty
            
            if total_uncertainty < uncertainty_threshold:
                decision = np.mean(predictions)
                confidence = "High"
            else:
                decision = None  # Defer decision
                confidence = "Low"
            
            return {
                "decision": decision,
                "confidence": confidence,
                "total_uncertainty": total_uncertainty
            }
        except Exception as e:
            print(f"Error in decision-making with uncertainty: {e}")
            return None

    def log_uncertainty_metrics(self):
        """
        Log uncertainty metrics for tracking and analysis
        """
        uncertainty_log = {
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "confidence_level": self.confidence_level
        }
        return uncertainty_log

