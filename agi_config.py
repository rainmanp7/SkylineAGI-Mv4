import json
from typing import Dict, Any

class AGIConfiguration:
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def get_dynamic_setting(self, key: str, default=None):
        return self.config.get(key, default)
