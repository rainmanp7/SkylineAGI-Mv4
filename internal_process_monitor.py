# Internal Process Monitor Start
# Internal Metacognitive Introspection 
# Nov6 self aware base preparation.
# install required dependencies
# pip install psutil

````python

import psutil
import time
from collections import deque
from typing import Dict, Optional

class InternalProcessMonitor:
    def __init__(self, max_history_size=100):
        # Existing histories
        self.cpu_usage_history = deque(maxlen=max_history_size)
        self.memory_usage_history = deque(maxlen=max_history_size)
        self.task_queue_length_history = deque(maxlen=max_history_size)
        self.knowledge_base_updates_history = deque(maxlen=max_history_size)
        self.model_training_time_history = deque(maxlen=max_history_size)
        self.model_inference_time_history = deque(maxlen=max_history_size)
        
        # Add new tracking elements
        self.timestamps = deque(maxlen=max_history_size)
        self.current_task: Optional[str] = None
        self.task_metrics: Dict[str, Dict] = {}

    def start_task_monitoring(self, task_name: str):
        self.current_task = task_name
        if task_name not in self.task_metrics:
            self.task_metrics[task_name] = {
                "start_time": time.time(),
                "cpu_usage": [],
                "memory_usage": []
            }

    def end_task_monitoring(self):
        if self.current_task:
            self.task_metrics[self.current_task]["end_time"] = time.time()
            self.current_task = None

    def monitor_cpu_usage(self):
        current_cpu = psutil.cpu_percent()
        current_time = time.time()
        self.cpu_usage_history.append(current_cpu)
        self.timestamps.append(current_time)
        
        if self.current_task:
            self.task_metrics[self.current_task]["cpu_usage"].append(current_cpu)

    def monitor_memory_usage(self):
        current_memory = psutil.virtual_memory().percent
        self.memory_usage_history.append(current_memory)
        
        if self.current_task:
            self.task_metrics[self.current_task]["memory_usage"].append(current_memory)

    def monitor_task_queue_length(self, queue_size):
        self.task_queue_length_history.append(queue_size)

    def monitor_knowledge_base_updates(self, num_updates):
        self.knowledge_base_updates_history.append(num_updates)

    def monitor_model_training_time(self, training_time):
        self.model_training_time_history.append(training_time)

    def monitor_model_inference_time(self, inference_time):
        self.model_inference_time_history.append(inference_time)

    def get_historical_data(self):
        base_data = {
            "cpu_usage": list(self.cpu_usage_history),
            "memory_usage": list(self.memory_usage_history),
            "task_queue_length": list(self.task_queue_length_history),
            "knowledge_base_updates": list(self.knowledge_base_updates_history),
            "model_training_time": list(self.model_training_time_history),
            "model_inference_time": list(self.model_inference_time_history),
            "timestamps": list(self.timestamps)
        }
        
        # Add task-specific metrics
        base_data["task_metrics"] = self.task_metrics
        return base_data

    def generate_task_report(self, task_name: str) -> Dict:
        if task_name not in self.task_metrics:
            return {"error": f"No data for task: {task_name}"}
            
        task_data = self.task_metrics[task_name]
        return {
            "task_name": task_name,
            "duration": task_data["end_time"] - task_data["start_time"],
            "avg_cpu": sum(task_data["cpu_usage"]) / len(task_data["cpu_usage"]) if task_data["cpu_usage"] else 0,
            "avg_memory": sum(task_data["memory_usage"]) / len(task_data["memory_usage"]) if task_data["memory_usage"] else 0,
            "peak_cpu": max(task_data["cpu_usage"]) if task_data["cpu_usage"] else 0,
            "peak_memory": max(task_data["memory_usage"]) if task_data["memory_usage"] else 0
        }
# ````
# Internal Process Monitor end.

