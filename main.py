# main.py

# 9 Base tier ready implemented Nov9
# This uses not a random but specific 
# Beginning of main.py
# Nov21 domain start
# Nov 23 main loop start diag added.
# Nov 25 adding database functionality.

import logging
import asyncio
from typing import List

# Import necessary modules
from domain_knowledge_base import DomainKnowledgeBase
from agi_config import AGIConfiguration
from internal_process_monitor import InternalProcessMonitor
from cross_domain_generalization import CrossDomainGeneralization
from complexity import ComplexityAnalyzer
from optimization import adjust_search_space, parallel_bayesian_optimization
from knowledge_base import TieredKnowledgeBase, KnowledgeBase
from main_knowledge_base import MainKnowledgeBase
from metacognitive_manager import MetaCognitiveManager
from memory_manager import MemoryManager
from uncertainty_quantification import UncertaintyQuantification
from async_process_manager import AsyncProcessManager
from models import ProcessTask, model_validator, SkylineAGIModel
from parallel_bayesian_optimization_wrks import BayesianOptimizer, ParallelBayesianOptimization  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run Startup Diagnostics
def run_startup_diagnostics():
    """ Perform a series of startup diagnostics to ensure the system is operational. """
    print("Running startup diagnostics...")
    try:
        config = AGIConfiguration()
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False

    try:
        knowledge_base = KnowledgeBase()
        print("Knowledge base initialized successfully.")
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return False

    try:
        model = SkylineAGIModel(config)
        print("AGI model created successfully.")
    except Exception as e:
        print(f"Error creating AGI model: {e}")
        return False

    try:
        process_monitor = InternalProcessMonitor()
        print("Process monitor initialized successfully.")
    except Exception as e:
        print(f"Error initializing process monitor: {e}")
        return False

    try:
        optimizer = BayesianOptimizer()  # Use the correct BayesianOptimizer implementation
        print("Bayesian optimizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing Bayesian optimizer: {e}")
        return False

    try:
        cross_domain_generalization = CrossDomainGeneralization()
        print("Cross-domain generalization initialized successfully.")
    except Exception as e:
        print(f"Error initializing cross-domain generalization: {e}")
        return False

    try:
        metacognitive_manager = MetaCognitiveManager()
        print("Metacognitive manager initialized successfully.")
    except Exception as e:
        print(f"Error initializing metacognitive manager: {e}")
        return False

    print("Startup diagnostics completed successfully.")
    return True


class SkylineAGI:
    def __init__(self):
        self.config = AGIConfiguration()
        self.knowledge_base = DomainKnowledgeBase()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.internal_monitor = InternalProcessMonitor()
        self.cross_domain_generator = CrossDomainGeneralization(self.knowledge_base, self.config)

    async def process_domain(self, domain: str):
        """ Asynchronously process a specific domain """
        try:
            complexity_factor = self.get_complexity_factor(domain)
            datasets = self.knowledge_base.get_dataset_paths(domain)
            for i, dataset in enumerate(datasets):
                await self.process_dataset(domain, dataset, complexity_factor, i)
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {e}")

    async def process_dataset(self, domain: str, dataset: str, complexity: float, index: int):
        """ Process individual datasets with complexity-aware optimization """
        try:
            optimizer = BayesianOptimizer()  # Instantiate the optimizer
            optimized_params = optimizer.optimize(dataset, complexity)  # Hypothetical optimization method

            # Load data with optimized parameters
            loaded_data = self.knowledge_base.load_domain_dataset(domain, index, optimized_params)
            self.internal_monitor.track_dataset_processing(dataset, complexity)
            self.cross_domain_generator.analyze_dataset(loaded_data)
        except Exception as e:
            logger.error(f"Dataset processing error: {e}", exc_info=True)

    def get_complexity_factor(self, domain: str) -> float:
        """ Determine complexity factor based on domain characteristics """
        try:
            base_complexity = self.config.get_dynamic_setting('complexity_factor', 10)
            domain_complexity = self.complexity_analyzer.analyze_domain(domain)
            return base_complexity * domain_complexity
        except Exception as e:
            logger.warning(f"Complexity calculation error: {e}", exc_info=True)
            return 10.0  # Default fallback


async def main():
    """ Main asynchronous execution entry point """
    process_manager = AsyncProcessManager()

    try:
        # Initialize Skyline AGI
        agi = SkylineAGI()

        # Define domains to process
        domains = ['Math', 'Science']  # Add or modify domains as needed

        # Filter domains to skip ones without datasets
        valid_domains = [domain for domain in domains if agi.knowledge_base.get_dataset_paths(domain)]

        if not valid_domains:
            logger.warning("No valid domains with datasets found. Exiting...")
            return

        # Parallel optimization for valid domains
        tasks = [
            asyncio.create_task(
                ParallelBayesianOptimization(domain, agi.knowledge_base.get_dataset_paths(domain))
            )
            for domain in valid_domains
        ]
        # Wait for all domain processing to complete
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)


async def run_monitoring(internal_monitor, process_manager, knowledge_base):
    """ Background monitoring loop """
    try:
        last_update_count = 0

        while True:
            internal_monitor.monitor_cpu_usage()
            internal_monitor.monitor_memory_usage()

            if not process_manager.task_queue.empty():
                internal_monitor.monitor_task_queue_length(process_manager.task_queue.qsize())

            current_update_count = len(knowledge_base.get_recent_updates())
            internal_monitor.monitor_knowledge_base_updates(current_update_count - last_update_count)
            last_update_count = current_update_count

            if hasattr(model_validator, 'metrics_history') and "model_key" in model_validator.metrics_history:
                metrics = model_validator.metrics_history["model_key"][-1]
                internal_monitor.monitor_model_training_time(metrics.training_time)
                internal_monitor.monitor_model_inference_time(metrics.prediction_latency)

            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Monitoring task canceled. Shutting down...")


if __name__ == "__main__":
    if not run_startup_diagnostics():
        print("Startup diagnostics failed. Exiting the application.")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Exiting the application.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
