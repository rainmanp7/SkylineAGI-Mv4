
# Beginning of parallel_utils.py
from multiprocessing import Pool
import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class LearningTask:
    strategy_name: str
    data: Any
    parameters: Dict[str, Any]
    priority: int = 0

class AsyncParallelExecutor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.results: Dict[str, Any] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        
    async def submit_task(self, task: LearningTask) -> None:
        """Submit a task to the priority queue."""
        await self.task_queue.put((task.priority, task))
        
    async def process_task(self, task: LearningTask) -> Any:
        """Process a single learning task."""
        if task.strategy_name not in self.locks:
            self.locks[task.strategy_name] = asyncio.Lock()
            
        async with self.locks[task.strategy_name]:
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    self.process_pool,
                    self._execute_strategy,
                    task
                )
                self.results[task.strategy_name] = result
                return result
            except Exception as e:
                logging.error(
                    f"Error processing task {task.strategy_name}: {str(e)}",
                    exc_info=True
                )
                return None
                
    def _execute_strategy(self, task: LearningTask) -> Any:
        """Execute learning strategy in separate process."""
        try:
            # Simulate strategy execution
            time.sleep(np.random.random())  # Replace with actual strategy
            return {
                'strategy': task.strategy_name,
                'status': 'completed',
                'parameters': task.parameters
            }
        except Exception as e:
            logging.error(f"Strategy execution error: {str(e)}", exc_info=True)
            return None
            
    async def run_parallel_tasks(self, tasks: List[LearningTask]) -> Dict[str, Any]:
        """Run multiple learning tasks in parallel."""
        for task in tasks:
            await self.submit_task(task)
            
        processors = []
        while not self.task_queue.empty():
            _, task = await self.task_queue.get()
            processors.append(self.process_task(task))
            
        await asyncio.gather(*processors)
        return self.results
        
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.process_pool.shutdown()
        self.results.clear()
        self.locks.clear()

# Usage example:
async def main():
    executor = AsyncParallelExecutor()
    
    tasks = [
        LearningTask("strategy1", data=None, parameters={'param1': 1}, priority=1),
        LearningTask("strategy2", data=None, parameters={'param2': 2}, priority=2)
    ]
    
    results = await executor.run_parallel_tasks(tasks)
    await executor.cleanup()
    
    return results

# Run the executor
if __name__ == "__main__":
    results = asyncio.run(main())

# End of parallel_utils.py
