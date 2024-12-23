
# memory_manager.py
# Created on Nov13 2024
####
# Use the memory management functions as needed # throughout the codebase. For example:
##
# Modified on Nov25 2024

```python

from typing import Any, Dict

class MemoryManager:
    def __init__(self):
        """Initialize memory stores for working, short-term, and long-term memories."""
        self.working_memory: Dict[str, Any] = {}
        self.short_term_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}

    def store_working_memory(self, key: str, value: Any) -> str:
        """Store a value in working memory."""
        self.working_memory[key] = value
        return "Working Memory Initialized"

    def store_short_term_memory(self, key: str, value: Any) -> str:
        """Store a value in short-term memory."""
        self.short_term_memory[key] = value
        return "Short-Term Memory Initialized"

    def store_long_term_memory(self, key: str, value: Any) -> str:
        """Store a value in long-term memory."""
        self.long_term_memory[key] = value
        return "Long-Term Memory Initialized"

    def memory_consolidation(self) -> str:
        """Consolidate memories from working and short-term to long-term."""
        # Store all values from working and short-term memories to long-term memory
        for key, value in self.working_memory.items():
            self.long_term_memory[key] = value

        for key, value in self.short_term_memory.items():
            self.long_term_memory[key] = value

        # Clear the working and short-term memories after consolidation
        self.working_memory.clear()
        self.short_term_memory.clear()

        return "Memory Consolidation Activated"

    def memory_retrieval(self, key: str, memory_type: str) -> Any:
        """Retrieve a value from specified memory type."""
        if memory_type == "working":
            return self.working_memory.get(key, None)
        elif memory_type == "short_term":
            return self.short_term_memory.get(key, None)
        elif memory_type == "long_term":
            return self.long_term_memory.get(key, None)
        else:
            return None

    def get_all_memories(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary containing all types of memories."""
        return {
            'working_memory': dict(self.working_memory),
            'short_term_memory': dict(self.short_term_memory),
            'long_term_memory': dict(self.long_term_memory)
        }


```
#end of memory management.
