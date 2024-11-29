# Beginning of knowledge_base.py
from collections import deque
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional

from internal_process_monitor import InternalProcessMonitor

class TieredKnowledgeBase:
    # Define complexity tiers
    TIERS = {
        'easy': (1, 3),
        'simp': (4, 7),
        'norm': (8, 11),
        'mods': (12, 15),
        'hard': (16, 19),
        'para': (20, 23),
        'vice': (24, 27),
        'zeta': (28, 31),
        'tetris': (32, 35)
    }

    def __init__(self, max_recent_items: int = 100):
        self.knowledge_bases = {tier: {} for tier in self.TIERS.keys()}
        self.recent_updates = deque(maxlen=max_recent_items)
        self.lock = Lock()
        self.knowledge_base_monitor = InternalProcessMonitor()

    def _get_tier(self, complexity: int) -> Optional[str]:
        """Determine which tier a piece of information belongs to based on its complexity."""
        for tier, (min_comp, max_comp) in self.TIERS.items():
            if min_comp <= complexity <= max_comp:
                return tier
        return None

    def create(self, key: str, value: Any, complexity: int) -> bool:
        """Add a new entry to the knowledge base."""
        tier = self._get_tier(complexity)
        if tier is None:
            return False
        with self.lock:
            if key not in self.knowledge_bases[tier]:
                self.knowledge_bases[tier][key] = value
                self.recent_updates.append((tier, key, value, complexity))
                self.knowledge_base_monitor.on_knowledge_update(tier, key, value, complexity)
                return True
            return False

    def read(self, key: str, complexity_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Retrieve an entry from the knowledge base."""
        results = {}
        with self.lock:
            if complexity_range:
                min_comp, max_comp = complexity_range
                relevant_tiers = [tier for tier in self.TIERS.keys() if not (self.TIERS[tier][1] < min_comp or self.TIERS[tier][0] > max_comp)]
            else:
                relevant_tiers = self.TIERS.keys()
                
            for tier in relevant_tiers:
                if key in self.knowledge_bases[tier]:
                    results[tier] = self.knowledge_bases[tier][key]
            return results

    def update(self, key: str, value: Any, complexity: int) -> bool:
        """Update an existing entry in the knowledge base."""
        tier = self._get_tier(complexity)
        if tier is None:
            return False
        with self.lock:
            if key in self.knowledge_bases[tier]:
                self.knowledge_bases[tier][key] = value
                self.recent_updates.append((tier, key, value, complexity))
                self.knowledge_base_monitor.on_knowledge_update(tier, key, value, complexity)
                return True
            return False

    def delete(self, key: str) -> bool:
        """Delete an entry from the knowledge base."""
        with self.lock:
            for tier in self.TIERS.keys():
                if key in self.knowledge_bases[tier]:
                    del self.knowledge_bases[tier][key]
                    return True
            return False

    def get_recent_updates(self, n: int = None) -> List[Tuple[str, str, Any]]:
        """Get recent updates across all tiers."""
        with self.lock:
            updates = list(self.recent_updates)
            if n is not None:
                updates = updates[-n:]
            return updates

    def get_tier_stats(self) -> Dict[str, int]:
        """Get statistics about how many items are stored in each tier."""
        with self.lock:
            return {tier: len(kb) for tier, kb in self.knowledge_bases.items()}

# End of knowledge_base.py