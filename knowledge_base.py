# Beginning of knowledge_base.py
from collections import deque
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional


# these numbers are defined within the Compx
# field inside the dataset file.
# Compx is the category name where the 
# complexity factor needs to place the number.
# these are the complexity tiers.
# this number arrangement should also be in the 
# complexity factor governing mathmatica.
# The Complexity is given to the data and 
# placed in the area that it comes closer too
# So the 2nd complexity factor number in the dataset has a start and end range for each piece of information after it's placed ,the 2nd part of the brain gives it a number in the range for domain transfer learning matching what it's alike too. That nember like the first winds up or down to fit also. This gives it alignment. So the 1st complexity is for placement ,the 2nd complexity is for domain likeness assignment and domain learning and transfer learning.

# this number arrangement here is for complexity placement. Match the nearest and put it in the right category dataset. The information here is for the assimilation to follow and just placed here for import etc.. and how the knowledge base aka domain dataset.

# each domain dataset has this num arrangement.

from internal_process_monitor import InternalProcessMonitor

class TieredKnowledgeBase:
    # Define complexity tiers
    TIERS = {
    # 1st Section
    'easy': (0001, 0228),
    'simp': (0229, 0456),
    'norm': (0457, 0684),

    # 2nd Section
    'mods': (0685, 0912),
    'hard': (0913, 1140),
    'para': (1141, 1368),

    # 3rd Section
    'vice': (1369, 1596),
    'zeta': (1597, 1824),
    'tetr': (1825, 2052),

    # 4th Section
    'eafv': (2053, 2280),
    'sipo': (2281, 2508),
    'nxxm': (2509, 2736),

    # 5th Section
    'mids': (2737, 2964),
    'haod': (2965, 3192),
    'parz': (3193, 3420),

    # 6th Section
    'viff': (3421, 3648),
    'zexa': (3649, 3876),
    'sip8': (3877, 4104),

    # 7th Section
    'nxVm': (4105, 4332),
    'Vids': (4333, 4560),
    'ha3d': (4561, 4788),

    # 8th Section
    'pfgz': (4789, 5016),
    'vpff': (5017, 5244),
    'z9xa': (5245, 5472),

    # 9th Section
    'Tipo': (5473, 5700),
    'nxNm': (5701, 5928),
    'mPd7': (5929, 6156)
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
