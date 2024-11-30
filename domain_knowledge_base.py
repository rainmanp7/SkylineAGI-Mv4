from knowledge_base import TieredKnowledgeBase  # Direct import from existing file
import json
import os

class DomainKnowledgeBase:
    def __init__(self, 
                 dataset_path='domain_dataset.json', 
                 knowledge_base=None):
        # If no knowledge_base provided, create a new TieredKnowledgeBase
        self.tiered_knowledge_base = knowledge_base or TieredKnowledgeBase()
        
        # Rest of the implementation remains the same as in previous response
        ...