"""
Success Memory for In-Context Adaptation

Stores successful (debtor_type, action, outcome) pairs and provides
context for LLM prompts about what worked in similar situations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os


@dataclass
class SuccessRecord:
    """Record of a successful strategy"""
    debtor_type: str          # HOSTILE, COOPERATIVE, etc.
    action: str               # Action that worked
    sentiment_before: float   # Sentiment before action
    sentiment_after: float    # Sentiment after action
    cooperation_gain: float   # Cooperation improvement
    context: str = ""         # Brief context of situation


class SuccessMemory:
    """
    Memory store for successful strategies.
    
    Tracks what works for different debtor types and provides
    in-context examples for LLM prompts.
    """
    
    def __init__(self, max_per_type: int = 10, save_path: str = None):
        """
        Initialize success memory.
        
        Args:
            max_per_type: Maximum records to keep per debtor type
            save_path: Optional path to persist memory
        """
        self.max_per_type = max_per_type
        self.save_path = save_path
        
        # Store by debtor type
        self.memories: Dict[str, List[SuccessRecord]] = defaultdict(list)
        
        # Load if exists
        if save_path and os.path.exists(save_path):
            self.load(save_path)
    
    def record_success(
        self,
        debtor_type: str,
        action: str,
        sentiment_before: float,
        sentiment_after: float,
        cooperation_gain: float,
        context: str = ""
    ):
        """
        Record a successful strategy.
        
        Args:
            debtor_type: Type of debtor (HOSTILE, COOPERATIVE, etc.)
            action: Action that worked
            sentiment_before/after: Sentiment change
            cooperation_gain: How much cooperation improved
            context: Optional context description
        """
        record = SuccessRecord(
            debtor_type=debtor_type,
            action=action,
            sentiment_before=sentiment_before,
            sentiment_after=sentiment_after,
            cooperation_gain=cooperation_gain,
            context=context
        )
        
        self.memories[debtor_type].append(record)
        
        # Keep only top successes (by cooperation gain)
        if len(self.memories[debtor_type]) > self.max_per_type:
            self.memories[debtor_type].sort(
                key=lambda x: x.cooperation_gain, 
                reverse=True
            )
            self.memories[debtor_type] = self.memories[debtor_type][:self.max_per_type]
        
        # Auto-save
        if self.save_path:
            self.save(self.save_path)
    
    def get_context_for_prompt(self, debtor_type: str, top_k: int = 3) -> str:
        """
        Get context string for LLM prompt.
        
        Args:
            debtor_type: Type of debtor we're dealing with
            top_k: Number of examples to include
            
        Returns:
            Context string to include in prompt
        """
        records = self.memories.get(debtor_type, [])
        
        if not records:
            return ""
        
        # Get top k by cooperation gain
        top_records = sorted(records, key=lambda x: x.cooperation_gain, reverse=True)[:top_k]
        
        lines = [f"In similar situations with {debtor_type} debtors, these strategies worked well:"]
        
        for i, r in enumerate(top_records, 1):
            sentiment_change = r.sentiment_after - r.sentiment_before
            lines.append(
                f"{i}. '{r.action}' improved cooperation by {r.cooperation_gain:.1%}"
                f" (sentiment: {sentiment_change:+.2f})"
            )
        
        return "\n".join(lines)
    
    def get_best_action(self, debtor_type: str) -> Optional[str]:
        """Get the most successful action for a debtor type"""
        records = self.memories.get(debtor_type, [])
        
        if not records:
            return None
        
        # Find action with best average cooperation gain
        action_gains: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            action_gains[r.action].append(r.cooperation_gain)
        
        best_action = max(
            action_gains.keys(),
            key=lambda a: sum(action_gains[a]) / len(action_gains[a])
        )
        
        return best_action
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        stats = {
            'total_records': sum(len(v) for v in self.memories.values()),
            'debtor_types': list(self.memories.keys()),
            'records_per_type': {k: len(v) for k, v in self.memories.items()}
        }
        
        # Best action per type
        stats['best_actions'] = {}
        for dtype in self.memories.keys():
            best = self.get_best_action(dtype)
            if best:
                stats['best_actions'][dtype] = best
        
        return stats
    
    def save(self, path: str):
        """Save memory to file"""
        data = {
            dtype: [
                {
                    'debtor_type': r.debtor_type,
                    'action': r.action,
                    'sentiment_before': r.sentiment_before,
                    'sentiment_after': r.sentiment_after,
                    'cooperation_gain': r.cooperation_gain,
                    'context': r.context
                }
                for r in records
            ]
            for dtype, records in self.memories.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load memory from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.memories = defaultdict(list)
        for dtype, records in data.items():
            for r in records:
                self.memories[dtype].append(SuccessRecord(**r))


# Test
if __name__ == "__main__":
    print("Testing SuccessMemory...")
    
    memory = SuccessMemory()
    
    # Record some successes
    memory.record_success(
        debtor_type="HOSTILE",
        action="empathetic_listening",
        sentiment_before=-0.7,
        sentiment_after=-0.2,
        cooperation_gain=0.3,
        context="Debtor was angry, de-escalated"
    )
    
    memory.record_success(
        debtor_type="HOSTILE",
        action="ask_about_situation",
        sentiment_before=-0.5,
        sentiment_after=0.0,
        cooperation_gain=0.4,
        context="Asked about their situation"
    )
    
    memory.record_success(
        debtor_type="COOPERATIVE",
        action="offer_payment_plan",
        sentiment_before=0.3,
        sentiment_after=0.7,
        cooperation_gain=0.5,
        context="Direct approach worked"
    )
    
    # Get context
    print("\n✓ Context for HOSTILE debtor:")
    print(memory.get_context_for_prompt("HOSTILE"))
    
    print(f"\n✓ Best action for HOSTILE: {memory.get_best_action('HOSTILE')}")
    print(f"✓ Best action for COOPERATIVE: {memory.get_best_action('COOPERATIVE')}")
    
    print(f"\n✓ Statistics: {memory.get_statistics()}")
    
    print("\n✅ SuccessMemory working!")
