"""
Debtor Type Classifier

Classifies debtor type based on conversation history (after 2-3 turns).
This enables the agent to adapt strategy based on inferred debtor personality.

Types:
- HOSTILE: Angry, threatening, uncooperative
- COOPERATIVE: Willing to work together, positive
- EVASIVE: Deflecting, delaying, avoiding commitment
- EMOTIONAL: Distressed, sharing hardships, needs empathy
- NEGOTIATOR: Asking questions, discussing amounts, bargaining
- UNKNOWN: Not enough data yet (< 2 turns)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TurnData:
    """Data from a single turn for classification"""
    intent: str
    sentiment: float
    cooperation: float
    shared_situation: bool = False
    payment_mentioned: bool = False
    response_length: int = 0
    question_count: int = 0


class DebtorTypeClassifier:
    """
    Classify debtor type from conversation history.
    
    Works after 2+ turns of conversation data.
    Uses rule-based classification with confidence scoring.
    """
    
    # Debtor types
    HOSTILE = "HOSTILE"
    COOPERATIVE = "COOPERATIVE"
    EVASIVE = "EVASIVE"
    EMOTIONAL = "EMOTIONAL"
    NEGOTIATOR = "NEGOTIATOR"
    UNKNOWN = "UNKNOWN"
    
    ALL_TYPES = [HOSTILE, COOPERATIVE, EVASIVE, EMOTIONAL, NEGOTIATOR]
    
    def __init__(self, min_turns: int = 2):
        """
        Initialize classifier.
        
        Args:
            min_turns: Minimum turns before classification (default 2)
        """
        self.min_turns = min_turns
    
    def classify(self, turn_history: List[TurnData]) -> Tuple[str, float]:
        """
        Classify debtor type from turn history.
        
        Args:
            turn_history: List of TurnData from conversation
            
        Returns:
            Tuple of (debtor_type, confidence)
        """
        if len(turn_history) < self.min_turns:
            return self.UNKNOWN, 0.0
        
        # Calculate aggregate features
        features = self._aggregate_features(turn_history)
        
        # Score each type
        scores = {
            self.HOSTILE: self._score_hostile(features),
            self.COOPERATIVE: self._score_cooperative(features),
            self.EVASIVE: self._score_evasive(features),
            self.EMOTIONAL: self._score_emotional(features),
            self.NEGOTIATOR: self._score_negotiator(features)
        }
        
        # Find best type
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Convert score to confidence (0-1)
        confidence = min(1.0, best_score / 3.0)  # Normalize
        
        # Require minimum confidence
        if confidence < 0.3:
            return self.UNKNOWN, confidence
        
        return best_type, confidence
    
    def _aggregate_features(self, turn_history: List[TurnData]) -> Dict:
        """Calculate aggregate features from turn history"""
        n = len(turn_history)
        
        # Intent counts
        intent_counts = {}
        for turn in turn_history:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
        
        # Averages
        avg_sentiment = sum(t.sentiment for t in turn_history) / n
        avg_cooperation = sum(t.cooperation for t in turn_history) / n
        avg_response_length = sum(t.response_length for t in turn_history) / n
        
        # Flags
        any_shared_situation = any(t.shared_situation for t in turn_history)
        any_payment_mentioned = any(t.payment_mentioned for t in turn_history)
        total_questions = sum(t.question_count for t in turn_history)
        
        return {
            'intent_counts': intent_counts,
            'avg_sentiment': avg_sentiment,
            'avg_cooperation': avg_cooperation,
            'avg_response_length': avg_response_length,
            'any_shared_situation': any_shared_situation,
            'any_payment_mentioned': any_payment_mentioned,
            'total_questions': total_questions,
            'n_turns': n
        }
    
    def _score_hostile(self, f: Dict) -> float:
        """Score for HOSTILE type"""
        score = 0.0
        
        # Hostile or refusing intents
        hostile_count = f['intent_counts'].get('hostile', 0)
        refusing_count = f['intent_counts'].get('refusing', 0)
        score += hostile_count * 2.0
        score += refusing_count * 1.0
        
        # Very negative sentiment
        if f['avg_sentiment'] < -0.4:
            score += 2.0
        elif f['avg_sentiment'] < -0.2:
            score += 1.0
        
        # Low cooperation
        if f['avg_cooperation'] < 0.2:
            score += 1.5
        
        return score
    
    def _score_cooperative(self, f: Dict) -> float:
        """Score for COOPERATIVE type"""
        score = 0.0
        
        # Willing or committing intents
        willing_count = f['intent_counts'].get('willing', 0)
        committing_count = f['intent_counts'].get('committing', 0)
        score += willing_count * 1.5
        score += committing_count * 3.0
        
        # Positive sentiment
        if f['avg_sentiment'] > 0.2:
            score += 1.5
        elif f['avg_sentiment'] > 0:
            score += 0.5
        
        # High cooperation
        if f['avg_cooperation'] > 0.6:
            score += 2.0
        elif f['avg_cooperation'] > 0.4:
            score += 1.0
        
        return score
    
    def _score_evasive(self, f: Dict) -> float:
        """Score for EVASIVE type"""
        score = 0.0
        
        # Avoidant intent
        avoidant_count = f['intent_counts'].get('avoidant', 0)
        score += avoidant_count * 2.0
        
        # Short responses (low engagement)
        if f['avg_response_length'] < 10:
            score += 1.5
        elif f['avg_response_length'] < 20:
            score += 0.5
        
        # Moderate sentiment (not strongly positive or negative)
        if -0.2 < f['avg_sentiment'] < 0.2:
            score += 0.5
        
        # Medium cooperation (not clearly refusing, not committing)
        if 0.3 < f['avg_cooperation'] < 0.5:
            score += 1.0
        
        return score
    
    def _score_emotional(self, f: Dict) -> float:
        """Score for EMOTIONAL type"""
        score = 0.0
        
        # Shared situation (hardship)
        if f['any_shared_situation']:
            score += 2.5
        
        # Explaining intent
        explaining_count = f['intent_counts'].get('explaining', 0)
        score += explaining_count * 1.5
        
        # Negative sentiment (distress)
        if f['avg_sentiment'] < -0.1:
            score += 1.0
        
        # Longer responses (sharing details)
        if f['avg_response_length'] > 30:
            score += 1.0
        
        return score
    
    def _score_negotiator(self, f: Dict) -> float:
        """Score for NEGOTIATOR type"""
        score = 0.0
        
        # Payment mentioned (discussing amounts)
        if f['any_payment_mentioned']:
            score += 2.0
        
        # Questioning intent
        questioning_count = f['intent_counts'].get('questioning', 0)
        score += questioning_count * 1.5
        
        # Asks questions
        if f['total_questions'] >= 2:
            score += 1.5
        elif f['total_questions'] >= 1:
            score += 0.5
        
        # Moderate to positive cooperation
        if f['avg_cooperation'] > 0.4:
            score += 1.0
        
        return score
    
    def classify_from_nlu_features(self, nlu_features_list: List) -> Tuple[str, float]:
        """
        Classify from list of NLUFeatures objects.
        
        Convenience method for integration with state_extractor.
        """
        turn_history = []
        for nlu in nlu_features_list:
            turn_data = TurnData(
                intent=nlu.intent,
                sentiment=nlu.sentiment,
                cooperation=nlu.cooperation,
                shared_situation=nlu.shared_situation,
                payment_mentioned=nlu.payment_mentioned,
                response_length=nlu.response_length,
                question_count=nlu.question_count
            )
            turn_history.append(turn_data)
        
        return self.classify(turn_history)


# Quick test
if __name__ == "__main__":
    classifier = DebtorTypeClassifier()
    
    print("=" * 50)
    print("Debtor Type Classifier Test")
    print("=" * 50)
    
    # Test 1: Hostile debtor
    hostile_history = [
        TurnData(intent='hostile', sentiment=-0.7, cooperation=0.1),
        TurnData(intent='refusing', sentiment=-0.5, cooperation=0.2)
    ]
    dtype, conf = classifier.classify(hostile_history)
    print(f"\nHostile debtor: {dtype} (confidence: {conf:.2f})")
    
    # Test 2: Cooperative debtor
    coop_history = [
        TurnData(intent='willing', sentiment=0.3, cooperation=0.7),
        TurnData(intent='committing', sentiment=0.5, cooperation=0.9)
    ]
    dtype, conf = classifier.classify(coop_history)
    print(f"Cooperative debtor: {dtype} (confidence: {conf:.2f})")
    
    # Test 3: Evasive debtor
    evasive_history = [
        TurnData(intent='avoidant', sentiment=0.0, cooperation=0.4, response_length=8),
        TurnData(intent='avoidant', sentiment=-0.1, cooperation=0.35, response_length=5)
    ]
    dtype, conf = classifier.classify(evasive_history)
    print(f"Evasive debtor: {dtype} (confidence: {conf:.2f})")
    
    # Test 4: Emotional debtor
    emotional_history = [
        TurnData(intent='explaining', sentiment=-0.3, cooperation=0.5, 
                 shared_situation=True, response_length=45),
        TurnData(intent='explaining', sentiment=-0.2, cooperation=0.55, 
                 shared_situation=True, response_length=40)
    ]
    dtype, conf = classifier.classify(emotional_history)
    print(f"Emotional debtor: {dtype} (confidence: {conf:.2f})")
    
    # Test 5: Negotiator debtor
    negotiator_history = [
        TurnData(intent='questioning', sentiment=0.1, cooperation=0.5, 
                 payment_mentioned=True, question_count=2),
        TurnData(intent='willing', sentiment=0.2, cooperation=0.6, 
                 payment_mentioned=True, question_count=1)
    ]
    dtype, conf = classifier.classify(negotiator_history)
    print(f"Negotiator debtor: {dtype} (confidence: {conf:.2f})")
    
    print("\n" + "=" * 50)
