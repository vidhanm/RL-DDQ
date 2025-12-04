"""
NLU-based state extraction from debtor text responses.

Key Design Principles:
1. DETERMINISTIC: Same text → same features (no LLM randomness)
2. PRODUCTION-READY: Same extraction works on real speech transcripts
3. INTERPRETABLE: Features map to meaningful behavioral signals

Components:
- VADER Sentiment: Fast, accurate sentiment analysis
- Intent Classifier: Rule-based intent detection
- Signal Extractor: Keyword/regex for specific signals
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class NLUFeatures:
    """Extracted features from debtor response text"""
    
    # Core sentiment (-1.0 to 1.0)
    sentiment: float
    
    # Intent classification
    intent: str  # refusing, explaining, willing, committing, hostile, avoidant, questioning
    
    # Cooperation estimate (0.0 to 1.0)
    cooperation: float
    
    # Boolean signals
    shared_situation: bool      # Mentioned hardship/circumstances
    feels_understood: bool      # Acknowledged agent's empathy
    commitment_signal: bool     # Expressed intent to pay
    quit_signal: bool          # Wants to end conversation
    payment_mentioned: bool    # Discussed specific payment
    
    # Additional features
    response_length: int       # Word count (engagement proxy)
    question_count: int        # Questions asked (engagement proxy)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for state encoding"""
        return {
            'sentiment': self.sentiment,
            'intent': self.intent,
            'cooperation': self.cooperation,
            'shared_situation': self.shared_situation,
            'feels_understood': self.feels_understood,
            'commitment_signal': self.commitment_signal,
            'quit_signal': self.quit_signal,
            'payment_mentioned': self.payment_mentioned,
            'response_length': self.response_length,
            'question_count': self.question_count
        }


class DebtorResponseAnalyzer:
    """
    Extract behavioral features from debtor text responses.
    
    All methods are DETERMINISTIC - same input always produces same output.
    This is critical for stable RL training signals.
    """
    
    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Intent patterns (order matters - first match wins for some)
        self.intent_patterns = {
            'committing': [
                r'\bi\'ll pay\b', r'\bi will pay\b', r'\bset up.*payment\b',
                r'\bsign me up\b', r'\bagree\b', r'\blet\'s do it\b',
                r'\bi promise\b', r'\bi\'ll.*tomorrow\b', r'\bschedule.*payment\b',
                r'\bcount me in\b', r'\bi commit\b', r'\bauto.?pay\b'
            ],
            'willing': [
                r'\bmaybe\b', r'\bperhaps\b', r'\bwhat if\b', r'\bi could\b',
                r'\blet me think\b', r'\bmight be able\b', r'\bi\'ll try\b',
                r'\bpossibly\b', r'\bi guess\b', r'\bi suppose\b',
                r'\btell me more\b', r'\bwhat.*options\b', r'\bhow much\b'
            ],
            'explaining': [
                r'\blost.*job\b', r'\bunemployed\b', r'\bmedical\b', r'\bhospital\b',
                r'\bdivorce\b', r'\bseparated\b', r'\bsick\b', r'\binjury\b',
                r'\blaid off\b', r'\bfired\b', r'\breduced hours\b',
                r'\bcan\'t afford\b', r'\bno income\b', r'\bstruggling\b',
                r'\bdifficult time\b', r'\bhard time\b', r'\bbehind on\b'
            ],
            'hostile': [
                r'\bstop calling\b', r'\bleave me alone\b', r'\bdon\'t call\b',
                r'\bscam\b', r'\bsue\b', r'\blawyer\b', r'\bharassment\b',
                r'\breport you\b', r'\bgo to hell\b', r'\bf[*u]ck\b',
                r'\bshut up\b', r'\bidiot\b', r'\bstupid\b'
            ],
            'refusing': [
                r'\bcan\'t pay\b', r'\bwon\'t pay\b', r'\bnot paying\b',
                r'\bno way\b', r'\bforget it\b', r'\bnot interested\b',
                r'\bno money\b', r'\bdon\'t have\b', r'\bimpossible\b',
                r'\bnot gonna happen\b', r'\babsolutely not\b'
            ],
            'avoidant': [
                r'\bcall.*later\b', r'\bcall.*back\b', r'\bbusy right now\b',
                r'\bnot a good time\b', r'\bcan\'t talk\b', r'\bgotta go\b',
                r'\blet me.*think\b', r'\bneed.*time\b', r'\bmaybe later\b',
                r'\bwe\'ll see\b', r'\bi don\'t know\b'
            ],
            'questioning': [
                r'\bwho.*calling\b', r'\bwhat.*about\b', r'\bwhy.*calling\b',
                r'\bhow much\b', r'\bwhat.*options\b', r'\bcan you explain\b',
                r'\bwhat happens if\b', r'\bwhat if i\b', r'\?'
            ]
        }
        
        # Signals patterns
        self.signal_patterns = {
            'shared_situation': [
                r'\blost.*job\b', r'\bunemployed\b', r'\bmedical\b',
                r'\bdivorce\b', r'\bsick\b', r'\binjury\b', r'\bstruggling\b',
                r'\bbehind on bills\b', r'\bfinancial\b', r'\bhard time\b',
                r'\bdifficult\b', r'\bcan\'t afford\b', r'\bno income\b'
            ],
            'feels_understood': [
                r'\bthank you\b', r'\bappreciate\b', r'\bthat helps\b',
                r'\bi understand\b', r'\byou\'re right\b', r'\bgood point\b',
                r'\bthat makes sense\b', r'\bi see\b', r'\bfair enough\b'
            ],
            'commitment_signal': [
                r'\bi\'ll pay\b', r'\bi will pay\b', r'\bi promise\b',
                r'\bset.*up\b', r'\bschedule\b', r'\bagree\b', r'\byes\b',
                r'\bok\b', r'\bsounds good\b', r'\blet\'s do\b', r'\bdeal\b'
            ],
            'quit_signal': [
                r'\bstop calling\b', r'\bdon\'t call\b', r'\bhanging up\b',
                r'\bleave me alone\b', r'\bremove.*number\b', r'\bdo not contact\b',
                r'\bi\'m done\b', r'\bgoodbye\b', r'\bend this\b'
            ],
            'payment_mentioned': [
                r'\$\d+', r'\bpay\b', r'\bpayment\b', r'\binstallment\b',
                r'\bmonthly\b', r'\bweekly\b', r'\bdown payment\b',
                r'\bsettle\b', r'\bamount\b', r'\bowe\b'
            ]
        }
        
        # Cooperation keywords (positive/negative weights)
        self.cooperation_positive = [
            'yes', 'okay', 'ok', 'sure', 'agree', 'understand', 'willing',
            'try', 'help', 'work', 'together', 'plan', 'option', 'possible',
            'appreciate', 'thank', 'fair', 'reasonable'
        ]
        self.cooperation_negative = [
            'no', 'never', 'refuse', 'won\'t', 'can\'t', 'impossible',
            'forget', 'stop', 'leave', 'harassment', 'lawyer', 'sue',
            'scam', 'fraud', 'illegal', 'rights'
        ]
    
    def analyze(self, text: str) -> NLUFeatures:
        """
        Extract all features from debtor response text.
        
        Args:
            text: Debtor's response text
            
        Returns:
            NLUFeatures with all extracted signals
        """
        if not text or not text.strip():
            return self._empty_features()
        
        text_lower = text.lower()
        
        # Extract all features
        sentiment = self._extract_sentiment(text)
        intent = self._classify_intent(text_lower)
        signals = self._extract_signals(text_lower)
        cooperation = self._estimate_cooperation(text_lower, sentiment, intent)
        response_length = len(text.split())
        question_count = text.count('?')
        
        return NLUFeatures(
            sentiment=sentiment,
            intent=intent,
            cooperation=cooperation,
            shared_situation=signals['shared_situation'],
            feels_understood=signals['feels_understood'],
            commitment_signal=signals['commitment_signal'],
            quit_signal=signals['quit_signal'],
            payment_mentioned=signals['payment_mentioned'],
            response_length=response_length,
            question_count=question_count
        )
    
    def _extract_sentiment(self, text: str) -> float:
        """
        Extract sentiment using VADER.
        
        Returns:
            Compound sentiment score from -1.0 (negative) to 1.0 (positive)
        """
        scores = self.vader.polarity_scores(text)
        return scores['compound']
    
    def _classify_intent(self, text_lower: str) -> str:
        """
        Classify debtor intent using keyword patterns.
        
        Priority order (first match in priority order):
        1. committing (strongest positive signal)
        2. hostile (strongest negative signal)
        3. quit_signal (via refusing with quit patterns)
        4. willing
        5. explaining
        6. refusing
        7. avoidant
        8. questioning
        9. neutral (default)
        """
        # Check patterns in priority order
        priority_order = [
            'committing', 'hostile', 'willing', 'explaining', 
            'refusing', 'avoidant', 'questioning'
        ]
        
        for intent in priority_order:
            patterns = self.intent_patterns.get(intent, [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return 'neutral'
    
    def _extract_signals(self, text_lower: str) -> Dict[str, bool]:
        """
        Extract boolean signals from text.
        
        Returns:
            Dictionary of signal_name: bool
        """
        signals = {}
        for signal_name, patterns in self.signal_patterns.items():
            signals[signal_name] = any(
                re.search(pattern, text_lower) for pattern in patterns
            )
        return signals
    
    def _estimate_cooperation(
        self, 
        text_lower: str, 
        sentiment: float, 
        intent: str
    ) -> float:
        """
        Estimate cooperation level based on multiple signals.
        
        Returns:
            Cooperation score from 0.0 (hostile/refusing) to 1.0 (fully cooperative)
        """
        # Base score from intent
        intent_scores = {
            'committing': 0.95,
            'willing': 0.7,
            'questioning': 0.6,
            'explaining': 0.5,
            'neutral': 0.5,
            'avoidant': 0.3,
            'refusing': 0.2,
            'hostile': 0.05
        }
        base_score = intent_scores.get(intent, 0.5)
        
        # Adjust based on sentiment
        sentiment_adjustment = sentiment * 0.15  # Max ±0.15
        
        # Adjust based on cooperation keywords
        words = text_lower.split()
        positive_count = sum(1 for w in words if w in self.cooperation_positive)
        negative_count = sum(1 for w in words if w in self.cooperation_negative)
        keyword_adjustment = (positive_count - negative_count) * 0.05
        
        # Combine and clamp
        final_score = base_score + sentiment_adjustment + keyword_adjustment
        return max(0.0, min(1.0, final_score))
    
    def _empty_features(self) -> NLUFeatures:
        """Return neutral features for empty/missing text"""
        return NLUFeatures(
            sentiment=0.0,
            intent='neutral',
            cooperation=0.5,
            shared_situation=False,
            feels_understood=False,
            commitment_signal=False,
            quit_signal=False,
            payment_mentioned=False,
            response_length=0,
            question_count=0
        )
    
    def get_cooperation_delta(
        self, 
        prev_cooperation: float, 
        nlu_features: NLUFeatures
    ) -> float:
        """
        Calculate cooperation change based on NLU analysis.
        
        Args:
            prev_cooperation: Previous cooperation value
            nlu_features: Current NLU features
            
        Returns:
            New cooperation value (smoothed transition)
        """
        # Smooth transition: 70% NLU estimate, 30% previous value
        target = nlu_features.cooperation
        smoothed = 0.7 * target + 0.3 * prev_cooperation
        return smoothed


# Quick test
if __name__ == "__main__":
    analyzer = DebtorResponseAnalyzer()
    
    test_responses = [
        "I don't have any money right now. I lost my job last month.",
        "Stop calling me! This is harassment!",
        "Okay, I can do $50 a month. Let's set that up.",
        "I need to think about it. Can you call back next week?",
        "What are my options? How much do I owe exactly?",
        "Fine, I'll pay. Just tell me where to send the money.",
        "I appreciate you being understanding. It's been a hard time.",
    ]
    
    print("=" * 70)
    print("NLU State Extractor Test")
    print("=" * 70)
    
    for response in test_responses:
        features = analyzer.analyze(response)
        print(f"\nText: \"{response[:60]}...\"" if len(response) > 60 else f"\nText: \"{response}\"")
        print(f"  Sentiment: {features.sentiment:.2f}")
        print(f"  Intent: {features.intent}")
        print(f"  Cooperation: {features.cooperation:.2f}")
        print(f"  Signals: situation={features.shared_situation}, understood={features.feels_understood}, commit={features.commitment_signal}, quit={features.quit_signal}")
