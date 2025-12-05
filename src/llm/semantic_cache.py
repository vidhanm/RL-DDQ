"""
Semantic Cache for LLM Responses
Caches responses by semantic similarity to reduce API costs by 50-70%
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import hashlib
import json
import os


class SemanticCache:
    """
    Cache LLM responses using semantic similarity.
    
    Uses sentence-transformers for embeddings and cosine similarity
    to find cached responses for similar prompts.
    """
    
    def __init__(
        self,
        threshold: float = 0.85,
        max_size: int = 10000,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize semantic cache.
        
        Args:
            threshold: Minimum similarity to return cached response (0.0-1.0)
            max_size: Maximum number of entries to store
            model_name: Sentence-transformer model name
            cache_dir: Directory to persist cache (optional)
        """
        self.threshold = threshold
        self.max_size = max_size
        self.cache_dir = cache_dir
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.responses: List[str] = []
        self.prompts: List[str] = []  # For debugging
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.stores = 0
        
        # Lazy load embedding model (saves memory if cache disabled)
        self._model = None
        self._model_name = model_name
        
        # Load from disk if exists
        if cache_dir:
            self._load_from_disk()
    
    @property
    def model(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Cache] Loading embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                print(f"[Cache] Model loaded successfully")
            except ImportError:
                print("[Cache] WARNING: sentence-transformers not installed!")
                print("[Cache] Run: pip install sentence-transformers")
                print("[Cache] Caching disabled.")
                return None
        return self._model
    
    def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response for similar prompt.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Cached response if similar prompt exists, None otherwise
        """
        if not self.embeddings or self.model is None:
            self.misses += 1
            return None
        
        # Embed the query prompt
        query_embedding = self.model.encode(prompt, convert_to_numpy=True)
        
        # Find most similar
        best_idx, best_score = self._find_most_similar(query_embedding)
        
        if best_score >= self.threshold:
            self.hits += 1
            return self.responses[best_idx]
        else:
            self.misses += 1
            return None
    
    def store(self, prompt: str, response: str) -> None:
        """
        Store prompt/response pair in cache.
        
        Args:
            prompt: The prompt
            response: The LLM response
        """
        if self.model is None:
            return
        
        # Embed the prompt
        embedding = self.model.encode(prompt, convert_to_numpy=True)
        
        # Check if we're at capacity
        if len(self.embeddings) >= self.max_size:
            # Remove oldest entry (FIFO)
            self.embeddings.pop(0)
            self.responses.pop(0)
            self.prompts.pop(0)
        
        # Store
        self.embeddings.append(embedding)
        self.responses.append(response)
        self.prompts.append(prompt[:100])  # Truncate for memory
        self.stores += 1
        
        # Persist periodically
        if self.cache_dir and self.stores % 100 == 0:
            self._save_to_disk()
    
    def _find_most_similar(self, query_embedding: np.ndarray) -> Tuple[int, float]:
        """Find the most similar cached embedding"""
        if not self.embeddings:
            return -1, 0.0
        
        # Stack all embeddings
        cache_embeddings = np.stack(self.embeddings)
        
        # Compute cosine similarity
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cache_norms = cache_embeddings / (np.linalg.norm(cache_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Dot product = cosine similarity for normalized vectors
        similarities = np.dot(cache_norms, query_norm)
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return int(best_idx), float(best_score)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.embeddings),
            "max_size": self.max_size,
            "stores": self.stores,
            "threshold": self.threshold
        }
    
    def print_stats(self) -> None:
        """Print cache statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("SEMANTIC CACHE STATISTICS")
        print("="*50)
        print(f"Cache Size: {stats['cache_size']}/{stats['max_size']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Similarity Threshold: {stats['threshold']}")
        if stats['hit_rate'] > 0:
            savings = stats['hits'] / (stats['hits'] + stats['misses'])
            print(f"Estimated API Cost Savings: {savings:.1%}")
        print("="*50 + "\n")
    
    def _save_to_disk(self) -> None:
        """Save cache to disk"""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Save embeddings as numpy
        if self.embeddings:
            np.save(
                os.path.join(self.cache_dir, "embeddings.npy"),
                np.stack(self.embeddings)
            )
        
        # Save responses and prompts as JSON
        with open(os.path.join(self.cache_dir, "cache_data.json"), "w") as f:
            json.dump({
                "responses": self.responses,
                "prompts": self.prompts,
                "stats": {"hits": self.hits, "misses": self.misses, "stores": self.stores}
            }, f)
    
    def _load_from_disk(self) -> None:
        """Load cache from disk"""
        if not self.cache_dir:
            return
        
        embeddings_path = os.path.join(self.cache_dir, "embeddings.npy")
        data_path = os.path.join(self.cache_dir, "cache_data.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(data_path):
            try:
                embeddings = np.load(embeddings_path)
                self.embeddings = [embeddings[i] for i in range(len(embeddings))]
                
                with open(data_path, "r") as f:
                    data = json.load(f)
                    self.responses = data["responses"]
                    self.prompts = data.get("prompts", [""] * len(self.responses))
                    stats = data.get("stats", {})
                    self.hits = stats.get("hits", 0)
                    self.misses = stats.get("misses", 0)
                    self.stores = stats.get("stores", 0)
                
                print(f"[Cache] Loaded {len(self.embeddings)} entries from disk")
            except Exception as e:
                print(f"[Cache] Failed to load from disk: {e}")
    
    def clear(self) -> None:
        """Clear the cache"""
        self.embeddings = []
        self.responses = []
        self.prompts = []
        self.hits = 0
        self.misses = 0
        self.stores = 0
        print("[Cache] Cache cleared")
