"""
Session management for conversation state
UUID-based sessions with TTL cleanup
"""

import uuid
import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Session:
    """Represents a single user session"""
    session_id: str
    created_at: float
    last_accessed: float
    persona: str
    current_state: Optional[np.ndarray] = None
    conversation_history: list = field(default_factory=list)
    is_episode_done: bool = False
    env: Any = None  # DebtCollectionEnv instance
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = time.time()


class SessionManager:
    """
    Manages user sessions with automatic cleanup
    
    Each session holds:
    - Environment instance
    - Current state
    - Conversation history
    - Episode status
    """
    
    def __init__(self, ttl_seconds: int = 1800):  # 30 min default
        self.sessions: Dict[str, Session] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._running = False
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None:
            self._running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
            self._cleanup_thread = None
    
    def _cleanup_loop(self):
        """Background loop to cleanup expired sessions"""
        while self._running:
            time.sleep(60)  # Check every minute
            self.cleanup_expired()
    
    def create_session(self, persona: str, env: Any) -> Session:
        """
        Create a new session
        
        Args:
            persona: Debtor persona type
            env: DebtCollectionEnv instance
            
        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        
        session = Session(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            persona=persona,
            env=env
        )
        
        with self._lock:
            self.sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session if found and not expired, None otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                # Check if expired
                if time.time() - session.last_accessed > self.ttl_seconds:
                    del self.sessions[session_id]
                    return None
                session.touch()
                return session
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions
        
        Returns:
            Number of sessions removed
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            expired_ids = [
                sid for sid, session in self.sessions.items()
                if now - session.last_accessed > self.ttl_seconds
            ]
            for sid in expired_ids:
                del self.sessions[sid]
                removed += 1
        
        if removed > 0:
            print(f"[SessionManager] Cleaned up {removed} expired sessions")
        
        return removed
    
    def get_active_count(self) -> int:
        """Get number of active sessions"""
        with self._lock:
            return len(self.sessions)
    
    def get_all_sessions(self) -> Dict[str, dict]:
        """Get summary of all sessions (for debugging)"""
        with self._lock:
            return {
                sid: {
                    "persona": s.persona,
                    "is_done": s.is_episode_done,
                    "turns": len(s.conversation_history),
                    "age_seconds": int(time.time() - s.created_at)
                }
                for sid, s in self.sessions.items()
            }


# Global session manager instance
session_manager = SessionManager()
