"""
Database models and repository for storing adversarial battle history.
Uses SQLAlchemy 2.0+ with async SQLite support.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.future import select

# Database file location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "battles.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# SQLAlchemy setup
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
SYNC_DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()


# ============== SQLAlchemy Models ==============

class TrainingSession(Base):
    """A complete self-play training run"""
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_generations = Column(Integer, nullable=False)
    episodes_per_gen = Column(Integer, nullable=False)
    use_llm = Column(Boolean, default=False)
    zero_sum = Column(Boolean, default=True)
    final_collector_win_rate = Column(Float, nullable=True)
    final_adversary_win_rate = Column(Float, nullable=True)
    status = Column(String(20), default="running")  # running, completed, stopped, error
    
    # Relationships
    generations = relationship("Generation", back_populates="session", cascade="all, delete-orphan")


class Generation(Base):
    """A single generation within a training session"""
    __tablename__ = "generations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False)
    generation_num = Column(Integer, nullable=False)
    collector_win_rate = Column(Float, nullable=False)
    adversary_win_rate = Column(Float, nullable=False)
    avg_collector_reward = Column(Float, nullable=False)
    avg_adversary_reward = Column(Float, nullable=False)
    collector_strategy_dist = Column(JSON, nullable=True)
    adversary_strategy_dist = Column(JSON, nullable=True)
    completed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="generations")
    episodes = relationship("Episode", back_populates="generation", cascade="all, delete-orphan")


class Episode(Base):
    """A single episode (battle) within a generation"""
    __tablename__ = "episodes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False)
    episode_num = Column(Integer, nullable=False)
    outcome = Column(String(20), nullable=False)  # collector_win, adversary_win, draw
    collector_total_reward = Column(Float, nullable=False)
    adversary_total_reward = Column(Float, nullable=False)
    num_turns = Column(Integer, nullable=False)
    completed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    generation = relationship("Generation", back_populates="episodes")
    turns = relationship("BattleTurn", back_populates="episode", cascade="all, delete-orphan")


class BattleTurn(Base):
    """A single turn in a battle episode"""
    __tablename__ = "battle_turns"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False)
    turn_num = Column(Integer, nullable=False)
    collector_strategy = Column(String(50), nullable=True)
    collector_utterance = Column(Text, nullable=True)
    adversary_strategy = Column(String(50), nullable=True)
    adversary_response = Column(Text, nullable=True)
    collector_reward = Column(Float, default=0.0)
    adversary_reward = Column(Float, default=0.0)
    
    # Relationships
    episode = relationship("Episode", back_populates="turns")


# ============== Database Engine & Session ==============

# Async engine for FastAPI
async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for background threads (training)
sync_engine = create_engine(SYNC_DATABASE_URL, echo=False)
SyncSessionLocal = sessionmaker(bind=sync_engine)


async def init_db():
    """Initialize database tables (async)"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"✓ Database initialized at {DB_PATH}")


def init_db_sync():
    """Initialize database tables (sync - for training thread)"""
    Base.metadata.create_all(bind=sync_engine)
    print(f"✓ Database initialized at {DB_PATH}")


@asynccontextmanager
async def get_async_session():
    """Async context manager for database sessions"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session():
    """Get sync session for training thread"""
    return SyncSessionLocal()


# ============== Repository Functions (Sync - for training) ==============

class BattleRepository:
    """Repository for managing battle data in training thread (sync operations)"""
    
    def __init__(self):
        self.session: Optional[Any] = None
        self.current_session_id: Optional[int] = None
        self.current_generation_id: Optional[int] = None
        self.current_episode_id: Optional[int] = None
        self.episode_turns: List[Dict] = []
    
    def start_training_session(
        self,
        total_generations: int,
        episodes_per_gen: int,
        use_llm: bool,
        zero_sum: bool
    ) -> int:
        """Create a new training session and return its ID"""
        init_db_sync()  # Ensure tables exist
        
        self.session = get_sync_session()
        training_session = TrainingSession(
            total_generations=total_generations,
            episodes_per_gen=episodes_per_gen,
            use_llm=use_llm,
            zero_sum=zero_sum,
            status="running"
        )
        self.session.add(training_session)
        self.session.commit()
        self.current_session_id = training_session.id
        print(f"📊 Started training session #{self.current_session_id}")
        return self.current_session_id
    
    def start_generation(self, generation_num: int) -> int:
        """Create a generation record at the start and return its ID"""
        if not self.session or not self.current_session_id:
            return -1
        
        generation = Generation(
            session_id=self.current_session_id,
            generation_num=generation_num,
            collector_win_rate=0.0,  # Will be updated at end
            adversary_win_rate=0.0,
            avg_collector_reward=0.0,
            avg_adversary_reward=0.0,
            collector_strategy_dist={},
            adversary_strategy_dist={}
        )
        self.session.add(generation)
        self.session.commit()
        self.current_generation_id = generation.id
        return generation.id
    
    def start_episode(self, episode_num: int) -> None:
        """Mark the start of a new episode"""
        self.current_episode_id = None
        self.episode_turns = []
    
    def add_turn(
        self,
        turn_num: int,
        collector_strategy: str,
        collector_utterance: str,
        adversary_strategy: str,
        adversary_response: str,
        collector_reward: float = 0.0,
        adversary_reward: float = 0.0
    ) -> None:
        """Buffer a turn (will be saved when episode ends)"""
        self.episode_turns.append({
            "turn_num": turn_num,
            "collector_strategy": collector_strategy,
            "collector_utterance": collector_utterance,
            "adversary_strategy": adversary_strategy,
            "adversary_response": adversary_response,
            "collector_reward": collector_reward,
            "adversary_reward": adversary_reward
        })
    
    def save_episode_with_turns(
        self,
        episode_num: int,
        outcome: str,
        collector_total_reward: float,
        adversary_total_reward: float,
        turns: List[Dict]
    ) -> int:
        """Save episode and its turns to database"""
        if not self.session or not self.current_generation_id:
            return -1
        
        episode = Episode(
            generation_id=self.current_generation_id,
            episode_num=episode_num,
            outcome=outcome,
            collector_total_reward=collector_total_reward,
            adversary_total_reward=adversary_total_reward,
            num_turns=len(turns)
        )
        self.session.add(episode)
        self.session.flush()  # Get the episode ID
        
        # Add all turns
        for turn_data in turns:
            turn = BattleTurn(episode_id=episode.id, **turn_data)
            self.session.add(turn)
        
        self.session.commit()
        return episode.id
    
    def end_episode(
        self,
        generation_id: int,
        episode_num: int,
        outcome: str,
        collector_total_reward: float,
        adversary_total_reward: float
    ) -> int:
        """Save the completed episode with all its turns"""
        if not self.session:
            return -1
        
        episode = Episode(
            generation_id=generation_id,
            episode_num=episode_num,
            outcome=outcome,
            collector_total_reward=collector_total_reward,
            adversary_total_reward=adversary_total_reward,
            num_turns=len(self.episode_turns)
        )
        self.session.add(episode)
        self.session.flush()  # Get the episode ID
        
        # Add all buffered turns
        for turn_data in self.episode_turns:
            turn = BattleTurn(episode_id=episode.id, **turn_data)
            self.session.add(turn)
        
        self.session.commit()
        self.current_episode_id = episode.id
        self.episode_turns = []
        return episode.id
    
    def end_generation(
        self,
        generation_num: int,
        collector_win_rate: float,
        adversary_win_rate: float,
        avg_collector_reward: float,
        avg_adversary_reward: float,
        collector_strategy_dist: Optional[Dict] = None,
        adversary_strategy_dist: Optional[Dict] = None
    ) -> int:
        """Update generation statistics at end of generation"""
        if not self.session or not self.current_generation_id:
            return -1
        
        generation = self.session.get(Generation, self.current_generation_id)
        if generation:
            generation.collector_win_rate = collector_win_rate
            generation.adversary_win_rate = adversary_win_rate
            generation.avg_collector_reward = avg_collector_reward
            generation.avg_adversary_reward = avg_adversary_reward
            generation.collector_strategy_dist = collector_strategy_dist or {}
            generation.adversary_strategy_dist = adversary_strategy_dist or {}
            generation.completed_at = datetime.utcnow()
            self.session.commit()
        
        return self.current_generation_id
    
    def end_training_session(
        self,
        status: str,
        final_collector_win_rate: float,
        final_adversary_win_rate: float
    ) -> None:
        """Mark training session as complete"""
        if not self.session or not self.current_session_id:
            return
        
        training_session = self.session.get(TrainingSession, self.current_session_id)
        if training_session:
            training_session.ended_at = datetime.utcnow()
            training_session.status = status
            training_session.final_collector_win_rate = final_collector_win_rate
            training_session.final_adversary_win_rate = final_adversary_win_rate
            self.session.commit()
            print(f"📊 Training session #{self.current_session_id} ended with status: {status}")
        
        self.session.close()
        self.session = None
        self.current_session_id = None


# ============== Async Query Functions (for API) ==============

async def get_all_sessions(limit: int = 50, offset: int = 0) -> List[TrainingSession]:
    """Get all training sessions, most recent first"""
    async with get_async_session() as session:
        result = await session.execute(
            select(TrainingSession)
            .order_by(TrainingSession.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()


async def get_session_by_id(session_id: int) -> Optional[TrainingSession]:
    """Get a specific training session with its generations"""
    async with get_async_session() as session:
        result = await session.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        return result.scalar_one_or_none()


async def get_generation_by_id(generation_id: int) -> Optional[Generation]:
    """Get a specific generation with its episodes"""
    async with get_async_session() as session:
        result = await session.execute(
            select(Generation).where(Generation.id == generation_id)
        )
        return result.scalar_one_or_none()


async def get_episode_by_id(episode_id: int) -> Optional[Episode]:
    """Get a specific episode with its turns"""
    async with get_async_session() as session:
        result = await session.execute(
            select(Episode).where(Episode.id == episode_id)
        )
        return result.scalar_one_or_none()


async def get_generations_for_session(session_id: int) -> List[Generation]:
    """Get all generations for a training session"""
    async with get_async_session() as session:
        result = await session.execute(
            select(Generation)
            .where(Generation.session_id == session_id)
            .order_by(Generation.generation_num)
        )
        return result.scalars().all()


async def get_episodes_for_generation(generation_id: int) -> List[Episode]:
    """Get all episodes for a generation"""
    async with get_async_session() as session:
        result = await session.execute(
            select(Episode)
            .where(Episode.generation_id == generation_id)
            .order_by(Episode.episode_num)
        )
        return result.scalars().all()


async def get_turns_for_episode(episode_id: int) -> List[BattleTurn]:
    """Get all turns for an episode"""
    async with get_async_session() as session:
        result = await session.execute(
            select(BattleTurn)
            .where(BattleTurn.episode_id == episode_id)
            .order_by(BattleTurn.turn_num)
        )
        return result.scalars().all()


async def delete_session(session_id: int) -> bool:
    """Delete a training session and all related data"""
    async with get_async_session() as session:
        result = await session.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        training_session = result.scalar_one_or_none()
        if training_session:
            await session.delete(training_session)
            return True
        return False


async def get_session_count() -> int:
    """Get total number of training sessions"""
    async with get_async_session() as session:
        from sqlalchemy import func
        result = await session.execute(
            select(func.count(TrainingSession.id))
        )
        return result.scalar() or 0
