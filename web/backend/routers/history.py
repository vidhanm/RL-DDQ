"""
Battle History Router
REST API endpoints for querying adversarial battle history from SQLite database.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List

from web.backend.database import (
    get_all_sessions,
    get_session_by_id,
    get_generation_by_id,
    get_episode_by_id,
    get_generations_for_session,
    get_episodes_for_generation,
    get_turns_for_episode,
    delete_session,
    get_session_count,
    init_db
)

from web.backend.schemas.models import (
    BattleHistoryResponse,
    SessionSummary,
    SessionDetailResponse,
    GenerationSummary,
    GenerationDetailResponse,
    EpisodeSummary,
    EpisodeDetailResponse,
    BattleTurnResponse
)

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/sessions", response_model=BattleHistoryResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get paginated list of training sessions.
    Most recent sessions are returned first.
    """
    sessions = await get_all_sessions(limit=limit, offset=offset)
    total = await get_session_count()
    
    session_summaries = []
    for session in sessions:
        # Get generation count for this session
        generations = await get_generations_for_session(session.id)
        
        session_summaries.append(SessionSummary(
            id=session.id,
            started_at=session.started_at.isoformat() if session.started_at else "",
            ended_at=session.ended_at.isoformat() if session.ended_at else None,
            total_generations=session.total_generations,
            episodes_per_gen=session.episodes_per_gen,
            use_llm=session.use_llm,
            zero_sum=session.zero_sum,
            final_collector_win_rate=session.final_collector_win_rate,
            final_adversary_win_rate=session.final_adversary_win_rate,
            status=session.status,
            generation_count=len(generations)
        ))
    
    return BattleHistoryResponse(
        sessions=session_summaries,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: int):
    """
    Get detailed information about a specific training session,
    including all generation summaries.
    """
    session = await get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    generations = await get_generations_for_session(session_id)
    
    generation_summaries = []
    for gen in generations:
        episodes = await get_episodes_for_generation(gen.id)
        generation_summaries.append(GenerationSummary(
            id=gen.id,
            generation_num=gen.generation_num,
            collector_win_rate=gen.collector_win_rate,
            adversary_win_rate=gen.adversary_win_rate,
            avg_collector_reward=gen.avg_collector_reward,
            avg_adversary_reward=gen.avg_adversary_reward,
            episode_count=len(episodes),
            completed_at=gen.completed_at.isoformat() if gen.completed_at else None
        ))
    
    return SessionDetailResponse(
        id=session.id,
        started_at=session.started_at.isoformat() if session.started_at else "",
        ended_at=session.ended_at.isoformat() if session.ended_at else None,
        total_generations=session.total_generations,
        episodes_per_gen=session.episodes_per_gen,
        use_llm=session.use_llm,
        zero_sum=session.zero_sum,
        final_collector_win_rate=session.final_collector_win_rate,
        final_adversary_win_rate=session.final_adversary_win_rate,
        status=session.status,
        generations=generation_summaries
    )


@router.get("/generations/{generation_id}", response_model=GenerationDetailResponse)
async def get_generation_detail(generation_id: int):
    """
    Get detailed information about a specific generation,
    including all episode summaries.
    """
    generation = await get_generation_by_id(generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail=f"Generation {generation_id} not found")
    
    episodes = await get_episodes_for_generation(generation_id)
    
    episode_summaries = [
        EpisodeSummary(
            id=ep.id,
            episode_num=ep.episode_num,
            outcome=ep.outcome,
            collector_total_reward=ep.collector_total_reward,
            adversary_total_reward=ep.adversary_total_reward,
            num_turns=ep.num_turns,
            completed_at=ep.completed_at.isoformat() if ep.completed_at else None
        )
        for ep in episodes
    ]
    
    return GenerationDetailResponse(
        id=generation.id,
        generation_num=generation.generation_num,
        collector_win_rate=generation.collector_win_rate,
        adversary_win_rate=generation.adversary_win_rate,
        avg_collector_reward=generation.avg_collector_reward,
        avg_adversary_reward=generation.avg_adversary_reward,
        collector_strategy_dist=generation.collector_strategy_dist,
        adversary_strategy_dist=generation.adversary_strategy_dist,
        completed_at=generation.completed_at.isoformat() if generation.completed_at else None,
        episodes=episode_summaries
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeDetailResponse)
async def get_episode_detail(episode_id: int):
    """
    Get detailed information about a specific episode,
    including all battle turns with dialogue.
    """
    episode = await get_episode_by_id(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    turns = await get_turns_for_episode(episode_id)
    
    turn_responses = [
        BattleTurnResponse(
            id=turn.id,
            turn_num=turn.turn_num,
            collector_strategy=turn.collector_strategy,
            collector_utterance=turn.collector_utterance,
            adversary_strategy=turn.adversary_strategy,
            adversary_response=turn.adversary_response,
            collector_reward=turn.collector_reward,
            adversary_reward=turn.adversary_reward
        )
        for turn in turns
    ]
    
    return EpisodeDetailResponse(
        id=episode.id,
        episode_num=episode.episode_num,
        outcome=episode.outcome,
        collector_total_reward=episode.collector_total_reward,
        adversary_total_reward=episode.adversary_total_reward,
        num_turns=episode.num_turns,
        completed_at=episode.completed_at.isoformat() if episode.completed_at else None,
        turns=turn_responses
    )


@router.delete("/sessions/{session_id}")
async def delete_training_session(session_id: int):
    """
    Delete a training session and all associated data.
    This includes all generations, episodes, and battle turns.
    """
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {"message": f"Session {session_id} deleted successfully"}


@router.get("/stats")
async def get_history_stats():
    """
    Get overall statistics about battle history.
    """
    total_sessions = await get_session_count()
    sessions = await get_all_sessions(limit=100)
    
    total_generations = 0
    total_episodes = 0
    completed_sessions = 0
    avg_collector_win_rate = 0.0
    
    for session in sessions:
        if session.status == "completed":
            completed_sessions += 1
            if session.final_collector_win_rate:
                avg_collector_win_rate += session.final_collector_win_rate
        
        generations = await get_generations_for_session(session.id)
        total_generations += len(generations)
        
        for gen in generations:
            episodes = await get_episodes_for_generation(gen.id)
            total_episodes += len(episodes)
    
    if completed_sessions > 0:
        avg_collector_win_rate /= completed_sessions
    
    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "total_generations": total_generations,
        "total_episodes": total_episodes,
        "avg_collector_win_rate": round(avg_collector_win_rate, 3)
    }
