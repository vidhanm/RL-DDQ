"""
DDQ Agent Backend - FastAPI Application
Main entry point for the REST API server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routers import conversation_router, models_router, training_router
from backend.dependencies import initialize_llm_client, load_model, get_agent, get_llm_client
from backend.services.session import session_manager
from backend.schemas.models import HealthResponse

# Create FastAPI app
app = FastAPI(
    title="DDQ Debt Collection Agent API",
    description="REST API for the Deep Q-Learning with Dyna (DDQ) debt collection agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS - allow frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(conversation_router)
app.include_router(models_router)
app.include_router(training_router)

# Static files - serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Serve figures directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("DDQ Debt Collection Agent API")
    print("=" * 60)
    
    # Initialize LLM client
    initialize_llm_client()
    
    # Start session cleanup thread
    session_manager.start_cleanup_thread()
    
    # Load default model (DQN)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    for model_type in ["dqn", "ddq"]:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_final.pt")
        if os.path.exists(checkpoint_path):
            try:
                load_model(model_type, checkpoint_path)
                break
            except Exception as e:
                print(f"Failed to load {model_type}: {e}")
    
    print("\nAPI ready at http://localhost:8000")
    print("Frontend at http://localhost:8000/")
    print("API docs at http://localhost:8000/api/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    session_manager.stop_cleanup_thread()
    print("Server shutting down...")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend index.html"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. API available at /api/docs"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agent = get_agent()
    llm = get_llm_client()
    
    return HealthResponse(
        status="healthy",
        llm_available=llm is not None,
        models_loaded={
            "dqn": agent is not None and hasattr(agent, 'policy_net'),
            "ddq": agent is not None and hasattr(agent, 'world_model')
        }
    )


@app.get("/api/sessions")
async def list_sessions():
    """Debug endpoint: list active sessions"""
    return {
        "active_count": session_manager.get_active_count(),
        "sessions": session_manager.get_all_sessions()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
