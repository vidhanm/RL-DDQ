#!/usr/bin/env python
"""
DDQ Agent Server Launcher
Simple entry point to start the FastAPI backend server
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Launch DDQ Agent Server")
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run on (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 DDQ Debt Collection Agent Server")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API docs at http://localhost:{args.port}/api/docs")
    print("=" * 60)
    
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
