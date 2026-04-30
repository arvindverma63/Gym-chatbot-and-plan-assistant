from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from assistant_core import FitPaxAssistant

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
assistant = FitPaxAssistant(BASE_DIR)

app = FastAPI(
    title="FitPax Pro AI API",
    description="Advanced Gym Chatbot and Plan Assistant API v2 (FastAPI Powered)",
    version="2.0.0",
)

# Enable CORS for cross-origin integration (e.g. with a Laravel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class ProfileUpdate(BaseModel):
    gender: Optional[str] = None
    goal: Optional[str] = None
    bmi_category: Optional[str] = None
    diet_type: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    age: Optional[int] = None
    medical_history: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    gender: Optional[str] = None
    goal: Optional[str] = None
    bmi_category: Optional[str] = None
    diet_type: Optional[str] = None
    description: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    session_id: str = "default"
    gender: Optional[str] = None
    goal: Optional[str] = None
    bmi_category: Optional[str] = None
    diet_type: Optional[str] = None
    description: Optional[str] = None

class FeedbackRequest(BaseModel):
    rating: str
    name: Optional[str] = None
    reply: Optional[str] = None
    message: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

# --- Endpoints ---

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}

@app.get("/options")
async def options():
    """Get available form options for gender, goals, etc."""
    return assistant.form_options()

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for conversational AI."""
    result = assistant.chat(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Chat failed"))
    return result

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Specific endpoint for generating workout and meal plans."""
    result = assistant.recommend(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Recommendation failed"))
    return result

@app.get("/profile/{session_id}")
async def get_profile(session_id: str):
    """Retrieve the current user profile from session memory."""
    memory = assistant._load_memory(session_id)
    return memory.get("profile", {})

@app.post("/profile/{session_id}")
async def update_profile(session_id: str, profile: ProfileUpdate):
    """Manually update the user profile for a session."""
    memory = assistant._load_memory(session_id)
    current_profile = memory.get("profile", {})
    
    # Update only provided fields
    update_data = profile.model_dump(exclude_unset=True)
    current_profile.update(update_data)
    
    memory["profile"] = current_profile
    assistant._save_memory(session_id, memory)
    return {"ok": True, "profile": current_profile}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve the interaction history for a session."""
    memory = assistant._load_memory(session_id)
    return memory.get("interactions", [])

@app.get("/exercises")
async def search_exercises(q: Optional[str] = None, muscle: Optional[str] = None, limit: int = 20):
    """Search and filter exercises from the knowledge base."""
    pool = assistant.visual_exercises + assistant.exercises
    results = pool
    
    if q:
        q = q.lower()
        results = [e for e in results if q in str(e.get("name", "")).lower() or q in str(e.get("instruction", "")).lower()]
    
    if muscle:
        muscle = muscle.lower()
        results = [e for e in results if muscle in str(e.get("muscles", "")).lower() or any(muscle in str(m).lower() for m in e.get("primaryMuscles", []))]
        
    return results[:limit]

@app.get("/nutrition")
async def search_nutrition(q: str, limit: int = 20):
    """Search for nutrition data for specific foods."""
    q = q.lower()
    results = [n for n in assistant.nutrition if q in str(n.get("name", "")).lower()]
    return results[:limit]

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Submit user feedback for AI responses."""
    result = assistant.record_feedback(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Feedback failed"))
    return result

@app.post("/retrain")
async def retrain():
    """Trigger a refresh of the assistant's knowledge from local files."""
    return assistant.retrain()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_v2:app", host="0.0.0.0", port=port, reload=True)
