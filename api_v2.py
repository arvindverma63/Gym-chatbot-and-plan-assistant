from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Path as APIPath, Query, Request
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

# --- Models ---

class ProfileUpdate(BaseModel):
    gender: Optional[str] = Field(None, examples=["Male"], description="User gender")
    goal: Optional[str] = Field(None, examples=["muscle_gain"], description="Fitness goal")
    bmi_category: Optional[str] = Field(None, examples=["Normal weight"], description="BMI classification")
    diet_type: Optional[str] = Field(None, examples=["vegan"], description="Dietary preference")
    weight: Optional[float] = Field(None, examples=[75.5], description="Weight in kg")
    height: Optional[float] = Field(None, examples=[180.0], description="Height in cm")
    age: Optional[int] = Field(None, examples=[25], description="Age in years")
    medical_history: Optional[str] = Field(None, examples=["None"], description="Relevant medical conditions")

class ChatRequest(BaseModel):
    message: str = Field(..., examples=["I want to build muscle"], description="The user's input message")
    session_id: str = Field("default", examples=["user_123"], description="Unique session identifier")
    gender: Optional[str] = None
    goal: Optional[str] = None
    bmi_category: Optional[str] = None
    diet_type: Optional[str] = None
    description: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    session_id: str = Field("default", examples=["user_123"])
    gender: Optional[str] = Field(None, examples=["Male"])
    goal: Optional[str] = Field(None, examples=["muscle_gain"])
    bmi_category: Optional[str] = Field(None, examples=["Normal weight"])
    diet_type: Optional[str] = Field(None, examples=["vegan"])
    description: Optional[str] = None

class FeedbackRequest(BaseModel):
    rating: str = Field(..., examples=["up"], description="'up'/'like' or 'down'/'dislike'")
    name: Optional[str] = Field(None, examples=["Bicep Curl"], description="Name of exercise or food being rated")
    reply: Optional[str] = None
    message: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

class Exercise(BaseModel):
    name: str
    muscles: str
    instruction: str
    primaryMuscles: List[str]
    secondaryMuscles: List[str]
    bodyParts: List[str]
    category: Optional[str]
    gif_url: Optional[str]
    exerciseId: Optional[str]

class NutritionRecord(BaseModel):
    name: str
    calories: Optional[float]
    protein: Optional[float]
    fiber: Optional[float]
    carbohydrate: Optional[float]
    fat: Optional[float]

class ChatResponse(BaseModel):
    ok: bool
    kind: str = Field(..., description="Type of response: answer, plan, memory, clarify")
    reply: str
    profile: Dict[str, Any]
    exercise_examples: List[Exercise]
    nutrition_examples: List[NutritionRecord]
    suggestions: List[str]
    parsed: Dict[str, Any]
    knowledge_examples: Optional[List[Dict[str, Any]]] = None

class RecommendResponse(BaseModel):
    ok: bool
    kind: str
    reply: str
    profile: Dict[str, Any]
    weekly_guidance: str
    meal_guidance: str
    exercise_examples: List[Exercise]
    nutrition_examples: List[NutritionRecord]
    suggestions: List[str]
    recommendation: Optional[Dict[str, Any]]
    assessment: str
    parsed: Dict[str, Any]

class GenericResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    error: Optional[str] = None

# --- Endpoints ---

@app.get("/health", tags=["System"], response_model=Dict[str, str], summary="Health Check")
async def health():
    """
    Check the status of the API server.
    """
    return {"status": "ok", "version": "2.0.0"}

@app.get("/options", tags=["System"], summary="Get Form Options")
async def options():
    """
    Retrieve valid options for profile fields like gender, goal, BMI, and diet.
    """
    return assistant.form_options()

@app.post("/chat", tags=["AI Core"], response_model=ChatResponse, summary="Conversational AI")
async def chat(request: ChatRequest):
    """
    Primary endpoint for chatting with FitPax AI. 
    It maintains context using the session_id and provides exercise/nutrition suggestions.
    """
    result = assistant.chat(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Chat failed"))
    return result

@app.post("/recommend", tags=["AI Core"], response_model=RecommendResponse, summary="Generate Fitness Plan")
async def recommend(request: RecommendRequest):
    """
    Generates a personalized workout and meal plan based on the user's profile and goal.
    """
    result = assistant.recommend(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Recommendation failed"))
    return result

@app.get("/profile/{session_id}", tags=["User Management"], response_model=Dict[str, Any], summary="Get User Profile")
async def get_profile(session_id: str = APIPath(..., description="The unique session ID for the user")):
    """
    Retrieves the current physiological and preference profile associated with the session.
    """
    memory = assistant._load_memory(session_id)
    return memory.get("profile", {})

@app.post("/profile/{session_id}", tags=["User Management"], response_model=GenericResponse, summary="Update User Profile")
async def update_profile(
    session_id: str = APIPath(..., description="The unique session ID for the user"), 
    profile: ProfileUpdate = ...
):
    """
    Updates specific fields in the user's profile for the given session.
    """
    memory = assistant._load_memory(session_id)
    current_profile = memory.get("profile", {})
    
    update_data = profile.model_dump(exclude_unset=True)
    current_profile.update(update_data)
    
    memory["profile"] = current_profile
    assistant._save_memory(session_id, memory)
    return {"ok": True, "message": "Profile updated successfully"}

@app.get("/history/{session_id}", tags=["User Management"], response_model=List[Dict[str, Any]], summary="Get Session History")
async def get_history(session_id: str = APIPath(..., description="The unique session ID for the user")):
    """
    Retrieves the full list of interactions (messages and replies) for the session.
    """
    memory = assistant._load_memory(session_id)
    return memory.get("interactions", [])

@app.get("/exercises", tags=["Knowledge Search"], response_model=List[Exercise], summary="Search Exercises")
async def search_exercises(
    q: Optional[str] = Query(None, description="Keywords to search in exercise names or instructions"), 
    muscle: Optional[str] = Query(None, description="Specific muscle group to filter by"), 
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results to return")
):
    """
    Searches the internal knowledge base for exercises matching the query or muscle group.
    """
    pool = assistant.visual_exercises + assistant.exercises
    results = pool
    
    if q:
        q = q.lower()
        results = [e for e in results if q in str(e.get("name", "")).lower() or q in str(e.get("instruction", "")).lower()]
    
    if muscle:
        muscle = muscle.lower()
        results = [e for e in results if muscle in str(e.get("muscles", "")).lower() or any(muscle in str(m).lower() for m in e.get("primaryMuscles", []))]
        
    return results[:limit]

@app.get("/nutrition", tags=["Knowledge Search"], response_model=List[NutritionRecord], summary="Search Nutrition")
async def search_nutrition(
    q: str = Query(..., description="Food name or search term"), 
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results to return")
):
    """
    Searches the nutrition database for specific food items and their caloric/macronutrient breakdown.
    """
    q = q.lower()
    results = [n for n in assistant.nutrition if q in str(n.get("name", "")).lower()]
    return results[:limit]

@app.post("/feedback", tags=["Maintenance"], response_model=GenericResponse, summary="Submit Feedback")
async def feedback(request: FeedbackRequest):
    """
    Records user feedback (likes/dislikes) to improve the AI's future ranking and recommendations.
    """
    result = assistant.record_feedback(request.model_dump())
    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Feedback failed"))
    return result

@app.post("/retrain", tags=["Maintenance"], response_model=Dict[str, Any], summary="Refresh Knowledge Base")
async def retrain():
    """
    Triggers a manual reload of all source data (CSVs, JSONs) into the assistant's memory.
    """
    return assistant.retrain()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_v2:app", host="0.0.0.0", port=port, reload=True)


