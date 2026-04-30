# FitPax Pro AI API v2: Complete Route Documentation

This document provides a detailed specification of all API endpoints available in the **api_v2.py** server.

## Base URL
Default: `http://127.0.0.1:8000`

---

## 1. System & Utility Endpoints

### GET `/health`
Check if the API server is alive.
- **Response (200 OK)**:
  ```json
  {
    "status": "ok",
    "version": "2.0.0"
  }
  ```

### GET `/options`
Retrieves available configuration options for user profiles (goals, BMI categories, etc.).
- **Response (200 OK)**:
  ```json
  {
    "gender": [{"value": "Male", "label": "Male"}, ...],
    "goal": [{"value": "fat_burn", "label": "Fat burn"}, ...],
    "bmi_category": [{"value": "Underweight", "label": "Underweight"}, ...],
    "diet_type": [{"value": "vegan", "label": "Vegan"}, ...]
  }
  ```

---

## 2. Core AI Endpoints

### POST `/chat`
Primary endpoint for conversational interaction. Handles memory and context automatically.
- **Request Body**:
  ```json
  {
    "message": "I want to build muscle",
    "session_id": "user_session_abc",
    "gender": "Male",
    "goal": "muscle_gain",
    "bmi_category": "Normal weight",
    "diet_type": "vegan",
    "description": "Optional long-form context",
    "state": {} 
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "ok": true,
    "kind": "answer | plan | memory | clarify",
    "reply": "The textual response from the AI",
    "profile": { "gender": "...", "goal": "...", "bmi_category": "...", "diet_type": "..." },
    "exercise_examples": [
      { "name": "Squat", "muscles": "Quads", "gif_url": "...", "instruction": "..." }
    ],
    "nutrition_examples": [
      { "name": "Tofu", "calories": 100, "protein": 10, "fiber": 2 }
    ],
    "suggestions": ["Tell me about cardio", "What should I eat for dinner?"],
    "parsed": { "intent": "workout_advice", "confidence": 0.95 },
    "knowledge_examples": [...]
  }
  ```

### POST `/recommend`
Generates a structured workout and meal plan based on user profile.
- **Request Body**:
  ```json
  {
    "session_id": "user_123",
    "gender": "Male",
    "goal": "fat_burn",
    "bmi_category": "Obesity",
    "diet_type": "vegetarian"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "ok": true,
    "kind": "plan",
    "reply": "Here is your plan...",
    "profile": { ... },
    "weekly_guidance": "Summary of frequency and intensity",
    "meal_guidance": "Nutrition strategy",
    "exercise_examples": [...],
    "nutrition_examples": [...],
    "recommendation": {
      "exercise_schedule": "...",
      "meal_plan": "..."
    },
    "assessment": "Physiological assessment text (BMI, BMR, TDEE, etc.)",
    "suggestions": [...]
  }
  ```

---

## 3. Session & Profile Management

### GET `/profile/{session_id}`
Retrieve the stored profile for a specific session.
- **Response (200 OK)**:
  ```json
  {
    "gender": "Male",
    "goal": "muscle_gain",
    "bmi_category": "Normal weight",
    "weight": 75,
    "height": 180
  }
  ```

### POST `/profile/{session_id}`
Manually update or initialize a user profile.
- **Request Body**:
  ```json
  {
    "gender": "Male",
    "goal": "muscle_gain",
    "weight": 80,
    "height": 185,
    "age": 25,
    "medical_history": "None"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "ok": true,
    "profile": { ... updated profile ... }
  }
  ```

### GET `/history/{session_id}`
Retrieve the full list of interactions for a session.
- **Response (200 OK)**:
  ```json
  [
    {
      "type": "question",
      "message": "Hello",
      "reply": "FitPax AI is ready...",
      "created_at": "..."
    },
    ...
  ]
  ```

---

## 4. Knowledge Base Search

### GET `/exercises`
Search the exercise database.
- **Query Parameters**:
  - `q`: Search keyword (name or instruction)
  - `muscle`: Filter by primary muscle group
  - `limit`: Default 20
- **Response (200 OK)**:
  ```json
  [
    {
      "name": "Bicep Curl",
      "muscles": "Bicep",
      "instruction": "...",
      "gif_url": "/exercise-gif/..."
    },
    ...
  ]
  ```

### GET `/nutrition`
Search for food nutrition data.
- **Query Parameters**:
  - `q`: Food name search term
- **Response (200 OK)**:
  ```json
  [
    {
      "name": "Oatmeal",
      "calories": 150,
      "protein": 5,
      "fiber": 4,
      "carbohydrate": 27,
      "fat": 3
    },
    ...
  ]
  ```

---

## 5. Feedback & Maintenance

### POST `/feedback`
Submit user feedback on a response to improve future rankings.
- **Request Body**:
  ```json
  {
    "rating": "up | down | like | dislike",
    "name": "Exercise or food name",
    "reply": "The AI's response text",
    "message": "The user's original message"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "ok": true,
    "message": "Feedback saved.",
    "feedback_count": 42
  }
  ```

### POST `/retrain`
Forces the assistant to reload all local datasets (CSVs, JSONs, Knowledge Base).
- **Response (200 OK)**:
  ```json
  {
    "ok": true,
    "message": "Assistant refreshed from local datasets.",
    "exercise_count": 1250,
    "nutrition_count": 8500
  }
  ```
