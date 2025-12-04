from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Updated Groq API endpoint

# Constants
MEMORY_EXTRACTION_SYSTEM_PROMPT = """You are an AI that analyzes chat messages to extract structured information about the user.
Extract the following information from the messages:
1. User Preferences (e.g., likes, dislikes, habits)
2. Emotional Patterns (e.g., how they express emotions, common emotional states)
3. Facts (e.g., personal information, background, important events)

Return the information in JSON format with the following structure:
{
    "preferences": {
        "category1": "value1",
        "category2": "value2"
    },
    "emotional_patterns": ["pattern1", "pattern2"],
    "facts": {
        "category1": "value1",
        "category2": "value2"
    }
}"""

PERSONALITY_PROMPTS = {
    "mentor": "You are a calm and experienced mentor. Provide helpful, structured guidance. Be supportive and encouraging, but also direct when needed. Offer clear explanations and practical advice.",
    "friend": "You are a witty and friendly companion. Keep the conversation casual, fun, and engaging. Use humor when appropriate and be empathetic. Keep responses concise and relatable.",
    "therapist": "You are an empathetic therapist. Be understanding, non-judgmental, and supportive. Help the user explore their thoughts and feelings. Ask thoughtful questions and provide gentle guidance.",
    "professional": "You are a professional assistant. Be clear, concise, and to the point. Focus on providing accurate information and practical solutions. Maintain a polite but formal tone."
}

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class MemoryData(BaseModel):
    preferences: Dict[str, Any]
    emotional_patterns: List[str]
    facts: Dict[str, Any]

# In-memory storage (replace with a database in production)
user_memory = {}

def call_groq_api(messages, temperature=0.7, max_tokens=1000, model="mixtral-8x7b-32768"):
    """Helper function to call the Groq API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,  # Using mixtral-8x7b-32768 as the default model
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Groq API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_messages(messages: List[ChatMessage]):
    """
    Analyze chat messages and extract memory data using Groq API
    """
    if len(messages) < 1:
        raise HTTPException(status_code=400, detail="At least one message is required")
    
    try:
        # Prepare messages for the LLM
        messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        # Prepare the API call
        api_messages = [
            {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract information from these messages:\n\n{messages_text}"}
        ]
        
        # Call the Groq API
        response = call_groq_api(
            messages=api_messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse the response
        try:
            content = response['choices'][0]['message']['content']
            # Clean the response to ensure it's valid JSON
            content = content.strip().strip('```json').strip('```').strip()
            memory_data = json.loads(content)
            
            # Ensure the structure is correct
            memory_data = {
                "preferences": memory_data.get("preferences", {}),
                "emotional_patterns": memory_data.get("emotional_patterns", []),
                "facts": memory_data.get("facts", {})
            }
            return {"status": "success", "memory": memory_data}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing API response: {str(e)}")
            print(f"Response content: {response}")
            # Fallback to a basic extraction if parsing fails
            return {
                "status": "success",
                "memory": {
                    "preferences": {"communication_style": "concise"},
                    "emotional_patterns": [],
                    "facts": {}
                }
            }
            
    except Exception as e:
        print(f"Error in analyze_messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing messages: {str(e)}")

@app.post("/api/generate")
async def generate_response(message: str, personality: str, memory: MemoryData):
    """
    Generate a response based on the message, selected personality, and memory
    """
    if personality not in PERSONALITY_PROMPTS:
        personality = "professional"
    
    try:
        # Format the memory into a string for the prompt
        memory_str = ""
        if memory.preferences:
            memory_str += "\nUser Preferences:\n"
            for k, v in memory.preferences.items():
                memory_str += f"- {k}: {v}\n"
        
        if memory.emotional_patterns:
            memory_str += "\nEmotional Patterns:\n"
            for pattern in memory.emotional_patterns:
                memory_str += f"- {pattern}\n"
        
        if memory.facts:
            memory_str += "\nKnown Facts:\n"
            for k, v in memory.facts.items():
                memory_str += f"- {k}: {v}\n"
        
        # Create the system prompt with personality and memory
        system_prompt = f"""{PERSONALITY_PROMPTS[personality]}
        
        Here's what you know about the user:
        {memory_str}
        
        Keep your response concise and in character. Use this context to provide more personalized responses."""
        
        # Prepare the API call
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Call the Groq API
        response = call_groq_api(
            messages=api_messages,
            temperature=0.7 if personality == "friend" else 0.5,
            max_tokens=500
        )
        
        # Get the generated response
        ai_response = response['choices'][0]['message']['content'].strip()
        
        return {"response": ai_response}
        
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        # Fallback response if there's an error
        return {
            "response": f"I'm having trouble generating a response right now. Please try again later. (Error: {str(e)})"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)