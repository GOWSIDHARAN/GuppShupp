"""
Production-ready GuppShupp Application
Advanced AI Personality System with Memory Extraction and Personality Engine
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Import production modules
from models.memory_models import MemoryExtractionRequest, UserMemory
from models.personality_models import (
    PersonalityType, ResponseGenerationRequest, 
    PersonalityTransformationRequest, PersonalityComparison
)
from services.memory_extractor import MemoryExtractor, MemoryExtractionError
from services.personality_engine import PersonalityEngine, PersonalityEngineError
from services.llm_client import LLMClient, LLMClientError
from services.database import DatabaseService, DatabaseError
from utils.logging_config import setup_logging, SecurityLogger, APILogger, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)
security_logger = SecurityLogger()
api_logger = APILogger()

# Global services
llm_client = None
memory_extractor = None
personality_engine = None
database = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting GuppShupp application...")
    
    try:
        # Initialize services
        global llm_client, memory_extractor, personality_engine, database
        
        llm_client = LLMClient()
        memory_extractor = MemoryExtractor(llm_client)
        personality_engine = PersonalityEngine(llm_client)
        database = DatabaseService()
        
        # Health check
        if not llm_client.health_check():
            raise RuntimeError("LLM service is not healthy")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GuppShupp application...")


# Create FastAPI app
app = FastAPI(
    title="GuppShupp AI Personality System",
    description="Production-ready AI system with memory extraction and personality transformation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Request models for API
class ChatMessage(BaseModel):
    role: str
    content: str


class AnalyzeRequest(BaseModel):
    messages: List[ChatMessage]
    user_id: Optional[str] = None


class GenerateRequest(BaseModel):
    message: str
    personality: PersonalityType
    user_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None


class CompareRequest(BaseModel):
    message: str
    personalities: Optional[List[PersonalityType]] = None
    user_id: Optional[str] = None


# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request start
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log request completion
        api_logger.log_request(
            method=request.method,
            endpoint=request.url.path,
            user_id=getattr(request.state, 'user_id', 'anonymous'),
            processing_time=process_time,
            status_code=response.status_code
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        api_logger.log_error(
            method=request.method,
            endpoint=request.url.path,
            user_id=getattr(request.state, 'user_id', 'anonymous'),
            error=str(e)
        )
        raise


# Helper functions
def get_user_id(request: Request) -> str:
    """Extract or generate user ID from request"""
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        user_id = f"anon_{uuid.uuid4().hex[:8]}"
    return user_id


async def get_user_memory(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user memory from database"""
    try:
        return database.get_user_memory(user_id)
    except Exception as e:
        logger.error(f"Error retrieving user memory: {str(e)}")
        return None


# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        llm_healthy = llm_client.health_check() if llm_client else False
        db_healthy = database is not None
        
        return {
            "status": "healthy" if llm_healthy and db_healthy else "unhealthy",
            "services": {
                "llm": "healthy" if llm_healthy else "unhealthy",
                "database": "healthy" if db_healthy else "unhealthy"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/api/analyze")
async def analyze_messages(request: Request, analyze_req: AnalyzeRequest):
    """Analyze chat messages and extract user memory"""
    start_time = time.time()
    user_id = analyze_req.user_id or get_user_id(request)
    request.state.user_id = user_id
    
    try:
        # Create user if doesn't exist
        database.create_user(user_id)
        
        # Build extraction request
        extraction_request = MemoryExtractionRequest(
            messages=[msg.dict() for msg in analyze_req.messages],
            user_id=user_id,
            analysis_depth="comprehensive"
        )
        
        # Extract memory
        memory = await memory_extractor.extract_memory(extraction_request)
        
        # Save to database
        # Use JSON-safe dump so datetime and other types are serializable
        try:
            memory_dict = memory.model_dump(mode="json")  # Pydantic v2
        except AttributeError:
            # Fallback for Pydantic v1-style if ever needed
            from pydantic.json import pydantic_encoder
            memory_dict = json.loads(json.dumps(memory.dict(), default=pydantic_encoder))
        success = database.save_user_memory(
            user_id=user_id,
            memory_data=memory_dict,
            message_count=len(analyze_req.messages),
            confidence_score=0.85  # Would be calculated from memory
        )
        
        # Log security event
        security_logger.log_data_extraction(
            user_id=user_id,
            message_count=len(analyze_req.messages),
            confidence_score=0.85
        )
        
        processing_time = time.time() - start_time
        api_logger.log_performance("/api/analyze", "memory_extraction", processing_time)
        
        return {
            "success": True,
            "user_id": user_id,
            "memory": memory_dict,
            "processing_time": processing_time,
            "message_count": len(analyze_req.messages)
        }
        
    except MemoryExtractionError as e:
        logger.error(f"Memory extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Memory extraction failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/generate")
async def generate_response(request: Request, generate_req: GenerateRequest):
    """Generate personality-based response"""
    start_time = time.time()
    user_id = generate_req.user_id or get_user_id(request)
    request.state.user_id = user_id
    
    try:
        # Get user memory
        user_memory = await get_user_memory(user_id)
        
        # Build generation request
        generation_request = ResponseGenerationRequest(
            user_message=generate_req.message,
            personality=generate_req.personality,
            user_memory=user_memory or generate_req.memory
        )
        
        # Generate response
        response = await personality_engine.generate_response(generation_request)
        
        # Log personality switch
        security_logger.log_personality_switch(
            user_id=user_id,
            from_personality="unknown",
            to_personality=generate_req.personality.value
        )
        
        processing_time = time.time() - start_time
        api_logger.log_performance("/api/generate", "response_generation", processing_time)
        
        return {
            "success": True,
            "user_id": user_id,
            "response": response.response,
            "personality": response.personality.value,
            "personalization_elements": response.personalization_elements,
            "memory_references": response.memory_references,
            "confidence": response.generation_confidence,
            "processing_time": processing_time
        }
        
    except PersonalityEngineError as e:
        logger.error(f"Response generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Generate endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/compare")
async def compare_personalities(request: Request, compare_req: CompareRequest):
    """Compare responses across different personalities"""
    start_time = time.time()
    user_id = compare_req.user_id or get_user_id(request)
    request.state.user_id = user_id
    
    try:
        # Generate base response (professional personality)
        base_request = ResponseGenerationRequest(
            user_message=compare_req.message,
            personality=PersonalityType.PROFESSIONAL,
            user_memory=await get_user_memory(user_id)
        )
        base_response = await personality_engine.generate_response(base_request)
        
        # Compare personalities
        comparison = await personality_engine.compare_personalities(
            user_message=compare_req.message,
            base_response=base_response.response,
            personalities=compare_req.personalities
        )
        
        # Save comparison to database
        for personality, response in comparison.personality_responses.items():
            database.save_personality_response(
                user_id=user_id,
                user_message=compare_req.message,
                base_response=base_response.response,
                personality_type=personality.value,
                personality_response=response
            )
        
        processing_time = time.time() - start_time
        api_logger.log_performance("/api/compare", "personality_comparison", processing_time)
        
        return {
            "success": True,
            "user_id": user_id,
            "user_message": compare_req.message,
            "base_response": base_response.response,
            "personality_responses": {
                k.value: v for k, v in comparison.personality_responses.items()
            },
            "comparison_analysis": comparison.comparison_analysis,
            "recommendations": comparison.recommendations,
            "processing_time": processing_time
        }
        
    except PersonalityEngineError as e:
        logger.error(f"Personality comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Compare endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/user/{user_id}/memory")
async def get_user_memory_endpoint(user_id: str):
    """Get user's memory data"""
    try:
        memory = database.get_user_memory(user_id)
        if not memory:
            raise HTTPException(status_code=404, detail="User memory not found")
        
        security_logger.log_memory_access(
            user_id="api_request",
            requested_user=user_id,
            authorized=True
        )
        
        return {"success": True, "memory": memory}
        
    except DatabaseError as e:
        logger.error(f"Database error retrieving memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    
    except Exception as e:
        logger.error(f"Get memory endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory")


@app.get("/api/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics"""
    try:
        stats = database.get_user_stats(user_id)
        return {"success": True, "stats": stats}
        
    except Exception as e:
        logger.error(f"Get stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")


@app.get("/api/user/{user_id}/history")
async def get_conversation_history(user_id: str, limit: int = 10):
    """Get conversation history for user"""
    try:
        history = database.get_conversation_history(user_id, limit)
        return {"success": True, "history": history}
        
    except Exception as e:
        logger.error(f"Get history endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@app.get("/api/personalities")
async def get_available_personalities():
    """Get list of available personality types"""
    try:
        personalities = [
            {
                "type": personality.value,
                "name": personality_engine.personalities[personality].name,
                "description": personality_engine.personalities[personality].description,
                "characteristics": personality_engine.personalities[personality].characteristics.dict()
            }
            for personality in PersonalityType
        ]
        
        return {"success": True, "personalities": personalities}
        
    except Exception as e:
        logger.error(f"Get personalities endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve personalities")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    api_logger.log_error(
        method=request.method,
        endpoint=request.url.path,
        user_id=getattr(request.state, 'user_id', 'anonymous'),
        error=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    api_logger.log_error(
        method=request.method,
        endpoint=request.url.path,
        user_id=getattr(request.state, 'user_id', 'anonymous'),
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )
