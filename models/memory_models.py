"""
Memory Models and Data Structures
Production-ready data models for user memory extraction and management
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class PreferenceType(str, Enum):
    """Types of user preferences"""
    COMMUNICATION = "communication"
    CONTENT = "content"
    BEHAVIOR = "behavior"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    PROFESSIONAL = "professional"
    LIFESTYLE = "lifestyle"
    TECHNICAL = "technical"


class EmotionalPattern(BaseModel):
    """Structured emotional pattern data"""
    pattern: str = Field(..., description="Description of the emotional pattern")
    frequency: float = Field(ge=0, le=1, description="Frequency of this pattern (0-1)")
    triggers: List[str] = Field(default_factory=list, description="Common triggers for this pattern")
    intensity: str = Field(..., description="Intensity level (low, medium, high)")
    context: Optional[str] = Field(None, description="Context where this pattern appears")

    @validator('intensity')
    def validate_intensity(cls, v):
        allowed = ['low', 'medium', 'high']
        if v not in allowed:
            raise ValueError(f"Intensity must be one of {allowed}")
        return v


class UserPreference(BaseModel):
    """Structured user preference data"""
    category: PreferenceType = Field(..., description="Category of preference")
    value: str = Field(..., description="The preference value")
    confidence: float = Field(ge=0, le=1, description="Confidence score (0-1)")
    context: Optional[str] = Field(None, description="Context where this preference was observed")
    examples: List[str] = Field(default_factory=list, description="Examples from chat messages")


class UserFact(BaseModel):
    """Structured factual information about user"""
    fact_type: str = Field(..., description="Type of fact (personal, background, event, etc.)")
    value: str = Field(..., description="The factual information")
    confidence: float = Field(ge=0, le=1, description="Confidence score (0-1)")
    source_context: Optional[str] = Field(None, description="Context where this fact was mentioned")
    temporal_relevance: Optional[str] = Field(None, description="Whether this fact is time-sensitive")


class UserMemory(BaseModel):
    """Comprehensive user memory structure"""
    user_id: Optional[str] = Field(None, description="Unique user identifier")
    preferences: Dict[PreferenceType, List[UserPreference]] = Field(default_factory=dict)
    emotional_patterns: List[EmotionalPattern] = Field(default_factory=list)
    facts: List[UserFact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, description="Number of messages analyzed")
    analysis_version: str = Field(default="1.0", description="Version of analysis logic")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryExtractionRequest(BaseModel):
    """Request model for memory extraction"""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages with role and content")
    user_id: Optional[str] = Field(None, description="User identifier for memory persistence")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    

class MemoryExtractionResponse(BaseModel):
    """Response model for memory extraction"""
    success: bool = Field(..., description="Whether extraction was successful")
    memory: UserMemory = Field(..., description="Extracted memory data")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence in extraction")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during processing")
