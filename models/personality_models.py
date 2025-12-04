"""
Personality Engine Models
Production-ready models for AI personality transformation and response generation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum


class PersonalityType(str, Enum):
    """Supported personality types"""
    MENTOR = "mentor"
    FRIEND = "friend"
    THERAPIST = "therapist"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ENTHUSIASTIC = "enthusiastic"


class ToneCharacteristics(BaseModel):
    """Characteristics of personality tone"""
    formality: str = Field(..., description="Formality level (formal, semi-formal, casual)")
    empathy: str = Field(..., description="Empathy level (high, medium, low)")
    directness: str = Field(..., description="Directness level (high, medium, low)")
    creativity: str = Field(..., description="Creativity level (high, medium, low)")
    humor: str = Field(..., description="Humor level (high, medium, low)")
    supportiveness: str = Field(..., description="Supportiveness level (high, medium, low)")

    @validator('formality')
    def validate_formality(cls, v):
        allowed = ['formal', 'semi-formal', 'casual']
        if v not in allowed:
            raise ValueError(f"Formality must be one of {allowed}")
        return v

    @validator('empathy', 'directness', 'creativity', 'humor', 'supportiveness')
    def validate_levels(cls, v):
        allowed = ['high', 'medium', 'low']
        if v not in allowed:
            raise ValueError(f"Level must be one of {allowed}")
        return v


class PersonalityProfile(BaseModel):
    """Complete personality profile"""
    name: str = Field(..., description="Personality name")
    type: PersonalityType = Field(..., description="Personality type")
    description: str = Field(..., description="Brief description of personality")
    characteristics: ToneCharacteristics = Field(..., description="Tone characteristics")
    system_prompt: str = Field(..., description="Base system prompt for this personality")
    response_guidelines: List[str] = Field(default_factory=list, description="Specific response guidelines")
    vocabulary_preferences: List[str] = Field(default_factory=list, description="Preferred vocabulary")
    response_patterns: List[str] = Field(default_factory=list, description="Common response patterns")


class PersonalityTransformationRequest(BaseModel):
    """Request for personality transformation"""
    original_response: str = Field(..., description="Original AI response")
    target_personality: PersonalityType = Field(..., description="Target personality")
    user_memory: Optional[Dict[str, Any]] = Field(None, description="User memory context")
    conversation_context: Optional[List[Dict[str, str]]] = Field(None, description="Recent conversation context")
    user_message: Optional[str] = Field(None, description="Original user message")


class PersonalityTransformationResponse(BaseModel):
    """Response from personality transformation"""
    original_response: str = Field(..., description="Original response")
    transformed_response: str = Field(..., description="Personality-transformed response")
    personality_type: PersonalityType = Field(..., description="Applied personality")
    transformation_explanation: str = Field(..., description="Explanation of changes made")
    tone_analysis: Dict[str, str] = Field(..., description="Analysis of tone changes")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in transformation")


class PersonalityComparison(BaseModel):
    """Comparison between different personality responses"""
    user_message: str = Field(..., description="Original user message")
    base_response: str = Field(..., description="Base AI response without personality")
    personality_responses: Dict[PersonalityType, str] = Field(..., description="Responses by personality")
    comparison_analysis: Dict[str, Any] = Field(..., description="Analysis of differences")
    recommendations: List[str] = Field(default_factory=list, description="Usage recommendations")


class ResponseGenerationRequest(BaseModel):
    """Request for generating personality-based response"""
    user_message: str = Field(..., description="User message to respond to")
    personality: PersonalityType = Field(..., description="Personality to use")
    user_memory: Optional[Dict[str, Any]] = Field(None, description="User memory for personalization")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Recent conversation history")
    context: Optional[str] = Field(None, description="Additional context")


class ResponseGenerationResponse(BaseModel):
    """Response from personality-based generation"""
    response: str = Field(..., description="Generated response")
    personality: PersonalityType = Field(..., description="Personality used")
    personalization_elements: List[str] = Field(default_factory=list, description="Elements that were personalized")
    tone_metrics: Dict[str, Any] = Field(..., description="Tone analysis metrics")
    memory_references: List[str] = Field(default_factory=list, description="References to user memory")
    generation_confidence: float = Field(ge=0, le=1, description="Confidence in response quality")
