"""
Tests for Personality Engine Service
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from models.personality_models import (
    PersonalityType, ResponseGenerationRequest, 
    PersonalityTransformationRequest, PersonalityComparison
)
from services.personality_engine import PersonalityEngine, PersonalityEngineError
from services.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def personality_engine(mock_llm_client):
    """Create personality engine with mock LLM client"""
    return PersonalityEngine(mock_llm_client)


@pytest.fixture
def sample_user_memory():
    """Sample user memory for testing"""
    return {
        "preferences": {
            "communication": "casual",
            "content": "technical topics"
        },
        "emotional_patterns": [
            "gets anxious with deadlines"
        ],
        "facts": {
            "profession": "software engineer",
            "location": "New York"
        }
    }


class TestPersonalityEngine:
    """Test cases for PersonalityEngine"""
    
    def test_initialization(self, personality_engine):
        """Test personality engine initialization"""
        assert len(personality_engine.personalities) == 7  # All personality types
        assert PersonalityType.MENTOR in personality_engine.personalities
        assert PersonalityType.FRIEND in personality_engine.personalities
    
    def test_personality_profiles_structure(self, personality_engine):
        """Test that all personality profiles have required structure"""
        for personality_type, profile in personality_engine.personalities.items():
            assert profile.name is not None
            assert profile.type == personality_type
            assert profile.description is not None
            assert profile.characteristics is not None
            assert profile.system_prompt is not None
            assert len(profile.response_guidelines) > 0
            assert len(profile.vocabulary_preferences) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_mentor(self, personality_engine, mock_llm_client):
        """Test generating mentor personality response"""
        # Setup mock
        mock_llm_client.generate_response.return_value = "Let me guide you through this step by step."
        
        request = ResponseGenerationRequest(
            user_message="I'm struggling with my career",
            personality=PersonalityType.MENTOR
        )
        
        result = await personality_engine.generate_response(request)
        
        assert result.response == "Let me guide you through this step by step."
        assert result.personality == PersonalityType.MENTOR
        assert result.generation_confidence > 0
        
        # Verify LLM was called with correct parameters
        mock_llm_client.generate_response.assert_called_once()
        call_args = mock_llm_client.generate_response.call_args
        assert call_args[1]['temperature'] == 0.5  # Mentor uses lower temperature
    
    @pytest.mark.asyncio
    async def test_generate_response_friend(self, personality_engine, mock_llm_client):
        """Test generating friend personality response"""
        # Setup mock
        mock_llm_client.generate_response.return_value = "OMG, totally get what you mean! Let's chat about it!"
        
        request = ResponseGenerationRequest(
            user_message="I'm having a bad day",
            personality=PersonalityType.FRIEND
        )
        
        result = await personality_engine.generate_response(request)
        
        assert result.response == "OMG, totally get what you mean! Let's chat about it!"
        assert result.personality == PersonalityType.FRIEND
        
        # Friend should use higher temperature
        call_args = mock_llm_client.generate_response.call_args
        assert call_args[1]['temperature'] == 0.7
    
    @pytest.mark.asyncio
    async def test_generate_response_with_memory(self, personality_engine, mock_llm_client, sample_user_memory):
        """Test generating response with user memory"""
        mock_llm_client.generate_response.return_value = "As a fellow software engineer, I understand the pressure."
        
        request = ResponseGenerationRequest(
            user_message="Work is stressful",
            personality=PersonalityType.PROFESSIONAL,
            user_memory=sample_user_memory
        )
        
        result = await personality_engine.generate_response(request)
        
        assert result.response == "As a fellow software engineer, I understand the pressure."
        assert len(result.memory_references) > 0
    
    @pytest.mark.asyncio
    async def test_transform_response(self, personality_engine, mock_llm_client):
        """Test response transformation between personalities"""
        mock_llm_client.generate_response.return_value = "Let me provide you with some structured guidance."
        
        request = PersonalityTransformationRequest(
            original_response="hey what's up",
            target_personality=PersonalityType.MENTOR,
            user_message="need advice"
        )
        
        result = await personality_engine.transform_response(request)
        
        assert result.original_response == "hey what's up"
        assert result.transformed_response == "Let me provide you with some structured guidance."
        assert result.personality_type == PersonalityType.MENTOR
        assert result.transformation_explanation is not None
        assert result.tone_analysis is not None
    
    @pytest.mark.asyncio
    async def test_compare_personalities(self, personality_engine, mock_llm_client):
        """Test comparing responses across personalities"""
        # Setup mock for different responses
        def mock_generate_side_effect(system_prompt, user_prompt, **kwargs):
            if "mentor" in system_prompt.lower():
                return "Let me guide you through this process."
            elif "friend" in system_prompt.lower():
                return "OMG, I can totally help you with that!"
            else:
                return "I will provide professional assistance."
        
        mock_llm_client.generate_response.side_effect = mock_generate_side_effect
        
        comparison = await personality_engine.compare_personalities(
            user_message="I need help with a project",
            base_response="I will help you with your project.",
            personalities=[PersonalityType.MENTOR, PersonalityType.FRIEND]
        )
        
        assert comparison.user_message == "I need help with a project"
        assert comparison.base_response in comparison.personality_responses

        assert len(comparison.personality_responses) >= 1
        assert comparison.comparison_analysis is not None
        assert len(comparison.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_llm_error(self, personality_engine, mock_llm_client):
        """Test response generation when LLM fails"""
        mock_llm_client.generate_response.side_effect = Exception("LLM API error")
        
        request = ResponseGenerationRequest(
            user_message="test message",
            personality=PersonalityType.PROFESSIONAL
        )
        
        with pytest.raises(PersonalityEngineError):
            await personality_engine.generate_response(request)
    
    def test_build_context_prompt(self, personality_engine):
        """Test building context-aware prompts"""
        profile = personality_engine.personalities[PersonalityType.MENTOR]
        
        prompt = personality_engine._build_context_prompt(
            user_message="I need career advice",
            profile=profile,
            user_memory={"preferences": {"communication": "formal"}},
            conversation_history=[{"role": "user", "content": "Hi"}],
            context="User is looking for job change"
        )
        
        assert "I need career advice" in prompt
        assert "User is looking for job change" in prompt
        assert "preferences" in prompt.lower()
    
    def test_format_memory_context(self, personality_engine, sample_user_memory):
        """Test memory context formatting"""
        context = personality_engine._format_memory_context(sample_user_memory)
        
        assert "software engineer" in context
        assert "New York" in context
        assert "communication" in context
    
    def test_format_conversation_context(self, personality_engine):
        """Test conversation context formatting"""
        conversation = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "I need some advice"}
        ]
        
        context = personality_engine._format_conversation_context(conversation)
        
        assert "user: Hello, how are you?" in context
        assert "assistant: I'm doing well" in context
    
    def test_extract_personalization_elements(self, personality_engine):
        """Test extraction of personalization elements"""
        user_memory = {
            "preferences": ["coffee lover", "tech enthusiast"],
            "facts": {"profession": "developer"}
        }
        
        response = "As a fellow tech enthusiast, I understand your love for coffee."
        
        elements = personality_engine._extract_personalization_elements(response, user_memory)
        
        assert len(elements) > 0
        assert any("tech enthusiast" in element for element in elements)
    
    def test_extract_memory_references(self, personality_engine):
        """Test extraction of memory references"""
        user_memory = {
            "facts": [
                {"value": "software engineer", "confidence": 0.9},
                {"value": "lives in New York", "confidence": 0.8}
            ]
        }
        
        response = "As a software engineer living in New York, I understand the challenges."
        
        references = personality_engine._extract_memory_references(response, user_memory)
        
        assert len(references) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
