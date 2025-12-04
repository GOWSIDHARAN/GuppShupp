"""
Tests for Memory Extractor Service
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from models.memory_models import MemoryExtractionRequest, UserMemory
from services.memory_extractor import MemoryExtractor, MemoryExtractionError
from services.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def memory_extractor(mock_llm_client):
    """Create memory extractor with mock LLM client"""
    return MemoryExtractor(mock_llm_client)


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing"""
    return [
        {"role": "user", "content": "I love coffee in the morning"},
        {"role": "assistant", "content": "That's great! Coffee is popular"},
        {"role": "user", "content": "I get anxious when I have deadlines"},
        {"role": "assistant", "content": "I understand that feeling"},
        {"role": "user", "content": "I work as a software engineer in New York"}
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for memory extraction"""
    return {
        "preferences": [
            {
                "category": "lifestyle",
                "value": "enjoys coffee in the morning",
                "confidence": 0.9,
                "context": "mentioned in first message",
                "examples": ["I love coffee in the morning"]
            }
        ],
        "emotional_patterns": [
            {
                "pattern": "gets anxious with deadlines",
                "frequency": 0.7,
                "triggers": ["deadlines", "pressure"],
                "intensity": "medium",
                "context": "when discussing work pressure"
            }
        ],
        "facts": [
            {
                "fact_type": "professional",
                "value": "software engineer",
                "confidence": 0.95,
                "source_context": "mentioned in last message",
                "temporal_relevance": "current"
            },
            {
                "fact_type": "location",
                "value": "lives in New York",
                "confidence": 0.9,
                "source_context": "mentioned with job",
                "temporal_relevance": "current"
            }
        ]
    }


class TestMemoryExtractor:
    """Test cases for MemoryExtractor"""
    
    @pytest.mark.asyncio
    async def test_extract_memory_success(self, memory_extractor, mock_llm_client, sample_messages, mock_llm_response):
        """Test successful memory extraction"""
        # Setup mock
        mock_llm_client.generate_response.return_value = '{"preferences": [], "emotional_patterns": [], "facts": []}'
        
        # Create request
        request = MemoryExtractionRequest(
            messages=sample_messages,
            user_id="test_user_123",
            analysis_depth="comprehensive"
        )
        
        # Execute
        result = await memory_extractor.extract_memory(request)
        
        # Assertions
        assert isinstance(result, UserMemory)
        assert result.user_id == "test_user_123"
        assert result.message_count == len(sample_messages)
        
        # Verify LLM was called
        mock_llm_client.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_memory_with_empty_messages(self, memory_extractor):
        """Test memory extraction with empty messages"""
        request = MemoryExtractionRequest(
            messages=[],
            user_id="test_user_123"
        )
        
        result = await memory_extractor.extract_memory(request)
        
        assert isinstance(result, UserMemory)
        assert result.message_count == 0
        assert len(result.preferences) == 0
        assert len(result.emotional_patterns) == 0
        assert len(result.facts) == 0
    
    @pytest.mark.asyncio
    async def test_extract_memory_llm_error(self, memory_extractor, mock_llm_client, sample_messages):
        """Test memory extraction when LLM fails"""
        # Setup mock to raise exception
        mock_llm_client.generate_response.side_effect = Exception("LLM API error")
        
        request = MemoryExtractionRequest(
            messages=sample_messages,
            user_id="test_user_123"
        )
        
        # Should raise MemoryExtractionError
        with pytest.raises(MemoryExtractionError):
            await memory_extractor.extract_memory(request)
    
    @pytest.mark.asyncio
    async def test_extract_memory_invalid_json(self, memory_extractor, mock_llm_client, sample_messages):
        """Test memory extraction with invalid JSON response"""
        # Setup mock to return invalid JSON
        mock_llm_client.generate_response.return_value = "This is not valid JSON"
        
        request = MemoryExtractionRequest(
            messages=sample_messages,
            user_id="test_user_123"
        )
        
        # Should raise MemoryExtractionError
        with pytest.raises(MemoryExtractionError):
            await memory_extractor.extract_memory(request)
    
    def test_format_messages(self, memory_extractor, sample_messages):
        """Test message formatting"""
        formatted = memory_extractor._format_messages(sample_messages)
        
        assert "[1] USER: I love coffee in the morning" in formatted
        assert "[2] ASSISTANT: That's great! Coffee is popular" in formatted
        assert "[3] USER: I get anxious when I have deadlines" in formatted
    
    @pytest.mark.asyncio
    async def test_parse_extraction_result(self, memory_extractor, mock_llm_response):
        """Test parsing extraction results"""
        result = memory_extractor._parse_extraction_result(mock_llm_response, "test_user")
        
        assert isinstance(result, UserMemory)
        assert result.user_id == "test_user"
        assert len(result.preferences) > 0
        assert len(result.emotional_patterns) > 0
        assert len(result.facts) > 0
    
    def test_enhance_memory_with_stats(self, memory_extractor):
        """Test memory enhancement with statistics"""
        from models.memory_models import UserMemory, UserPreference, EmotionalPattern, UserFact
        
        # Create basic memory
        memory = UserMemory(
            preferences={"lifestyle": [UserPreference(
                category="lifestyle",
                value="enjoys coffee",
                confidence=0.9
            )]},
            emotional_patterns=[EmotionalPattern(
                pattern="anxious with deadlines",
                frequency=0.7,
                triggers=["deadlines"],
                intensity="medium"
            )],
            facts=[UserFact(
                fact_type="professional",
                value="software engineer",
                confidence=0.95
            )]
        )
        
        messages = [{"role": "user", "content": "test"}]
        
        enhanced = memory_extractor._enhance_memory_with_stats(memory, messages)
        
        assert enhanced.message_count == 1
        assert enhanced.updated_at is not None


if __name__ == "__main__":
    pytest.main([__file__])
