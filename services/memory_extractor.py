"""
Memory Extraction Service
Production-ready module for extracting structured user memory from chat messages
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.memory_models import (
    UserMemory, UserPreference, EmotionalPattern, UserFact, 
    PreferenceType, MemoryExtractionRequest
)
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Advanced memory extraction service with structured output parsing
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self._setup_system_prompts()
    
    def _setup_system_prompts(self):
        """Setup comprehensive system prompts for different extraction tasks"""
        
        self.memory_extraction_prompt = """You are an expert AI psychologist and data analyst specializing in extracting structured user memory from chat conversations.

Your task is to analyze the provided chat messages and extract three types of information:

1. USER PREFERENCES: What the user likes, dislikes, prefers, or tends to do
2. EMOTIONAL PATTERNS: How the user expresses emotions, common emotional states, triggers
3. FACTS: Personal information, background details, important events, or contextual information

For each extraction, provide:
- High confidence (0.8-1.0): Directly stated or clearly implied
- Medium confidence (0.5-0.8): Reasonably inferred but not explicit
- Low confidence (0.3-0.5): Possible but uncertain

Return your analysis in this exact JSON format:

{
    "preferences": [
        {
            "category": "communication|content|behavior|emotional|social|professional|lifestyle|technical",
            "value": "specific preference description",
            "confidence": 0.85,
            "context": "brief context from messages",
            "examples": ["relevant quote 1", "relevant quote 2"]
        }
    ],
    "emotional_patterns": [
        {
            "pattern": "description of emotional pattern",
            "frequency": 0.7,
            "triggers": ["trigger1", "trigger2"],
            "intensity": "low|medium|high",
            "context": "when this pattern appears"
        }
    ],
    "facts": [
        {
            "fact_type": "personal|background|event|relationship|work|health",
            "value": "factual information",
            "confidence": 0.9,
            "source_context": "where this was mentioned",
            "temporal_relevance": "current|past|ongoing"
        }
    ]
}

Guidelines:
- Be specific and precise in your extractions
- Include confidence scores for all items
- Provide context and examples when possible
- Focus on information that would be valuable for personalized interactions
- Avoid making assumptions beyond what's supported by the messages
- If uncertain, use lower confidence scores rather than omitting information"""

        self.message_analysis_prompt = """You are a conversation analyst. Your task is to analyze individual messages and identify key insights about the user.

For each message, identify:
1. Emotional state or tone
2. Any preferences expressed
3. Any factual information shared
4. Communication style indicators

Return a concise analysis focusing on actionable insights for building user understanding."""

    async def extract_memory(self, request: MemoryExtractionRequest) -> UserMemory:
        """
        Extract comprehensive user memory from chat messages
        
        Args:
            request: Memory extraction request with messages and metadata
            
        Returns:
            UserMemory: Structured memory data
        """
        try:
            logger.info(f"Starting memory extraction for {len(request.messages)} messages")
            
            # Prepare messages for analysis
            formatted_messages = self._format_messages(request.messages)
            
            # Extract memory using LLM
            extraction_result = await self._extract_with_llm(formatted_messages)
            
            # Parse and validate the extraction
            memory = self._parse_extraction_result(extraction_result, request.user_id)
            
            # Calculate confidence scores and statistics
            memory = self._enhance_memory_with_stats(memory, request.messages)
            
            logger.info(f"Successfully extracted memory: {len(memory.preferences)} preferences, "
                       f"{len(memory.emotional_patterns)} patterns, {len(memory.facts)} facts")
            
            return memory
            
        except Exception as e:
            logger.error(f"Error in memory extraction: {str(e)}")
            raise MemoryExtractionError(f"Failed to extract memory: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for LLM analysis"""
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '').strip()
            if content:
                formatted.append(f"[{i}] {role}: {content}")
        return "\n".join(formatted)
    
    async def _extract_with_llm(self, formatted_messages: str) -> Dict[str, Any]:
        """Extract memory information using LLM"""
        
        user_prompt = f"""Analyze these chat messages and extract user memory information:

{formatted_messages}

Focus on identifying patterns, preferences, and facts that would be useful for creating personalized AI responses. Provide your analysis in the specified JSON format."""
        
        response = await self.llm_client.generate_response(
            system_prompt=self.memory_extraction_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Clean and parse JSON response
        try:
            # Remove any markdown formatting
            cleaned_response = response.strip().strip('```json').strip('```').strip()
            extraction_data = json.loads(cleaned_response)
            return extraction_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response}")
            raise MemoryExtractionError(f"Invalid JSON response from LLM: {str(e)}")
    
    def _parse_extraction_result(self, extraction_data: Dict[str, Any], user_id: Optional[str]) -> UserMemory:
        """Parse and validate extraction results into UserMemory model"""
        
        try:
            # Parse preferences
            preferences_dict = {}
            for pref_data in extraction_data.get('preferences', []):
                try:
                    preference = UserPreference(**pref_data)
                    if preference.category not in preferences_dict:
                        preferences_dict[preference.category] = []
                    preferences_dict[preference.category].append(preference)
                except Exception as e:
                    logger.warning(f"Failed to parse preference: {pref_data}, error: {str(e)}")
            
            # Parse emotional patterns
            emotional_patterns = []
            for pattern_data in extraction_data.get('emotional_patterns', []):
                try:
                    pattern = EmotionalPattern(**pattern_data)
                    emotional_patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to parse emotional pattern: {pattern_data}, error: {str(e)}")
            
            # Parse facts
            facts = []
            for fact_data in extraction_data.get('facts', []):
                try:
                    fact = UserFact(**fact_data)
                    facts.append(fact)
                except Exception as e:
                    logger.warning(f"Failed to parse fact: {fact_data}, error: {str(e)}")

            # Determine message count in a robust way. The LLM may return either:
            # - an integer: "messages_analyzed": 4
            # - a list: "messages_analyzed": [ ... ]
            # - or omit the field entirely.
            raw_messages_analyzed = extraction_data.get('messages_analyzed')
            if isinstance(raw_messages_analyzed, int):
                message_count = raw_messages_analyzed
            elif isinstance(raw_messages_analyzed, list):
                message_count = len(raw_messages_analyzed)
            else:
                message_count = 0
            
            return UserMemory(
                user_id=user_id,
                preferences=preferences_dict,
                emotional_patterns=emotional_patterns,
                facts=facts,
                message_count=message_count,
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error parsing extraction result: {str(e)}")
            raise MemoryExtractionError(f"Failed to parse extraction results: {str(e)}")
    
    def _enhance_memory_with_stats(self, memory: UserMemory, messages: List[Dict[str, str]]) -> UserMemory:
        """Enhance memory with statistics and confidence scores"""
        
        # Calculate overall confidence scores
        all_confidences = []
        
        # Collect confidence scores from preferences
        for prefs in memory.preferences.values():
            all_confidences.extend([p.confidence for p in prefs])
        
        # Collect confidence scores from emotional patterns
        all_confidences.extend([p.frequency for p in memory.emotional_patterns])
        
        # Collect confidence scores from facts
        all_confidences.extend([f.confidence for f in memory.facts])
        
        # Calculate average confidence
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        # Add processing statistics
        processing_stats = {
            'total_messages': len(messages),
            'preferences_count': sum(len(prefs) for prefs in memory.preferences.values()),
            'emotional_patterns_count': len(memory.emotional_patterns),
            'facts_count': len(memory.facts),
            'overall_confidence': overall_confidence,
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        # Update memory with enhanced data
        memory.message_count = len(messages)
        memory.updated_at = datetime.utcnow()
        
        return memory


class MemoryExtractionError(Exception):
    """Custom exception for memory extraction errors"""
    pass
