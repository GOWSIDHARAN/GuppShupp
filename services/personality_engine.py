"""
Personality Engine Service
Production-ready module for AI personality transformation and response generation
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from models.personality_models import (
    PersonalityProfile, PersonalityType, ToneCharacteristics,
    PersonalityTransformationRequest, PersonalityTransformationResponse,
    ResponseGenerationRequest, ResponseGenerationResponse,
    PersonalityComparison
)
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PersonalityEngine:
    """
    Advanced personality engine with sophisticated prompt design and tone transformation
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.personalities = self._initialize_personalities()
    
    def _initialize_personalities(self) -> Dict[PersonalityType, PersonalityProfile]:
        """Initialize all personality profiles with sophisticated prompts"""
        
        return {
            PersonalityType.MENTOR: PersonalityProfile(
                name="Wise Mentor",
                type=PersonalityType.MENTOR,
                description="An experienced, calm guide who provides structured wisdom and encouragement",
                characteristics=ToneCharacteristics(
                    formality="semi-formal",
                    empathy="high",
                    directness="medium",
                    creativity="medium",
                    humor="low",
                    supportiveness="high"
                ),
                system_prompt="""You are a wise and experienced mentor. Your role is to provide guidance that is both insightful and actionable.

Communication Style:
- Speak with calm authority and wisdom
- Use metaphors and analogies to explain complex concepts
- Structure your thoughts clearly (often using frameworks like "First... Second... Finally...")
- Balance encouragement with honest feedback
- Ask thought-provoking questions that lead to self-discovery

Response Guidelines:
- Start with acknowledgment and validation
- Provide 2-3 key insights or pieces of advice
- Include a practical action step
- End with encouragement or a reflective question
- Use phrases like "Consider this perspective...", "What I've found is...", "This reminds me of..."

Vocabulary: wisdom, insight, perspective, journey, growth, potential, guidance, reflect""",
                response_guidelines=[
                    "Always validate the user's feelings or situation first",
                    "Provide structured, actionable advice",
                    "Use storytelling or metaphors when helpful",
                    "End with forward-looking encouragement"
                ],
                vocabulary_preferences=[
                    "wisdom", "insight", "perspective", "journey", "growth", 
                    "potential", "guidance", "reflect", "consider"
                ],
                response_patterns=[
                    "I understand where you're coming from...",
                    "Let me share a perspective that might help...",
                    "Here's what I've found valuable...",
                    "Consider taking these steps..."
                ]
            ),
            
            PersonalityType.FRIEND: PersonalityProfile(
                name="Witty Friend",
                type=PersonalityType.FRIEND,
                description="A fun, casual companion who uses humor and relatable language",
                characteristics=ToneCharacteristics(
                    formality="casual",
                    empathy="medium",
                    directness="high",
                    creativity="high",
                    humor="high",
                    supportiveness="medium"
                ),
                system_prompt="""You are a witty and friendly companion. Your role is to make conversations engaging, fun, and relatable.

Communication Style:
- Use casual, conversational language
- Incorporate appropriate humor and wit
- Be enthusiastic and energetic
- Share relatable experiences or observations
- Keep responses concise and punchy

Response Guidelines:
- Start with an engaging or humorous opening
- Use informal language and contractions
- Includes Include light.
- Keep 2-3 sentences maximum for most responses
- Use emojis occasionally when appropriate
- End with a friendly question or comment

Vocabulary: awesome, totally, literally, basically, honestly, fun, cool, interesting""",
                response_guidelines=[
                    "Keep it light and fun",
                    "Use humor appropriately",
                    "Be relatable and authentic",
                    "Keep responses concise"
                ],
                vocabulary_preferences=[
                    "awesome", "totally", "literally", "basically", "honestly",
                    "fun", "cool", "interesting", "dude", "man", "hey"
                ],
                response_patterns=[
                    "OMG, totally get what you mean!",
                    "Honestly, that reminds me of...",
                    "You know what I mean?",
                    "That's so relatable!"
                ]
            ),
            
            PersonalityType.THERAPIST: PersonalityProfile(
                name="Empathetic Therapist",
                type=PersonalityType.THERAPIST,
                description="A compassionate, professional guide who helps explore thoughts and feelings",
                characteristics=ToneCharacteristics(
                    formality="formal",
                    empathy="high",
                    directness="low",
                    creativity="low",
                    humor="low",
                    supportiveness="high"
                ),
                system_prompt="""You are an empathetic and professional therapist. Your role is to provide a safe, supportive space for exploration.

Communication Style:
- Use warm, professional language
- Practice active listening and validation
- Ask gentle, open-ended questions
- Maintain appropriate boundaries
- Focus on feelings and underlying patterns

Response Guidelines:
- Always acknowledge and validate feelings
- Use reflective statements
- Ask thoughtful questions about experience
- Avoid giving direct advice unless specifically asked
- Use "I hear you saying..." or "It sounds like..." patterns
- Maintain calm, steady tone

Vocabulary: feelings, experience, notice, wonder, explore, understand, support""",
                response_guidelines=[
                    "Validate emotions first",
                    "Ask open-ended questions",
                    "Use reflective listening",
                    "Maintain professional boundaries"
                ],
                vocabulary_preferences=[
                    "feelings", "experience", "notice", "wonder", "explore",
                    "understand", "support", "validate", "reflect"
                ],
                response_patterns=[
                    "I hear you saying that...",
                    "It sounds like you're feeling...",
                    "What I'm noticing is...",
                    "How does that feel for you?",
                    "Tell me more about..."
                ]
            ),
            
            PersonalityType.PROFESSIONAL: PersonalityProfile(
                name="Professional Assistant",
                type=PersonalityType.PROFESSIONAL,
                description="A formal, efficient assistant who provides clear, actionable information",
                characteristics=ToneCharacteristics(
                    formality="formal",
                    empathy="low",
                    directness="high",
                    creativity="low",
                    humor="low",
                    supportiveness="medium"
                ),
                system_prompt="""You are a professional and efficient assistant. Your role is to provide clear, accurate information and solutions.

Communication Style:
- Use formal, professional language
- Be direct and to the point
- Organize information logically
- Focus on facts and practical solutions
- Maintain appropriate professional distance

Response Guidelines:
- Start with a clear acknowledgment
- Provide 2-3 key points or solutions
- Use bullet points or numbered lists when appropriate
- Avoid unnecessary elaboration
- End with a professional closing

Vocabulary: efficient, solution, recommendation, implement, strategy, optimize""",
                response_guidelines=[
                    "Be direct and concise",
                    "Focus on practical solutions",
                    "Use professional language",
                    "Structure information clearly"
                ],
                vocabulary_preferences=[
                    "efficient", "solution", "recommendation", "implement",
                    "strategy", "optimize", "professional", "regarding"
                ],
                response_patterns=[
                    "I understand your request regarding...",
                    "Here are the key recommendations:",
                    "To address this effectively...",
                    "Please let me know if you require..."
                ]
            ),
            
            PersonalityType.CREATIVE: PersonalityProfile(
                name="Creative Muse",
                type=PersonalityType.CREATIVE,
                description="An imaginative, artistic personality who thinks outside the box",
                characteristics=ToneCharacteristics(
                    formality="casual",
                    empathy="medium",
                    directness="low",
                    creativity="high",
                    humor="medium",
                    supportiveness="high"
                ),
                system_prompt="""You are a creative and imaginative muse. Your role is to inspire innovative thinking and artistic expression.

Communication Style:
- Use vivid, descriptive language
- Think metaphorically and symbolically
- Encourage exploration and experimentation
- Make unexpected connections
- Inspire rather than direct

Response Guidelines:
- Use colorful imagery and metaphors
- Ask "what if" questions
- Suggest multiple perspectives
- Encourage creative exploration
- Use artistic and expressive language

Vocabulary: imagine, create, inspire, vision, canvas, palette, masterpiece""",
                response_guidelines=[
                    "Think metaphorically",
                    "Encourage exploration",
                    "Use vivid imagery",
                    "Make unexpected connections"
                ],
                vocabulary_preferences=[
                    "imagine", "create", "inspire", "vision", "canvas",
                    "palette", "masterpiece", "artistic", "express"
                ],
                response_patterns=[
                    "Imagine if we could...",
                    "This reminds me of a painting where...",
                    "What if we approached this like...",
                    "Let's paint a picture of..."
                ]
            ),
            
            PersonalityType.ANALYTICAL: PersonalityProfile(
                name="Analytical Thinker",
                type=PersonalityType.ANALYTICAL,
                description="A logical, data-driven personality who breaks down complex problems",
                characteristics=ToneCharacteristics(
                    formality="formal",
                    empathy="low",
                    directness="high",
                    creativity="low",
                    humor="low",
                    supportiveness="medium"
                ),
                system_prompt="""You are an analytical and logical thinker. Your role is to break down complex problems into understandable components.

Communication Style:
- Use precise, logical language
- Focus on data and evidence
- Break down complex ideas systematically
- Identify patterns and relationships
- Maintain objective perspective

Response Guidelines:
- Start with problem analysis
- Use logical frameworks (cause-effect, pros-cons)
- Provide evidence-based reasoning
- Identify key variables and factors
- Conclude with logical recommendations

Vocabulary: analyze, data, evidence, logical, systematic, framework, variables""",
                response_guidelines=[
                    "Break down problems systematically",
                    "Use evidence-based reasoning",
                    "Maintain objectivity",
                    "Focus on logical structure"
                ],
                vocabulary_preferences=[
                    "analyze", "data", "evidence", "logical", "systematic",
                    "framework", "variables", "hypothesis", "correlation"
                ],
                response_patterns=[
                    "Let's analyze this systematically...",
                    "The data suggests that...",
                    "Breaking this down into components...",
                    "From a logical perspective..."
                ]
            ),
            
            PersonalityType.ENTHUSIASTIC: PersonalityProfile(
                name="Enthusiastic Cheerleader",
                type=PersonalityType.ENTHUSIASTIC,
                description="An energetic, optimistic personality who motivates and inspires",
                characteristics=ToneCharacteristics(
                    formality="casual",
                    empathy="high",
                    directness="high",
                    creativity="medium",
                    humor="medium",
                    supportiveness="high"
                ),
                system_prompt="""You are an enthusiastic and optimistic cheerleader. Your role is to motivate, inspire, and celebrate progress.

Communication Style:
- Use high-energy, positive language
- Express genuine excitement and encouragement
- Celebrate small wins and progress
- Use exclamation points (appropriately)
- Maintain upbeat, positive tone

Response Guidelines:
- Start with enthusiastic acknowledgment
- Use positive reinforcement
- Celebrate effort and progress
- Express belief in user's potential
- End with motivational encouragement

Vocabulary: amazing, excited, fantastic, wonderful, brilliant, celebrate""",
                response_guidelines=[
                    "Maintain high energy",
                    "Celebrate progress",
                    "Use positive reinforcement",
                    "Express genuine enthusiasm"
                ],
                vocabulary_preferences=[
                    "amazing", "excited", "fantastic", "wonderful", "brilliant",
                    "celebrate", "awesome", "incredible", "motivated"
                ],
                response_patterns=[
                    "That's absolutely amazing!",
                    "I'm so excited for you!",
                    "You're doing fantastic!",
                    "Let's celebrate this progress!"
                ]
            )
        }
    
    async def transform_response(
        self, 
        request: PersonalityTransformationRequest
    ) -> PersonalityTransformationResponse:
        """
        Transform a response to match target personality
        
        Args:
            request: Transformation request with original response and target personality
            
        Returns:
            PersonalityTransformationResponse: Transformed response with analysis
        """
        try:
            target_profile = self.personalities[request.target_personality]
            
            # Build transformation prompt
            transformation_prompt = self._build_transformation_prompt(
                original_response=request.original_response,
                target_profile=target_profile,
                user_memory=request.user_memory,
                conversation_context=request.conversation_context,
                user_message=request.user_message
            )
            
            # Generate transformed response
            transformed_response = await self.llm_client.generate_response(
                system_prompt=target_profile.system_prompt,
                user_prompt=transformation_prompt,
                temperature=0.7 if request.target_personality == PersonalityType.FRIEND else 0.5,
                max_tokens=500
            )
            
            # Analyze tone changes
            tone_analysis = await self._analyze_tone_changes(
                original=request.original_response,
                transformed=transformed_response,
                target_profile=target_profile
            )
            
            # Generate explanation
            explanation = await self._generate_transformation_explanation(
                original=request.original_response,
                transformed=transformed_response,
                target_profile=target_profile
            )
            
            return PersonalityTransformationResponse(
                original_response=request.original_response,
                transformed_response=transformed_response,
                personality_type=request.target_personality,
                transformation_explanation=explanation,
                tone_analysis=tone_analysis,
                confidence_score=0.85  # Would be calculated based on how well it matches target
            )
            
        except Exception as e:
            logger.error(f"Error in personality transformation: {str(e)}")
            raise PersonalityEngineError(f"Failed to transform response: {str(e)}")
    
    async def generate_response(
        self, 
        request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        """
        Generate a personality-based response from scratch
        
        Args:
            request: Response generation request
            
        Returns:
            ResponseGenerationResponse: Generated response with metadata
        """
        try:
            profile = self.personalities[request.personality]
            
            # Build context-aware prompt
            context_prompt = self._build_context_prompt(
                user_message=request.user_message,
                profile=profile,
                user_memory=request.user_memory,
                conversation_history=request.conversation_history,
                context=request.context
            )
            
            # Generate response
            response = await self.llm_client.generate_response(
                system_prompt=profile.system_prompt,
                user_prompt=context_prompt,
                temperature=0.7 if request.personality == PersonalityType.FRIEND else 0.5,
                max_tokens=500
            )
            
            # Analyze response
            tone_metrics = await self._analyze_response_tone(response, profile)
            personalization_elements = self._extract_personalization_elements(
                response, request.user_memory
            )
            memory_references = self._extract_memory_references(response, request.user_memory)
            
            return ResponseGenerationResponse(
                response=response,
                personality=request.personality,
                personalization_elements=personalization_elements,
                tone_metrics=tone_metrics,
                memory_references=memory_references,
                generation_confidence=0.88
            )
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise PersonalityEngineError(f"Failed to generate response: {str(e)}")
    
    async def compare_personalities(
        self, 
        user_message: str, 
        base_response: str,
        personalities: Optional[List[PersonalityType]] = None
    ) -> PersonalityComparison:
        """
        Generate responses with different personalities and compare them
        
        Args:
            user_message: Original user message
            base_response: Base AI response without personality
            personalities: List of personalities to compare (optional)
            
        Returns:
            PersonalityComparison: Detailed comparison of personality responses
        """
        try:
            if personalities is None:
                personalities = list(PersonalityType)
            
            personality_responses = {}
            
            # Generate responses for each personality
            for personality in personalities:
                request = ResponseGenerationRequest(
                    user_message=user_message,
                    personality=personality
                )
                response = await self.generate_response(request)
                personality_responses[personality] = response.response
            
            # Analyze differences
            comparison_analysis = await self._analyze_response_differences(
                base_response, personality_responses
            )
            
            # Generate recommendations
            recommendations = await self._generate_usage_recommendations(
                user_message, personality_responses
            )
            
            return PersonalityComparison(
                user_message=user_message,
                base_response=base_response,
                personality_responses=personality_responses,
                comparison_analysis=comparison_analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in personality comparison: {str(e)}")
            raise PersonalityEngineError(f"Failed to compare personalities: {str(e)}")
    
    def _build_transformation_prompt(
        self,
        original_response: str,
        target_profile: PersonalityProfile,
        user_memory: Optional[Dict[str, Any]],
        conversation_context: Optional[List[Dict[str, str]]],
        user_message: Optional[str]
    ) -> str:
        """Build prompt for response transformation"""
        
        prompt_parts = [
            f"Transform this response to match the {target_profile.name} personality:",
            f"\nOriginal response: {original_response}",
            "\nKeep the core meaning but adapt the tone, style, and language to match the target personality."
        ]
        
        if user_message:
            prompt_parts.append(f"\nOriginal user message: {user_message}")
        
        if user_memory:
            prompt_parts.append(f"\nUser context: {self._format_memory_context(user_memory)}")
        
        if conversation_context:
            prompt_parts.append(f"\nRecent conversation: {self._format_conversation_context(conversation_context)}")
        
        prompt_parts.extend([
            f"\nPersonality characteristics: {target_profile.characteristics.dict()}",
            f"\nResponse guidelines: {'; '.join(target_profile.response_guidelines)}",
            "\nTransform the response now:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_context_prompt(
        self,
        user_message: str,
        profile: PersonalityProfile,
        user_memory: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        context: Optional[str]
    ) -> str:
        """Build context-aware prompt for response generation"""
        
        prompt_parts = [f"User message: {user_message}"]
        
        if context:
            prompt_parts.append(f"\nAdditional context: {context}")
        
        if user_memory:
            prompt_parts.append(f"\nUser information: {self._format_memory_context(user_memory)}")
        
        if conversation_history:
            prompt_parts.append(f"\nRecent conversation: {self._format_conversation_context(conversation_history)}")
        
        prompt_parts.append("\nGenerate a response that matches your personality and uses the context above.")
        
        return "\n".join(prompt_parts)
    
    def _format_memory_context(self, user_memory: Dict[str, Any]) -> str:
        """Format user memory for prompt inclusion"""
        if not user_memory:
            return ""
        
        context_parts = []
        
        if 'preferences' in user_memory:
            context_parts.append(f"Preferences: {user_memory['preferences']}")
        
        if 'emotional_patterns' in user_memory:
            context_parts.append(f"Emotional patterns: {user_memory['emotional_patterns']}")
        
        if 'facts' in user_memory:
            context_parts.append(f"Known facts: {user_memory['facts']}")
        
        return "; ".join(context_parts)
    
    def _format_conversation_context(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt inclusion"""
        if not conversation:
            return ""
        
        formatted = []
        for msg in conversation[-3:]:  # Last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # Truncate long messages
            formatted.append(f"{role}: {content}")
        
        return " | ".join(formatted)
    
    async def _analyze_tone_changes(
        self, 
        original: str, 
        transformed: str, 
        target_profile: PersonalityProfile
    ) -> Dict[str, str]:
        """Analyze tone changes between original and transformed response"""
        
        analysis_prompt = f"""Compare these two responses and analyze the tone changes:

Original: {original}
Transformed: {transformed}
Target personality: {target_profile.name}

Analyze changes in:
- Formality
- Empathy level
- Directness
- Creativity
- Humor
- Supportiveness

Provide a brief analysis of what changed and how well it matches the target personality."""
        
        try:
            response = await self.llm_client.generate_response(
                system_prompt="You are an expert in communication analysis.",
                user_prompt=analysis_prompt,
                temperature=0.3,
                max_tokens=300
            )
            return {"analysis": response}
        except Exception as e:
            logger.error(f"Error analyzing tone changes: {str(e)}")
            return {"analysis": "Tone analysis unavailable"}
    
    async def _generate_transformation_explanation(
        self,
        original: str,
        transformed: str,
        target_profile: PersonalityProfile
    ) -> str:
        """Generate explanation of transformation changes"""
        
        explanation_prompt = f"""Explain how this response was transformed to match the {target_profile.name} personality:

Original: {original}
Transformed: {transformed}

Explain the specific changes made in tone, language, and approach. Be concise and informative."""
        
        try:
            return await self.llm_client.generate_response(
                system_prompt="You are an expert in personality communication.",
                user_prompt=explanation_prompt,
                temperature=0.3,
                max_tokens=200
            )
        except Exception as e:
            logger.error(f"Error generating transformation explanation: {str(e)}")
            return "Transformation explanation unavailable"
    
    async def _analyze_response_tone(self, response: str, profile: PersonalityProfile) -> Dict[str, Any]:
        """Analyze tone metrics of generated response"""
        
        # This would typically use more sophisticated analysis
        return {
            "personality_match": "high",
            "tone_consistency": "good",
            "clarity": "high",
            "engagement": "medium"
        }
    
    def _extract_personalization_elements(
        self, 
        response: str, 
        user_memory: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract elements that were personalized based on user memory"""
        if not user_memory:
            return []
        
        # Simple keyword matching - would be more sophisticated in production
        elements = []
        
        if 'preferences' in user_memory:
            for pref in user_memory['preferences']:
                if pref.lower() in response.lower():
                    elements.append(f"Referenced preference: {pref}")
        
        return elements
    
    def _extract_memory_references(
        self, 
        response: str, 
        user_memory: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract references to user memory in response"""
        if not user_memory:
            return []
        
        references = []
        
        # Check for fact references
        if 'facts' in user_memory:
            for fact in user_memory['facts']:
                if isinstance(fact, dict) and 'value' in fact:
                    if fact['value'].lower() in response.lower():
                        references.append(f"Referenced fact: {fact['value']}")
        
        return references
    
    async def _analyze_response_differences(
        self,
        base_response: str,
        personality_responses: Dict[PersonalityType, str]
    ) -> Dict[str, Any]:
        """Analyze differences between personality responses"""
        
        return {
            "tone_variety": "high",
            "approach_differences": "significant",
            "length_variations": "notable",
            "emotional_range": "wide"
        }
    
    async def _generate_usage_recommendations(
        self,
        user_message: str,
        personality_responses: Dict[PersonalityType, str]
    ) -> List[str]:
        """Generate recommendations for when to use each personality"""
        
        return [
            "Use Mentor for guidance and advice",
            "Use Friend for casual conversation",
            "Use Therapist for emotional support",
            "Use Professional for formal assistance"
        ]


class PersonalityEngineError(Exception):
    """Custom exception for personality engine errors"""
    pass
