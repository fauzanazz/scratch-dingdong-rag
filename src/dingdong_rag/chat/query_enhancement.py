import time
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import openai

from .models import ChatMessage, ConversationContext


@dataclass
class QueryEnhancementConfig:
    """Configuration for query enhancement."""
    # LLM settings for rephrasing
    rephrasing_model: str = "gpt-4o-mini"
    rephrasing_temperature: float = 0.3
    rephrasing_max_tokens: int = 200
    
    # Enhancement strategies
    enable_rephrasing: bool = True
    enable_expansion: bool = True
    enable_context_awareness: bool = True
    enable_domain_adaptation: bool = True
    
    # Validation settings
    min_query_length: int = 10
    max_query_length: int = 1000
    forbidden_patterns: List[str] = None
    
    # Domain-specific settings
    domain_keywords: Set[str] = None
    technical_terms: Set[str] = None
    
    def __post_init__(self):
        if self.forbidden_patterns is None:
            self.forbidden_patterns = [
                r'\b(?:hack|exploit|bypass|crack)\b',  # Security-related
                r'\b(?:sql injection|xss|malware)\b',  # Attack patterns
                r'\b(?:illegal|unlawful|prohibited)\b'  # Legal issues
            ]
        
        if self.domain_keywords is None:
            self.domain_keywords = set()
        
        if self.technical_terms is None:
            self.technical_terms = set()


@dataclass
class QueryValidationResult:
    """Result of query validation."""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]


@dataclass
class QueryEnhancementResult:
    """Result of query enhancement/rephrasing."""
    original_query: str
    enhanced_query: str
    enhancement_type: str
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class QueryValidator:
    """Validates user queries for quality and appropriateness."""
    
    def __init__(self, config: QueryEnhancementConfig):
        self.config = config
    
    def validate_query(self, query: str) -> QueryValidationResult:
        """Validate query for basic quality checks."""
        issues = []
        suggestions = []
        confidence_score = 1.0
        
        # Length checks
        if len(query.strip()) < self.config.min_query_length:
            issues.append("Query too short")
            suggestions.append(f"Please provide at least {self.config.min_query_length} characters")
            confidence_score -= 0.3
        
        if len(query.strip()) > self.config.max_query_length:
            issues.append("Query too long")
            suggestions.append(f"Please limit query to {self.config.max_query_length} characters")
            confidence_score -= 0.2
        
        # Content quality checks
        if not query.strip():
            issues.append("Empty query")
            suggestions.append("Please provide a question or topic to search for")
            confidence_score = 0.0
        
        # Check for forbidden patterns
        for pattern in self.config.forbidden_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append("Query contains potentially problematic content")
                suggestions.append("Please rephrase your query appropriately")
                confidence_score -= 0.5
        
        # Basic linguistic quality
        word_count = len(query.split())
        if word_count == 1 and len(query) > 20:
            issues.append("Query appears to be a single long word")
            suggestions.append("Consider using multiple words to describe your question")
            confidence_score -= 0.2
        
        # Check for question structure
        if not self._has_question_structure(query):
            suggestions.append("Consider phrasing as a question for better results")
            confidence_score -= 0.1
        
        is_valid = len(issues) == 0 and confidence_score > 0.3
        
        return QueryValidationResult(
            is_valid=is_valid,
            confidence_score=max(0.0, confidence_score),
            issues=issues,
            suggestions=suggestions,
            metadata={
                "word_count": word_count,
                "character_count": len(query),
                "has_question_mark": "?" in query,
                "has_question_words": self._count_question_words(query)
            }
        )
    
    def _has_question_structure(self, query: str) -> bool:
        """Check if query has question-like structure."""
        question_words = {"what", "when", "where", "why", "who", "how", "which", "can", "could", "would", "should", "is", "are", "do", "does", "did"}
        words = query.lower().split()
        
        # Check for question mark
        if "?" in query:
            return True
        
        # Check for question words at the beginning
        if words and words[0] in question_words:
            return True
        
        return False
    
    def _count_question_words(self, query: str) -> int:
        """Count question words in query."""
        question_words = {"what", "when", "where", "why", "who", "how", "which", "can", "could", "would", "should"}
        words = set(query.lower().split())
        return len(words.intersection(question_words))


class QueryRephraser:
    """Enhances and rephrases queries for better RAG retrieval."""
    
    def __init__(self, config: QueryEnhancementConfig):
        self.config = config
        self.enhancement_history: List[Dict[str, Any]] = []
    
    def enhance_query(self, 
                     query: str, 
                     conversation_context: Optional[ConversationContext] = None,
                     domain_context: Optional[str] = None) -> QueryEnhancementResult:
        """Enhance query for better retrieval performance."""
        start_time = time.time()
        
        if not self.config.enable_rephrasing:
            return QueryEnhancementResult(
                original_query=query,
                enhanced_query=query,
                enhancement_type="none",
                confidence_score=1.0,
                processing_time=time.time() - start_time,
                metadata={"reason": "rephrasing_disabled"}
            )
        
        # Choose enhancement strategy
        enhancement_type = self._determine_enhancement_strategy(query, conversation_context)
        
        try:
            if enhancement_type == "context_aware":
                enhanced_query = self._enhance_with_context(query, conversation_context)
            elif enhancement_type == "domain_adaptive":
                enhanced_query = self._enhance_with_domain(query, domain_context)
            elif enhancement_type == "expansion":
                enhanced_query = self._expand_query(query)
            elif enhancement_type == "rephrasing":
                enhanced_query = self._rephrase_query(query)
            else:
                enhanced_query = query
                enhancement_type = "none"
            
            # Calculate confidence based on enhancement quality
            confidence_score = self._calculate_enhancement_confidence(query, enhanced_query)
            
            result = QueryEnhancementResult(
                original_query=query,
                enhanced_query=enhanced_query,
                enhancement_type=enhancement_type,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                metadata={
                    "length_change": len(enhanced_query) - len(query),
                    "word_change": len(enhanced_query.split()) - len(query.split()),
                    "has_domain_terms": self._has_domain_terms(enhanced_query)
                }
            )
            
            # Store in history
            self.enhancement_history.append({
                "timestamp": time.time(),
                "original_length": len(query),
                "enhanced_length": len(enhanced_query),
                "enhancement_type": enhancement_type,
                "confidence": confidence_score,
                "processing_time": result.processing_time
            })
            
            return result
            
        except Exception as e:
            # Return original query if enhancement fails
            return QueryEnhancementResult(
                original_query=query,
                enhanced_query=query,
                enhancement_type="error",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _determine_enhancement_strategy(self, 
                                      query: str, 
                                      conversation_context: Optional[ConversationContext]) -> str:
        """Determine the best enhancement strategy for the query."""
        query_lower = query.lower()
        
        # Check if query is very short and needs expansion
        if len(query.split()) <= 3:
            return "expansion"
        
        # Check if we have conversation context to leverage
        if (conversation_context and 
            self.config.enable_context_awareness and 
            conversation_context.get_conversation_length() > 2):
            return "context_aware"
        
        # Check if query contains domain-specific terms
        if (self.config.enable_domain_adaptation and 
            self._has_domain_terms(query)):
            return "domain_adaptive"
        
        # Default to simple rephrasing
        return "rephrasing"
    
    def _enhance_with_context(self, query: str, conversation_context: ConversationContext) -> str:
        """Enhance query using conversation context."""
        # Get recent conversation history
        recent_messages = conversation_context.get_recent_messages(5)
        context_text = self._format_conversation_context(recent_messages)
        
        prompt = f"""Given this conversation context, enhance the user's query to be more specific and relevant:

Conversation context:
{context_text}

Current query: "{query}"

Enhance this query by:
1. Adding relevant context from the conversation
2. Making implicit references explicit
3. Clarifying ambiguous terms
4. Maintaining the user's intent

Enhanced query:"""
        
        return self._call_llm_for_enhancement(prompt)
    
    def _enhance_with_domain(self, query: str, domain_context: Optional[str]) -> str:
        """Enhance query with domain-specific knowledge."""
        domain_info = domain_context or "general academic and technical knowledge"
        
        prompt = f"""Enhance the following query for better search in a knowledge base about {domain_info}:

Query: "{query}"

Enhance by:
1. Adding relevant technical terms
2. Clarifying domain-specific concepts
3. Expanding abbreviations if needed
4. Making the query more specific

Enhanced query:"""
        
        return self._call_llm_for_enhancement(prompt)
    
    def _expand_query(self, query: str) -> str:
        """Expand short queries with relevant terms."""
        prompt = f"""Expand this short query to be more comprehensive for document search:

Query: "{query}"

Expand by:
1. Adding synonyms and related terms
2. Including relevant concepts
3. Making the search intent clearer
4. Keeping it focused and relevant

Expanded query:"""
        
        return self._call_llm_for_enhancement(prompt)
    
    def _rephrase_query(self, query: str) -> str:
        """Rephrase query for better search performance."""
        prompt = f"""Rephrase this query to improve search effectiveness in a knowledge base:

Query: "{query}"

Rephrase by:
1. Using clearer, more specific language
2. Organizing concepts logically
3. Removing unnecessary words
4. Maintaining the original intent

Rephrased query:"""
        
        return self._call_llm_for_enhancement(prompt)
    
    def _call_llm_for_enhancement(self, prompt: str) -> str:
        """Call LLM to perform query enhancement."""
        messages = [
            {"role": "system", "content": "You are a query optimization expert. Provide only the enhanced query without explanations."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.chat.completions.create(
            model=self.config.rephrasing_model,
            messages=messages,
            temperature=self.config.rephrasing_temperature,
            max_tokens=self.config.rephrasing_max_tokens
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        
        # Clean up response (remove quotes, extra whitespace)
        enhanced_query = enhanced_query.strip('"\'').strip()
        
        return enhanced_query
    
    def _format_conversation_context(self, messages: List[ChatMessage]) -> str:
        """Format recent messages for context."""
        formatted = []
        for msg in messages[-3:]:  # Last 3 messages for context
            if msg.role != "system":
                formatted.append(f"{msg.role}: {msg.content[:100]}...")
        return "\n".join(formatted)
    
    def _calculate_enhancement_confidence(self, original: str, enhanced: str) -> float:
        """Calculate confidence score for enhancement."""
        if enhanced == original:
            return 0.5  # No change
        
        # Check if enhanced query is reasonable
        if len(enhanced) < 5:
            return 0.1  # Too short
        
        if len(enhanced) > len(original) * 3:
            return 0.3  # Too much expansion
        
        # Check for meaningful enhancement
        original_words = set(original.lower().split())
        enhanced_words = set(enhanced.lower().split())
        
        word_overlap = len(original_words.intersection(enhanced_words))
        total_words = len(original_words.union(enhanced_words))
        
        if total_words == 0:
            return 0.1
        
        # Good enhancement should have some overlap but add new terms
        overlap_ratio = word_overlap / len(original_words) if original_words else 0
        
        if overlap_ratio > 0.5 and len(enhanced_words) > len(original_words):
            return 0.8  # Good enhancement
        elif overlap_ratio > 0.3:
            return 0.6  # Decent enhancement
        else:
            return 0.4  # Uncertain enhancement
    
    def _has_domain_terms(self, text: str) -> bool:
        """Check if text contains domain-specific terms."""
        words = set(text.lower().split())
        return bool(words.intersection(self.config.domain_keywords or set()) or 
                   words.intersection(self.config.technical_terms or set()))
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        if not self.enhancement_history:
            return {"total_enhancements": 0}
        
        total_time = sum(h["processing_time"] for h in self.enhancement_history)
        avg_confidence = sum(h["confidence"] for h in self.enhancement_history) / len(self.enhancement_history)
        
        enhancement_types = {}
        for h in self.enhancement_history:
            etype = h["enhancement_type"]
            enhancement_types[etype] = enhancement_types.get(etype, 0) + 1
        
        return {
            "total_enhancements": len(self.enhancement_history),
            "avg_processing_time": total_time / len(self.enhancement_history),
            "avg_confidence": avg_confidence,
            "enhancement_types": enhancement_types,
            "config": asdict(self.config)
        }


class QueryEnhancementEngine:
    """Main engine that combines validation and enhancement."""
    
    def __init__(self, config: QueryEnhancementConfig = None):
        self.config = config or QueryEnhancementConfig()
        self.validator = QueryValidator(self.config)
        self.rephraser = QueryRephraser(self.config)
    
    def process_query(self, 
                     query: str,
                     conversation_context: Optional[ConversationContext] = None,
                     domain_context: Optional[str] = None,
                     validate_only: bool = False) -> Tuple[QueryValidationResult, Optional[QueryEnhancementResult]]:
        """Process query with validation and optional enhancement."""
        
        # Always validate first
        validation_result = self.validator.validate_query(query)
        
        # If validation fails or we only want validation, return early
        if not validation_result.is_valid or validate_only:
            return validation_result, None
        
        # Enhance the query
        enhancement_result = self.rephraser.enhance_query(
            query, 
            conversation_context, 
            domain_context
        )
        
        return validation_result, enhancement_result
    
    def get_best_query(self, 
                      query: str,
                      conversation_context: Optional[ConversationContext] = None,
                      domain_context: Optional[str] = None) -> str:
        """Get the best version of the query (original or enhanced)."""
        validation_result, enhancement_result = self.process_query(
            query, conversation_context, domain_context
        )
        
        if not validation_result.is_valid:
            return query  # Return original if invalid
        
        if not enhancement_result:
            return query  # Return original if no enhancement
        
        # Use enhanced query if confidence is high enough
        if enhancement_result.confidence_score > 0.5:
            return enhancement_result.enhanced_query
        else:
            return query  # Return original if enhancement confidence is low


def create_query_enhancement_engine(
    enable_rephrasing: bool = True,
    enable_context_awareness: bool = True,
    domain_keywords: Optional[Set[str]] = None
) -> QueryEnhancementEngine:
    """Factory function to create query enhancement engine."""
    
    config = QueryEnhancementConfig(
        enable_rephrasing=enable_rephrasing,
        enable_context_awareness=enable_context_awareness,
        domain_keywords=domain_keywords or set()
    )
    
    return QueryEnhancementEngine(config)


def create_production_enhancement_engine() -> QueryEnhancementEngine:
    """Create production-ready query enhancement engine."""
    
    # Common academic and technical terms
    domain_keywords = {
        "algorithm", "data", "analysis", "research", "study", "method", "approach", 
        "system", "model", "framework", "theory", "concept", "principle", "technique",
        "implementation", "evaluation", "performance", "optimization", "design"
    }
    
    technical_terms = {
        "api", "database", "server", "client", "protocol", "interface", "architecture",
        "scalability", "security", "authentication", "encryption", "deployment", "testing",
        "debugging", "version", "configuration", "integration", "automation"
    }
    
    config = QueryEnhancementConfig(
        rephrasing_model="gpt-4o",  # Better quality
        rephrasing_temperature=0.2,  # More focused
        enable_rephrasing=True,
        enable_expansion=True,
        enable_context_awareness=True,
        enable_domain_adaptation=True,
        domain_keywords=domain_keywords,
        technical_terms=technical_terms
    )
    
    return QueryEnhancementEngine(config)