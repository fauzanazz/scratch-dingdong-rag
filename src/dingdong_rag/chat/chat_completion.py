try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently fail if dotenv is not available

import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import openai

from ..retrieval.vector_store import SearchResult
from ..retrieval.reranking import RerankingResult
from .context_compression import ContextCompressor, AutoCompressor, CompressionResult, create_context_compressor
from .query_enhancement import QueryEnhancementEngine, QueryValidationResult, QueryEnhancementResult, create_query_enhancement_engine


from .models import ChatMessage, ConversationContext


@dataclass
class ChatCompletionConfig:
    """Configuration for chat completion."""
    # Model settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1500
    top_p: float = 0.9
    
    # Context management
    max_context_tokens: int = 4000
    max_conversation_messages: int = 10
    context_compression: bool = True
    
    # RAG settings
    include_sources: bool = True
    source_citation_format: str = "[{source_doc}]"
    context_separator: str = "\n\n---\n\n"
    
    # Query enhancement settings
    enable_query_enhancement: bool = True
    enable_query_validation: bool = True
    enable_auto_compression: bool = True
    
    # System prompt
    system_prompt: str = """You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately and comprehensively. 
If the context doesn't contain enough information, say so clearly.
Always cite your sources when referencing specific information."""
    
    # OpenAI settings
    openai_api_key: Optional[str] = None


@dataclass
class ChatResponse:
    """Response from chat completion."""
    message: str
    conversation_id: str
    sources_used: List[str]
    context_snippets: List[str]
    response_time: float
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]
    
    # Enhancement results
    query_validation: Optional[QueryValidationResult] = None
    query_enhancement: Optional[QueryEnhancementResult] = None
    compression_result: Optional[CompressionResult] = None


class ContextPruner:
    """Prunes and optimizes context for chat completion."""
    
    def __init__(self, config: ChatCompletionConfig):
        self.config = config
    
    def prune_context(self, 
                     reranked_results: List[RerankingResult], 
                     query: str,
                     conversation_history: List[ChatMessage] = None) -> Tuple[str, List[str]]:
        """Prune context to fit within token limits."""
        
        if not reranked_results:
            return "", []
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        available_tokens = self.config.max_context_tokens
        
        # Reserve tokens for conversation history
        if conversation_history:
            history_text = self._format_conversation_history(conversation_history)
            history_tokens = len(history_text) // 4
            available_tokens -= history_tokens
        
        # Reserve tokens for system prompt and query
        system_tokens = len(self.config.system_prompt) // 4
        query_tokens = len(query) // 4
        available_tokens -= (system_tokens + query_tokens + 200)  # 200 buffer
        
        # Select context chunks that fit
        selected_chunks = []
        selected_sources = []
        current_tokens = 0
        
        for result in reranked_results:
            chunk_tokens = len(result.search_result.content) // 4
            
            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(result.search_result.content)
                source = result.search_result.metadata.get('source_doc', 'Unknown')
                if source not in selected_sources:
                    selected_sources.append(source)
                current_tokens += chunk_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space
                    truncated_content = result.search_result.content[:remaining_tokens * 4]
                    selected_chunks.append(truncated_content + "...")
                    source = result.search_result.metadata.get('source_doc', 'Unknown')
                    if source not in selected_sources:
                        selected_sources.append(source)
                break
        
        # Format context
        context = self.config.context_separator.join(selected_chunks)
        
        return context, selected_sources
    
    def _format_conversation_history(self, messages: List[ChatMessage]) -> str:
        """Format conversation history for context."""
        formatted_history_lines = []
        for message in messages[-self.config.max_conversation_messages:]:
            formatted_history_lines.append(f"{message.role.title()}: {message.content}")
        return "\n".join(formatted_history_lines)


class ConversationManager:
    """Manages conversation sessions and memory."""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}
    
    def create_conversation(self, system_prompt: Optional[str] = None) -> str:
        """Create a new conversation session."""
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            messages=[],
            retrieved_contexts=[],
            created_at=time.time(),
            last_updated=time.time()
        )
        
        # Add system message if provided
        if system_prompt:
            context.add_message("system", system_prompt)
        
        self.conversations[conversation_id] = context
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def add_user_message(self, conversation_id: str, message: str, retrieved_context: Optional[Dict] = None):
        """Add user message to conversation."""
        if conversation_id in self.conversations:
            context = self.conversations[conversation_id]
            context.add_message("user", message)
            
            if retrieved_context:
                context.retrieved_contexts.append(retrieved_context)
    
    def add_assistant_message(self, conversation_id: str, message: str, metadata: Optional[Dict] = None):
        """Add assistant message to conversation."""
        if conversation_id in self.conversations:
            context = self.conversations[conversation_id]
            context.add_message("assistant", message, metadata)
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Remove old conversations to free memory."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_remove = []
        for conv_id, context in self.conversations.items():
            if context.last_updated < cutoff_time:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]
        
        if to_remove:
            print(f"✓ Cleaned up {len(to_remove)} old conversations")


class ChatCompletionEngine:
    """Main chat completion engine with enhanced query processing and context management."""
    
    def __init__(self, config: ChatCompletionConfig = None):
        self.config = config or ChatCompletionConfig()
        self.conversation_manager = ConversationManager()
        self.context_pruner = ContextPruner(self.config)
        self.completion_history = []
        
        # Initialize enhancement components
        if self.config.enable_query_enhancement or self.config.enable_query_validation:
            self.query_enhancer = create_query_enhancement_engine(
                enable_rephrasing=self.config.enable_query_enhancement,
                enable_context_awareness=True
            )
        else:
            self.query_enhancer = None
        
        if self.config.enable_auto_compression:
            compressor = create_context_compressor()
            self.auto_compressor = AutoCompressor(compressor)
        else:
            self.auto_compressor = None
        
        # Set up OpenAI
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        elif not openai.api_key:
            import os
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def complete_chat(self, 
                     query: str,
                     reranked_results: List[RerankingResult],
                     conversation_id: Optional[str] = None,
                     domain_context: Optional[str] = None) -> ChatResponse:
        """Generate chat completion using retrieved context with enhanced query processing."""
        
        start_time = time.time()
        
        # Initialize result tracking
        query_validation = None
        query_enhancement = None
        compression_result = None
        processed_query = query
        
        # Create or get conversation
        if not conversation_id:
            conversation_id = self.conversation_manager.create_conversation(self.config.system_prompt)
        
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            conversation_id = self.conversation_manager.create_conversation(self.config.system_prompt)
            conversation = self.conversation_manager.get_conversation(conversation_id)
        
        # Apply automatic compression if enabled
        if self.auto_compressor:
            compression_result = self.auto_compressor.process_conversation(conversation)
        
        # Enhance/validate query if enabled
        if self.query_enhancer:
            try:
                query_validation, query_enhancement = self.query_enhancer.process_query(
                    query, conversation, domain_context
                )
                
                # Use enhanced query if available and high confidence
                if (query_enhancement and 
                    query_enhancement.confidence_score > 0.5 and 
                    query_validation.is_valid):
                    processed_query = query_enhancement.enhanced_query
                elif not query_validation.is_valid:
                    # If validation fails, we'll still proceed but log the issues
                    print(f"Query validation issues: {query_validation.issues}")
                    
            except Exception as e:
                print(f"Query enhancement failed: {e}")
                # Continue with original query if enhancement fails
        
        # Prune context to fit token limits (use processed query for better context)
        context, sources_used = self.context_pruner.prune_context(
            reranked_results, 
            processed_query, 
            conversation.get_recent_messages()
        )
        
        # Store retrieved context
        retrieved_context = {
            'original_query': query,
            'processed_query': processed_query,
            'context': context,
            'sources': sources_used,
            'result_count': len(reranked_results),
            'timestamp': time.time()
        }
        
        # Add user message (store original query for conversation history)
        self.conversation_manager.add_user_message(conversation_id, query, retrieved_context)
        
        # Generate response (use processed query for better results)
        try:
            response_text, token_usage = self._generate_response(processed_query, context, conversation)
            
            # Add assistant message
            response_metadata = {
                'sources_used': sources_used,
                'token_usage': token_usage,
                'response_time': time.time() - start_time
            }
            self.conversation_manager.add_assistant_message(conversation_id, response_text, response_metadata)
            
            # Create response object with enhancement results
            chat_response = ChatResponse(
                message=response_text,
                conversation_id=conversation_id,
                sources_used=sources_used,
                context_snippets=[r.search_result.content[:200] + "..." for r in reranked_results[:3]],
                response_time=time.time() - start_time,
                token_usage=token_usage,
                metadata=response_metadata,
                query_validation=query_validation,
                query_enhancement=query_enhancement,
                compression_result=compression_result
            )
            
            # Store in history with enhancement information
            history_entry = {
                'timestamp': time.time(),
                'original_query': query,
                'processed_query': processed_query,
                'response_length': len(response_text),
                'sources_count': len(sources_used),
                'response_time': chat_response.response_time,
                'token_usage': token_usage,
                'query_enhanced': query != processed_query,
                'compression_applied': compression_result is not None
            }
            
            if query_enhancement:
                history_entry['enhancement_confidence'] = query_enhancement.confidence_score
                history_entry['enhancement_type'] = query_enhancement.enhancement_type
            
            if compression_result:
                history_entry['tokens_saved'] = compression_result.tokens_saved
                history_entry['messages_compressed'] = compression_result.original_message_count
            
            self.completion_history.append(history_entry)
            
            return chat_response
            
        except Exception as e:
            error_response = ChatResponse(
                message=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                conversation_id=conversation_id,
                sources_used=[],
                context_snippets=[],
                response_time=time.time() - start_time,
                token_usage={'error': 1},
                metadata={'error': str(e)},
                query_validation=query_validation,
                query_enhancement=query_enhancement,
                compression_result=compression_result
            )
            
            return error_response
    
    def _generate_response(self, 
                          query: str, 
                          context: str, 
                          conversation: ConversationContext) -> Tuple[str, Dict[str, int]]:
        """Generate response using OpenAI API."""
        
        # Prepare messages
        messages = []
        
        # Add system message
        system_content = self.config.system_prompt
        if context:
            system_content += f"\n\nRelevant context:\n{context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (recent messages only)
        recent_messages = conversation.get_recent_messages(self.config.max_conversation_messages)
        for message in recent_messages:
            if message.role != "system":  # Skip system messages (already added)
                messages.append({"role": message.role, "content": message.content})
        
        # Generate response
        response = openai.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        
        response_text = response.choices[0].message.content
        token_usage = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        
        return response_text, token_usage
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session."""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            return [asdict(msg) for msg in conversation.messages]
        return None
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get comprehensive chat completion statistics."""
        if not self.completion_history:
            return {"total_completions": 0}
        
        total_tokens = sum(h.get('token_usage', {}).get('total_tokens', 0) for h in self.completion_history)
        total_time = sum(h['response_time'] for h in self.completion_history)
        avg_time = total_time / len(self.completion_history)
        
        # Enhancement statistics
        enhanced_queries = sum(1 for h in self.completion_history if h.get('query_enhanced', False))
        compressions_applied = sum(1 for h in self.completion_history if h.get('compression_applied', False))
        total_tokens_saved = sum(h.get('tokens_saved', 0) for h in self.completion_history)
        total_messages_compressed = sum(h.get('messages_compressed', 0) for h in self.completion_history)
        
        stats = {
            'total_completions': len(self.completion_history),
            'total_tokens_used': total_tokens,
            'avg_response_time': avg_time,
            'total_response_time': total_time,
            'active_conversations': len(self.conversation_manager.conversations),
            'model': self.config.model,
            'config': asdict(self.config),
            
            # Enhancement statistics
            'query_enhancement': {
                'enhanced_queries': enhanced_queries,
                'enhancement_rate': enhanced_queries / len(self.completion_history) if self.completion_history else 0,
            },
            
            # Compression statistics
            'context_compression': {
                'compressions_applied': compressions_applied,
                'compression_rate': compressions_applied / len(self.completion_history) if self.completion_history else 0,
                'total_tokens_saved': total_tokens_saved,
                'total_messages_compressed': total_messages_compressed
            }
        }
        
        # Add component-specific stats if available
        if self.query_enhancer:
            stats['query_enhancement'].update(self.query_enhancer.get_enhancement_stats())
        
        if self.auto_compressor:
            stats['context_compression'].update(self.auto_compressor.compressor.get_compression_stats())
        
        return stats
    
    def cleanup_resources(self):
        """Cleanup old conversations and resources."""
        self.conversation_manager.cleanup_old_conversations()


def create_chat_completion_engine(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    enable_query_enhancement: bool = True,
    enable_auto_compression: bool = True,
    openai_api_key: Optional[str] = None
) -> ChatCompletionEngine:
    """Factory function to create chat completion engine with enhancements."""
    
    config = ChatCompletionConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_query_enhancement=enable_query_enhancement,
        enable_query_validation=enable_query_enhancement,  # Enable validation with enhancement
        enable_auto_compression=enable_auto_compression,
        openai_api_key=openai_api_key
    )
    
    return ChatCompletionEngine(config)


def create_production_chat_engine(openai_api_key: Optional[str] = None) -> ChatCompletionEngine:
    """Create production-ready chat completion engine with full enhancements."""
    
    config = ChatCompletionConfig(
        model="gpt-4o",  # Higher quality model
        temperature=0.6,  # Slightly more focused
        max_tokens=2000,
        max_context_tokens=6000,  # More context
        include_sources=True,
        context_compression=True,
        enable_query_enhancement=True,
        enable_query_validation=True,
        enable_auto_compression=True,
        openai_api_key=openai_api_key
    )
    
    return ChatCompletionEngine(config)


def create_fast_chat_engine(openai_api_key: Optional[str] = None) -> ChatCompletionEngine:
    """Create fast chat completion engine for development."""
    
    config = ChatCompletionConfig(
        model="gpt-4o-mini",  # Faster model
        temperature=0.8,
        max_tokens=1000,
        max_context_tokens=2000,  # Less context for speed
        max_conversation_messages=5,
        enable_query_enhancement=False,  # Disable for speed
        enable_query_validation=True,   # Keep validation for safety
        enable_auto_compression=False,  # Disable for speed
        openai_api_key=openai_api_key
    )
    
    return ChatCompletionEngine(config)