import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import openai

from .models import ChatMessage, ConversationContext


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    # Compression triggers
    max_uncompressed_messages: int = 15
    max_total_tokens: int = 8000
    compression_ratio: float = 0.4  # Target: compress to 40% of original
    
    # Compression strategy
    preserve_recent_messages: int = 5  # Always keep last N messages uncompressed
    preserve_system_messages: bool = True
    preserve_user_queries: bool = True
    
    # LLM settings for compression
    compression_model: str = "gpt-4o-mini"
    compression_temperature: float = 0.3
    compression_max_tokens: int = 1000
    
    # Summarization prompt
    summarization_prompt: str = """Summarize the following conversation while preserving:
1. Key information and context
2. Important decisions or conclusions
3. User's main questions and concerns
4. Assistant's key responses and recommendations

Focus on maintaining continuity for the ongoing conversation. Be concise but comprehensive."""


@dataclass
class CompressionResult:
    """Result of context compression operation."""
    compressed_summary: str
    original_message_count: int
    compressed_message_count: int
    tokens_saved: int
    compression_time: float
    metadata: Dict[str, Any]


class ContextCompressor:
    """Handles compression of long conversation histories."""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.compression_history: List[Dict[str, Any]] = []
    
    def should_compress(self, conversation: ConversationContext) -> bool:
        """Determine if conversation needs compression."""
        message_count = conversation.get_conversation_length()
        
        # Check message count threshold
        if message_count > self.config.max_uncompressed_messages:
            return True
        
        # Check token count threshold (rough estimate)
        total_content = sum(len(message.content) for message in conversation.messages)
        estimated_tokens = total_content // 4  # Rough estimation
        
        if estimated_tokens > self.config.max_total_tokens:
            return True
        
        return False
    
    def compress_conversation(self, conversation: ConversationContext) -> CompressionResult:
        """Compress conversation history while preserving important context."""
        start_time = time.time()
        
        if not self.should_compress(conversation):
            return CompressionResult(
                compressed_summary="",
                original_message_count=0,
                compressed_message_count=0,
                tokens_saved=0,
                compression_time=0,
                metadata={"reason": "compression_not_needed"}
            )
        
        # Separate messages to compress vs preserve
        messages_to_compress, messages_to_preserve = self._separate_messages(conversation.messages)
        
        if not messages_to_compress:
            return CompressionResult(
                compressed_summary="",
                original_message_count=0,
                compressed_message_count=0,
                tokens_saved=0,
                compression_time=time.time() - start_time,
                metadata={"reason": "no_messages_to_compress"}
            )
        
        # Generate summary of messages to compress
        compressed_summary = self._generate_summary(messages_to_compress)
        
        # Calculate metrics
        original_tokens = sum(len(message.content) // 4 for message in messages_to_compress)
        compressed_tokens = len(compressed_summary) // 4
        tokens_saved = original_tokens - compressed_tokens
        
        result = CompressionResult(
            compressed_summary=compressed_summary,
            original_message_count=len(messages_to_compress),
            compressed_message_count=1,  # Summary is 1 message
            tokens_saved=tokens_saved,
            compression_time=time.time() - start_time,
            metadata={
                "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0,
                "messages_preserved": len(messages_to_preserve),
                "original_tokens_estimate": original_tokens,
                "compressed_tokens_estimate": compressed_tokens
            }
        )
        
        # Store compression history
        self.compression_history.append({
            "timestamp": time.time(),
            "conversation_id": conversation.conversation_id,
            "result": asdict(result)
        })
        
        return result
    
    def apply_compression(self, conversation: ConversationContext, compression_result: CompressionResult):
        """Apply compression result to conversation."""
        if not compression_result.compressed_summary:
            return  # No compression needed
        
        # Separate messages
        messages_to_compress, messages_to_preserve = self._separate_messages(conversation.messages)
        
        # Create new message list with compressed summary + preserved messages
        new_messages = []
        
        # Add compressed summary as a system message
        if compression_result.compressed_summary:
            summary_message = ChatMessage(
                role="system",
                content=f"[CONVERSATION SUMMARY]\n{compression_result.compressed_summary}",
                timestamp=time.time(),
                metadata={
                    "is_compression_summary": True,
                    "original_message_count": compression_result.original_message_count,
                    "compression_time": compression_result.compression_time
                }
            )
            new_messages.append(summary_message)
        
        # Add preserved messages
        new_messages.extend(messages_to_preserve)
        
        # Update conversation
        conversation.messages = new_messages
        conversation.last_updated = time.time()
    
    def _separate_messages(self, messages: List[ChatMessage]) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """Separate messages into those to compress and those to preserve."""
        if len(messages) <= self.config.preserve_recent_messages:
            return [], messages
        
        # Always preserve recent messages
        messages_to_preserve = messages[-self.config.preserve_recent_messages:]
        messages_to_compress = messages[:-self.config.preserve_recent_messages]
        
        # Filter out messages we should preserve even in older history
        filtered_compress = []
        additional_preserve = []
        
        for message in messages_to_compress:
            # Skip compression summaries (don't compress already compressed content)
            if message.metadata and message.metadata.get("is_compression_summary"):
                additional_preserve.append(message)
            # Preserve system messages if configured
            elif self.config.preserve_system_messages and message.role == "system":
                additional_preserve.append(message)
            else:
                filtered_compress.append(message)
        
        # Combine preserved messages (maintain chronological order)
        all_preserved = additional_preserve + messages_to_preserve
        
        return filtered_compress, all_preserved
    
    def _generate_summary(self, messages: List[ChatMessage]) -> str:
        """Generate LLM-based summary of message history."""
        if not messages:
            return ""
        
        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(messages)
        
        # Prepare summarization prompt
        messages_for_llm = [
            {"role": "system", "content": self.config.summarization_prompt},
            {"role": "user", "content": f"Conversation to summarize:\n\n{conversation_text}"}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=self.config.compression_model,
                messages=messages_for_llm,
                temperature=self.config.compression_temperature,
                max_tokens=self.config.compression_max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback: create simple extractive summary
            return self._create_fallback_summary(messages)
    
    def _format_messages_for_summary(self, messages: List[ChatMessage]) -> str:
        """Format messages for LLM summarization."""
        formatted_messages = []
        
        for message in messages:
            # Skip system messages that are already summaries
            if message.metadata and message.metadata.get("is_compression_summary"):
                continue
            
            role_label = message.role.title()
            timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M")
            formatted_messages.append(f"[{timestamp}] {role_label}: {message.content}")
        
        return "\n".join(formatted_messages)
    
    def _create_fallback_summary(self, messages: List[ChatMessage]) -> str:
        """Create simple extractive summary as fallback."""
        # Group by role and extract key information
        user_queries = []
        assistant_responses = []
        
        for message in messages:
            if message.role == "user":
                # Take first 100 chars of user queries
                user_queries.append(message.content[:100] + "..." if len(message.content) > 100 else message.content)
            elif message.role == "assistant":
                # Take first 150 chars of assistant responses
                assistant_responses.append(message.content[:150] + "..." if len(message.content) > 150 else message.content)
        
        # Create summary
        summary_parts = []
        
        if user_queries:
            summary_parts.append(f"User queries: {' | '.join(user_queries[-3:])}")  # Last 3 queries
        
        if assistant_responses:
            summary_parts.append(f"Key responses: {' | '.join(assistant_responses[-3:])}")  # Last 3 responses
        
        return "\n".join(summary_parts) if summary_parts else "Previous conversation context."
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compression_history:
            return {"total_compressions": 0}
        
        total_tokens_saved = sum(
            h["result"]["tokens_saved"] 
            for h in self.compression_history
        )
        
        total_messages_compressed = sum(
            h["result"]["original_message_count"] 
            for h in self.compression_history
        )
        
        avg_compression_time = sum(
            h["result"]["compression_time"] 
            for h in self.compression_history
        ) / len(self.compression_history)
        
        return {
            "total_compressions": len(self.compression_history),
            "total_tokens_saved": total_tokens_saved,
            "total_messages_compressed": total_messages_compressed,
            "avg_compression_time": avg_compression_time,
            "config": asdict(self.config)
        }


class AutoCompressor:
    """Automatically manages compression for conversation contexts."""
    
    def __init__(self, compressor: ContextCompressor):
        self.compressor = compressor
        self.auto_compression_enabled = True
    
    def process_conversation(self, conversation: ConversationContext) -> Optional[CompressionResult]:
        """Process conversation and apply compression if needed."""
        if not self.auto_compression_enabled:
            return None
        
        if not self.compressor.should_compress(conversation):
            return None
        
        # Perform compression
        compression_result = self.compressor.compress_conversation(conversation)
        
        # Apply compression to conversation
        self.compressor.apply_compression(conversation, compression_result)
        
        return compression_result
    
    def enable_auto_compression(self):
        """Enable automatic compression."""
        self.auto_compression_enabled = True
    
    def disable_auto_compression(self):
        """Disable automatic compression."""
        self.auto_compression_enabled = False


def create_context_compressor(
    max_uncompressed_messages: int = 15,
    max_total_tokens: int = 8000,
    compression_ratio: float = 0.4
) -> ContextCompressor:
    """Factory function to create context compressor."""
    
    config = CompressionConfig(
        max_uncompressed_messages=max_uncompressed_messages,
        max_total_tokens=max_total_tokens,
        compression_ratio=compression_ratio
    )
    
    return ContextCompressor(config)


def create_production_compressor() -> ContextCompressor:
    """Create production-ready context compressor."""
    
    config = CompressionConfig(
        max_uncompressed_messages=20,
        max_total_tokens=12000,
        compression_ratio=0.3,  # More aggressive compression
        preserve_recent_messages=8,
        compression_model="gpt-4o",  # Better quality
        compression_temperature=0.2  # More focused
    )
    
    return ContextCompressor(config)