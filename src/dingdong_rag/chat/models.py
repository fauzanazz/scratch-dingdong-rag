import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # system, user, assistant
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Context for a conversation session."""
    conversation_id: str
    messages: List[ChatMessage]
    retrieved_contexts: List[Dict[str, Any]]  # Store retrieval results per turn
    created_at: float
    last_updated: float
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_updated = time.time()
    
    def get_recent_messages(self, max_messages: int = 10) -> List[ChatMessage]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]
    
    def get_conversation_length(self) -> int:
        """Get total number of messages."""
        return len(self.messages)