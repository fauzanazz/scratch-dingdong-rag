import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import chonkie
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    chonkie = None


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    start_char: int
    end_char: int
    chunk_id: str
    source_doc: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update({
            'char_count': len(self.content),
            'word_count': len(self.content.split()),
            'start_char': self.start_char,
            'end_char': self.end_char
        })


@dataclass 
class ChunkingConfig:
    """Configuration for chunking strategies."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    semantic_similarity_threshold: float = 0.7
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    
    # Chonkie-specific configurations
    chonkie_chunker_type: str = "word"  # word, sentence, token, semantic, sdpm
    chonkie_tokenizer: str = "gpt2"  # for token-based chunkers
    chonkie_embedding_model: str = "all-MiniLM-L6-v2"  # for semantic chunkers
    chonkie_similarity_threshold: float = 0.5  # for semantic chunkers
    chonkie_initial_sentences: int = 1  # for SDPM chunker


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    @abstractmethod
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        """Chunk text according to strategy."""
        pass
    
    def _create_chunk_id(self, source_doc: str, chunk_idx: int) -> str:
        """Create unique chunk ID with deterministic hash to avoid duplicates."""
        import hashlib
        
        # Clean the source_doc path to remove problematic characters
        clean_source = source_doc.replace('/', '_').replace('\\', '_').replace(' ', '_')
        
        # Create a deterministic hash based on the full source path and chunk index
        # This ensures uniqueness while being reproducible
        full_identifier = f"{source_doc}_{self.__class__.__name__}_{chunk_idx}"
        content_hash = hashlib.md5(full_identifier.encode()).hexdigest()[:8]
        
        return f"{clean_source}_{self.__class__.__name__}_{chunk_idx:04d}_{content_hash}"


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        chunks = []
        text_len = len(text)
        
        for i in range(0, text_len, self.config.chunk_size - self.config.chunk_overlap):
            start = i
            end = min(i + self.config.chunk_size, text_len)
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if len(chunk_content) < self.config.min_chunk_size and i > 0:
                # Merge small chunk with previous one if possible
                if chunks:
                    prev_chunk = chunks[-1]
                    prev_chunk.content += " " + chunk_content
                    prev_chunk.end_char = end
                    prev_chunk.metadata['char_count'] = len(prev_chunk.content)
                    prev_chunk.metadata['word_count'] = len(prev_chunk.content.split())
                    continue
            
            if chunk_content:
                chunk_id = self._create_chunk_id(source_doc, len(chunks))
                chunk = Chunk(
                    content=chunk_content,
                    start_char=start,
                    end_char=end,
                    chunk_id=chunk_id,
                    source_doc=source_doc
                )
                chunks.append(chunk)
        
        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-aware chunking that preserves sentence boundaries."""
    
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        # Split into sentences using regex
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk_id = self._create_chunk_id(source_doc, len(chunks))
                    end_pos = current_start + len(current_chunk)
                    
                    chunk = Chunk(
                        content=current_chunk,
                        start_char=current_start,
                        end_char=end_pos,
                        chunk_id=chunk_id,
                        source_doc=source_doc
                    )
                    chunks.append(chunk)
                    
                    # Handle overlap by keeping last sentences
                    if self.config.chunk_overlap > 0:
                        overlap_sentences = current_chunk.split('. ')[-2:]
                        current_chunk = '. '.join(overlap_sentences) + ". " + sentence
                        current_start = end_pos - len('. '.join(overlap_sentences))
                    else:
                        current_chunk = sentence
                        current_start = end_pos
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunk_id = self._create_chunk_id(source_doc, len(chunks))
            chunk = Chunk(
                content=current_chunk,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_id=chunk_id,
                source_doc=source_doc
            )
            chunks.append(chunk)
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking using sentence embeddings with memory optimization."""
    
    def __init__(self, config: ChunkingConfig, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(config)
        from ..embeddings.embedding_manager import get_embedding_model
        self.model = get_embedding_model(model_name)
    
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        
        if len(sentences) <= 1:
            # Fallback to fixed chunking for very short texts
            return FixedSizeChunking(self.config).chunk_text(text, source_doc)
        
        # Get sentence embeddings
        embeddings = self.model.encode(sentences)
        
        # Find semantic boundaries using cosine similarity
        chunks = []
        current_sentences = [sentences[0]]
        current_start_idx = 0
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and previous
            similarity = cosine_similarity(
                embeddings[i-1:i], 
                embeddings[i:i+1]
            )[0][0]
            
            # Check if we should start a new chunk
            current_text = " ".join(current_sentences + [sentences[i]])
            should_split = (
                similarity < self.config.semantic_similarity_threshold or
                len(current_text) > self.config.max_chunk_size
            )
            
            if should_split and len(" ".join(current_sentences)) >= self.config.min_chunk_size:
                # Create chunk from current sentences
                chunk_content = " ".join(current_sentences)
                start_char = sum(len(s) + 1 for s in sentences[:current_start_idx]) if current_start_idx > 0 else 0
                end_char = start_char + len(chunk_content)
                
                chunk_id = self._create_chunk_id(source_doc, len(chunks))
                chunk = Chunk(
                    content=chunk_content,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_id=chunk_id,
                    source_doc=source_doc,
                    metadata={'semantic_boundary': True, 'similarity_score': similarity}
                )
                chunks.append(chunk)
                
                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else []
                    current_sentences = overlap_sentences + [sentences[i]]
                    current_start_idx = i - len(overlap_sentences)
                else:
                    current_sentences = [sentences[i]]
                    current_start_idx = i
            else:
                current_sentences.append(sentences[i])
        
        # Add final chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            if len(chunk_content) >= self.config.min_chunk_size:
                start_char = sum(len(s) + 1 for s in sentences[:current_start_idx]) if current_start_idx > 0 else 0
                end_char = start_char + len(chunk_content)
                
                chunk_id = self._create_chunk_id(source_doc, len(chunks))
                chunk = Chunk(
                    content=chunk_content,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_id=chunk_id,
                    source_doc=source_doc,
                    metadata={'semantic_boundary': False}
                )
                chunks.append(chunk)
        
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking that tries to preserve document structure."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.separators = [
            "\n\n\n",  # Multiple newlines (sections)
            "\n\n",    # Double newlines (paragraphs)  
            "\n",      # Single newlines
            ". ",      # Sentences
            " "        # Words
        ]
    
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        return self._recursive_split(text, 0, source_doc, 0)
    
    def _recursive_split(self, text: str, start_char: int, source_doc: str, chunk_counter: int) -> List[Chunk]:
        """Recursively split text using hierarchy of separators."""
        if len(text) <= self.config.chunk_size:
            if len(text.strip()) >= self.config.min_chunk_size:
                chunk_id = self._create_chunk_id(source_doc, chunk_counter)
                return [Chunk(
                    content=text.strip(),
                    start_char=start_char,
                    end_char=start_char + len(text),
                    chunk_id=chunk_id,
                    source_doc=source_doc,
                    metadata={'recursive_level': len(self.separators)}
                )]
            else:
                return []
        
        # Try each separator in order
        for sep_idx, separator in enumerate(self.separators):
            if separator in text:
                splits = text.split(separator)
                if len(splits) > 1:
                    chunks = []
                    current_pos = start_char
                    current_chunk = ""
                    
                    for i, split in enumerate(splits):
                        potential_chunk = current_chunk + separator + split if current_chunk else split
                        
                        if len(potential_chunk) <= self.config.chunk_size:
                            current_chunk = potential_chunk
                        else:
                            # Process current chunk
                            if current_chunk.strip():
                                sub_chunks = self._recursive_split(
                                    current_chunk, current_pos, source_doc, len(chunks)
                                )
                                chunks.extend(sub_chunks)
                                current_pos += len(current_chunk)
                            
                            current_chunk = split
                    
                    # Process final chunk
                    if current_chunk.strip():
                        sub_chunks = self._recursive_split(
                            current_chunk, current_pos, source_doc, len(chunks)
                        )
                        chunks.extend(sub_chunks)
                    
                    return chunks
        
        # If no separators work, force split
        if len(text) > self.config.max_chunk_size:
            mid = len(text) // 2
            left_chunks = self._recursive_split(text[:mid], start_char, source_doc, 0)
            right_chunks = self._recursive_split(text[mid:], start_char + mid, source_doc, len(left_chunks))
            return left_chunks + right_chunks
        
        # Single chunk fallback
        chunk_id = self._create_chunk_id(source_doc, chunk_counter)
        return [Chunk(
            content=text.strip(),
            start_char=start_char,
            end_char=start_char + len(text),
            chunk_id=chunk_id,
            source_doc=source_doc,
            metadata={'recursive_level': -1}  # Force split
        )]


class ChonkieChunking(ChunkingStrategy):
    """Chonkie-based chunking with multiple chunker types."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.chunker = None
        self._initialize_chunker()
    
    def _initialize_chunker(self):
        """Initialize Chonkie chunker based on configuration."""
        if not CHONKIE_AVAILABLE:
            raise ImportError("Chonkie library not available. Install with: pip install chonkie")
        
        try:
            chunker_type = self.config.chonkie_chunker_type.lower()
            
            if chunker_type == "word":
                # Use RecursiveChunker as word-based alternative since WordChunker doesn't exist
                self.chunker = chonkie.RecursiveChunker(
                    tokenizer_or_token_counter="character",
                    chunk_size=self.config.chunk_size,
                    min_characters_per_chunk=self.config.min_chunk_size
                )
                
            elif chunker_type == "sentence":
                self.chunker = chonkie.SentenceChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                
            elif chunker_type == "token":
                self.chunker = chonkie.TokenChunker(
                    tokenizer=self.config.chonkie_tokenizer,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                
            elif chunker_type == "semantic":
                self.chunker = chonkie.SemanticChunker(
                    embedding_model=self.config.chonkie_embedding_model,
                    chunk_size=self.config.chunk_size,
                    threshold=self.config.chonkie_similarity_threshold,
                    min_sentences=self.config.chonkie_initial_sentences
                )
                
            elif chunker_type == "sdpm":  # Semantic Double-Pass Merge
                self.chunker = chonkie.SDPMChunker(
                    embedding_model=self.config.chonkie_embedding_model,
                    chunk_size=self.config.chunk_size,
                    threshold=self.config.chonkie_similarity_threshold,
                    min_sentences=self.config.chonkie_initial_sentences
                )
                
            else:
                raise ValueError(f"Unknown Chonkie chunker type: {chunker_type}. "
                               f"Available: word, sentence, token, semantic, sdpm")
            
            print(f"✓ Initialized Chonkie {chunker_type} chunker")
            
        except Exception as e:
            print(f"✗ Failed to initialize Chonkie chunker: {e}")
            raise
    
    def chunk_text(self, text: str, source_doc: str = "unknown") -> List[Chunk]:
        """Chunk text using Chonkie."""
        if not self.chunker:
            raise RuntimeError("Chonkie chunker not initialized")
        
        try:
            # Use Chonkie to chunk the text
            chonkie_chunks = self.chunker.chunk(text)
            
            # Convert Chonkie chunks to our Chunk format
            chunks = []
            for i, chonkie_chunk in enumerate(chonkie_chunks):
                # Chonkie chunks have text and start_index properties
                chunk_text = chonkie_chunk.text
                start_char = getattr(chonkie_chunk, 'start_index', getattr(chonkie_chunk, 'start_char', i * self.config.chunk_size))
                end_char = getattr(chonkie_chunk, 'end_index', getattr(chonkie_chunk, 'end_char', start_char + len(chunk_text)))
                
                # Filter out chunks that are too small
                if len(chunk_text.strip()) < self.config.min_chunk_size:
                    continue
                
                chunk_id = self._create_chunk_id(source_doc, len(chunks))
                
                # Get additional metadata from Chonkie chunk if available
                chonkie_metadata = {}
                if hasattr(chonkie_chunk, 'metadata') and chonkie_chunk.metadata:
                    chonkie_metadata = chonkie_chunk.metadata
                elif hasattr(chonkie_chunk, '__dict__'):
                    # Extract relevant attributes as metadata
                    for attr, value in chonkie_chunk.__dict__.items():
                        if not attr.startswith('_') and attr not in ['text', 'start_index']:
                            chonkie_metadata[f'chonkie_{attr}'] = value
                
                chunk = Chunk(
                    content=chunk_text.strip(),
                    start_char=start_char,
                    end_char=end_char,
                    chunk_id=chunk_id,
                    source_doc=source_doc,
                    metadata={
                        'chonkie_type': self.config.chonkie_chunker_type,
                        **chonkie_metadata
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"✗ Chonkie chunking failed: {e}")
            # Fallback to recursive chunking
            print("  → Falling back to recursive chunking...")
            return RecursiveChunking(self.config).chunk_text(text, source_doc)


def get_chunking_strategy(strategy_name: str, config: ChunkingConfig) -> ChunkingStrategy:
    """Factory function to get chunking strategy by name."""
    strategies = {
        'fixed': FixedSizeChunking,
        'sentence': SentenceChunking,
        'semantic': SemanticChunking,
        'recursive': RecursiveChunking,
        'chonkie': ChonkieChunking
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](config)