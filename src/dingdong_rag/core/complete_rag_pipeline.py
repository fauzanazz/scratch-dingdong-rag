try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently fail if dotenv is not available

import os
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

from ..parsing.pdf_parser import ValidationConfig
from ..chunking.chunking import ChunkingConfig
from ..embeddings.embedding_config import EmbeddingFunction
from ..retrieval.vector_store import VectorStore
from ..retrieval.reranking import RerankingPipeline
from ..retrieval.crag import CRAGRefinement
from ..chat.chat_completion import ChatCompletionEngine


@dataclass
class CompleteRAGConfig:
    """Configuration for the complete RAG pipeline."""
    # Document processing
    documents_dir: str = "./documents"
    working_dir: str = "./complete_rag"
    max_documents: Optional[int] = None
    
    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1200
    chunk_overlap: int = 300
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    
    # Vector storage
    vector_store_type: str = "chroma"  # chroma or pinecone
    collection_name: str = "rag_collection"
    
    # Pinecone specific (if using)
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "rag-index"
    
    # Retrieval
    retrieval_top_k: int = 50
    
    # Reranking
    reranking_strategy: str = "hybrid"  # cross_encoder, bm25, hybrid
    reranking_top_k: int = 10
    
    # CRAG refinement (new)
    enable_crag: bool = False
    crag_trigger_mode: str = "hybrid"  # always, score, token_coverage, hybrid
    crag_min_similarity: float = 0.22
    crag_min_token_coverage: float = 0.35
    crag_max_reformulations: int = 2
    crag_reformulation_model: str = "gpt-4o-mini"
    
    # Chat completion
    llm_model: str = "gpt-4o-mini"
    max_response_tokens: int = 1500
    temperature: float = 0.7
    
    # Self-RAG settings removed in library version
    
    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")


@dataclass
class RAGResponse:
    """Complete response from RAG pipeline."""
    query: str
    answer: str
    conversation_id: str
    
    # Retrieval details
    retrieved_chunks: int
    reranked_chunks: int
    sources_used: List[str]
    
    # Performance metrics
    retrieval_time: float
    reranking_time: float
    completion_time: float
    total_time: float
    
    # Token usage
    token_usage: Dict[str, int]
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompleteRAGPipeline:
    """Complete RAG pipeline integrating all components."""
    
    def __init__(self, config: CompleteRAGConfig):
        self.config = config
        
        # Initialize components
        self.embedding_function: Optional[EmbeddingFunction] = None
        self.vector_store: Optional[VectorStore] = None
        self.crag_refinement: Optional[CRAGRefinement] = None
        self.reranking_pipeline: Optional[RerankingPipeline] = None
        self.chat_engine: Optional[ChatCompletionEngine] = None
        
        # Performance tracking
        from .services.stats import initialize_stats
        self.pipeline_stats = initialize_stats()
        
        # Create working directory
        Path(self.config.working_dir).mkdir(parents=True, exist_ok=True)
    
    def initialize_pipeline(self) -> Dict[str, Any]:
        """Initialize the complete RAG pipeline."""
        
        print("Initializing Complete RAG Pipeline")
        print("=" * 50)
        
        initialization_results = {
            'steps_completed': [],
            'steps_failed': [],
            'total_time': 0
        }
        from .services.initializer import initialize_components
        components, summary = initialize_components(self.config)
        self.embedding_function = components.embedding_function
        self.vector_store = components.vector_store
        self.crag_refinement = components.crag_refinement
        self.reranking_pipeline = components.reranking_pipeline
        self.chat_engine = components.chat_engine
        print(f"Components initialized: {len(summary.get('steps_completed', []))}")
        return summary
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest documents into the vector store."""
        
        if not self.vector_store:
            return {'error': 'Vector store not initialized'}
        
        from .services.ingestor import ingest_documents as run_ingestion
        result = run_ingestion(self.config, self.vector_store)
        if isinstance(result, dict) and 'error' in result:
            return result
        self.pipeline_stats['total_documents_processed'] = result.documents_processed
        self.pipeline_stats['total_chunks_stored'] = result.chunks_stored
        print(f"\nDocument ingestion complete in {result.ingestion_time:.2f}s")
        return {
            'documents_processed': result.documents_processed,
            'documents_failed': result.documents_failed,
            'chunks_created': result.chunks_created,
            'chunks_stored': result.chunks_stored,
            'ingestion_time': result.ingestion_time,
            'average_chunks_per_doc': result.average_chunks_per_doc,
        }
    
    def query_pipeline(self, 
                      query: str, 
                      conversation_id: Optional[str] = None,
                      save_results: bool = True) -> RAGResponse:
        """Execute complete RAG pipeline for a query."""
        
        if not all([self.vector_store, self.reranking_pipeline, self.chat_engine]):
            return RAGResponse(
                query=query,
                answer="Error: Pipeline not fully initialized",
                conversation_id="",
                retrieved_chunks=0,
                reranked_chunks=0,
                sources_used=[],
                retrieval_time=0,
                reranking_time=0,
                completion_time=0,
                total_time=0,
                token_usage={},
                metadata={'error': 'Pipeline not initialized'}
            )
        
        from .services.query_executor import execute_query
        result = execute_query(
            query=query,
            config=self.config,
            vector_store=self.vector_store,
            reranking_pipeline=self.reranking_pipeline,
            chat_engine=self.chat_engine,
            crag_refinement=self.crag_refinement,
            conversation_id=conversation_id,
            save_results=save_results,
        )
        from .services.stats import update_after_query
        self.pipeline_stats = update_after_query(
            self.pipeline_stats,
            retrieval_time=result.get('retrieval_time', 0.0),
            reranking_time=result.get('reranking_time', 0.0),
            completion_time=result.get('completion_time', 0.0),
            total_tokens_used_increment=(result.get('token_usage', {}) or {}).get('total_tokens', 0),
        )
        return RAGResponse(
            query=query,
            answer=result.get('answer', ''),
            conversation_id=result.get('conversation_id', conversation_id or ''),
            retrieved_chunks=result.get('retrieved_chunks', 0),
            reranked_chunks=result.get('reranked_chunks', 0),
            sources_used=result.get('sources_used', []),
            retrieval_time=result.get('retrieval_time', 0.0),
            reranking_time=result.get('reranking_time', 0.0),
            completion_time=result.get('completion_time', 0.0),
            total_time=result.get('total_time', 0.0),
            token_usage=result.get('token_usage', {}),
            metadata=result.get('metadata', {}),
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        from .services.stats import build_stats_payload
        return build_stats_payload(
            base_stats=self.pipeline_stats,
            config=self.config,
            vector_store=self.vector_store,
            reranking_pipeline=self.reranking_pipeline,
            chat_engine=self.chat_engine,
        )
    
    def save_pipeline_state(self, filepath: Optional[str] = None):
        """Save pipeline configuration and stats."""
        
        if not filepath:
            filepath = f"{self.config.working_dir}/pipeline_state.json"
        
        state = {
            'config': asdict(self.config),
            'stats': self.get_pipeline_stats(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Pipeline state saved to: {filepath}")
    
        # Step 6: Self-RAG integration removed


def create_complete_rag_pipeline(
    documents_dir: str = "./documents",
    working_dir: str = "./complete_rag",
    vector_store_type: str = "chroma",
    embedding_model: str = "all-MiniLM-L6-v2",
    openai_api_key: Optional[str] = None
) -> CompleteRAGPipeline:
    """Factory function to create complete RAG pipeline."""
    
    config = CompleteRAGConfig(
        documents_dir=documents_dir,
        working_dir=working_dir,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key
    )
    
    return CompleteRAGPipeline(config)


def create_production_rag_pipeline(
    documents_dir: str = "./documents",
    working_dir: str = "./production_rag",
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None
) -> CompleteRAGPipeline:
    """Create production-ready RAG pipeline."""
    
    config = CompleteRAGConfig(
        documents_dir=documents_dir,
        working_dir=working_dir,
        
        # Production settings
        embedding_model="text-embedding-3-large",
        vector_store_type="pinecone",
        reranking_strategy="hybrid",
        llm_model="gpt-4o",
        
        # Higher retrieval settings
        retrieval_top_k=100,
        reranking_top_k=15,
        max_response_tokens=2000,
        temperature=0.6,
        
        # API keys
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    
    return CompleteRAGPipeline(config)