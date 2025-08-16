from dataclasses import dataclass
from typing import Optional, Dict, Any

from .core.complete_rag_pipeline import (
    CompleteRAGConfig,
    CompleteRAGPipeline,
    RAGResponse,
)


@dataclass
class DingDongRAG:
    documents_dir: str = "./documents"
    working_dir: str = "./rag_working_dir"

    # Chunking
    chunking_strategy: str = "recursive"  # fixed | sentence | recursive | semantic | chonkie
    chunk_size: int = 1200
    chunk_overlap: int = 300

    # Embeddings
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    openai_api_key: Optional[str] = None

    # Vector store
    vector_store_type: str = "chroma"  # chroma | pinecone
    collection_name: str = "rag_collection"
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "rag-index"

    # Retrieval / reranking
    retrieval_top_k: int = 75
    reranking_strategy: str = "hybrid"  # cross_encoder | bm25 | hybrid | cohere
    reranking_top_k: int = 20

    # CRAG (optional)
    enable_crag: bool = False
    crag_trigger_mode: str = "hybrid"  # always | score | token_coverage | hybrid
    crag_min_similarity: float = 0.22
    crag_min_token_coverage: float = 0.35
    crag_max_reformulations: int = 2
    crag_reformulation_model: str = "gpt-4o-mini"

    # Chat completion
    llm_model: str = "gpt-4o"
    max_response_tokens: int = 2000
    temperature: float = 0.6

    # Internals (initialized later)
    _config: Optional[CompleteRAGConfig] = None
    _pipeline: Optional[CompleteRAGPipeline] = None

    # ----- Lifecycle -----------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        """Initialize embeddings, vector store, reranker, and chat engine."""
        self._config = CompleteRAGConfig(
            documents_dir=self.documents_dir,
            working_dir=self.working_dir,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
            openai_api_key=self.openai_api_key,
            vector_store_type=self.vector_store_type,
            collection_name=self.collection_name,
            pinecone_api_key=self.pinecone_api_key,
            pinecone_index=self.pinecone_index,
            retrieval_top_k=self.retrieval_top_k,
            reranking_strategy=self.reranking_strategy,
            reranking_top_k=self.reranking_top_k,
            enable_crag=self.enable_crag,
            crag_trigger_mode=self.crag_trigger_mode,
            crag_min_similarity=self.crag_min_similarity,
            crag_min_token_coverage=self.crag_min_token_coverage,
            crag_max_reformulations=self.crag_max_reformulations,
            crag_reformulation_model=self.crag_reformulation_model,
            llm_model=self.llm_model,
            max_response_tokens=self.max_response_tokens,
            temperature=self.temperature,
        )

        self._pipeline = CompleteRAGPipeline(self._config)
        return self._pipeline.initialize_pipeline()

    def ingest(self) -> Dict[str, Any]:
        """Process PDFs, chunk, and store embeddings in the vector DB."""
        assert self._pipeline is not None, "Call initialize() first"
        return self._pipeline.ingest_documents()

    def setup(self) -> Dict[str, Any]:
        """Convenience method: initialize components and ingest documents.

        Returns the ingestion result dictionary.
        """
        self.initialize()
        return self.ingest()

    
    def query(self, question: str, conversation_id: Optional[str] = None, save_results: bool = True) -> RAGResponse:
        """Run a complete RAG query and return a structured response."""
        assert self._pipeline is not None, "Call initialize() first"
        return self._pipeline.query_pipeline(
            query=question,
            conversation_id=conversation_id,
            save_results=save_results,
        )

    def stats(self) -> Dict[str, Any]:
        """Return current pipeline statistics."""
        assert self._pipeline is not None, "Call initialize() first"
        return self._pipeline.get_pipeline_stats()

    def save_state(self, filepath: Optional[str] = None) -> None:
        """Persist config and stats for reproducibility and auditing."""
        assert self._pipeline is not None, "Call initialize() first"
        self._pipeline.save_pipeline_state(filepath)

    @classmethod
    def create_default(cls, documents_dir: str = "./documents", working_dir: str = "./rag_working_dir") -> "DingDongRAG":
        """Create with CLI-like defaults for a quick start."""
        return cls(
            documents_dir=documents_dir,
            working_dir=working_dir,
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            vector_store_type="chroma",
            retrieval_top_k=75,
            reranking_strategy="hybrid",
            reranking_top_k=20,
            llm_model="gpt-4o",
            max_response_tokens=2000,
            temperature=0.6,
        )


