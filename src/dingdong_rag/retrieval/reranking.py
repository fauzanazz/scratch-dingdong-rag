import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere = None

from .vector_store import SearchResult


@dataclass
class RerankingConfig:
    """Configuration for reranking strategies."""
    strategy: str = "cross_encoder"  # cross_encoder, bm25, hybrid, cohere
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieve: int = 50  # Retrieve more for reranking
    top_k_final: int = 10     # Final number after reranking
    
    # Cohere settings
    cohere_model: str = "rerank-english-v3.0"
    cohere_api_key: Optional[str] = None
    
    # Hybrid weighting
    cross_encoder_weight: float = 0.7
    bm25_weight: float = 0.2
    semantic_weight: float = 0.1
    cohere_weight: float = 0.0  # Used in hybrid mode when Cohere is enabled
    
    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # Performance settings
    batch_size: int = 32
    cache_models: bool = True


@dataclass
class RerankingResult:
    """Result from reranking process."""
    search_result: SearchResult
    original_rank: int
    new_rank: int
    reranking_score: float
    score_breakdown: Dict[str, float]  # Individual component scores


class RerankingStrategy(ABC):
    """Abstract base class for reranking strategies."""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
    
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank search results based on relevance to query."""
        pass


class CrossEncoderReranker(RerankingStrategy):
    """Cross-encoder based reranking using sentence transformers."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize cross-encoder model."""
        try:
            self.model = CrossEncoder(self.config.cross_encoder_model)
            print(f"✓ Loaded cross-encoder model: {self.config.cross_encoder_model}")
        except Exception as e:
            print(f"✗ Failed to load cross-encoder model: {e}")
            self.model = None
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank using cross-encoder scores."""
        if not self.model or not results:
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score}
                )
                for i, result in enumerate(results)
            ]
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [(query, result.content) for result in results]
            
            # Get cross-encoder scores
            cross_encoder_scores = self.model.predict(query_doc_pairs)
            
            # Create reranking results
            reranking_results = []
            for i, (result, ce_score) in enumerate(zip(results, cross_encoder_scores)):
                rerank_result = RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=0,  # Will be set after sorting
                    reranking_score=float(ce_score),
                    score_breakdown={
                        'cross_encoder': float(ce_score),
                        'original': result.score
                    }
                )
                reranking_results.append(rerank_result)
            
            # Sort by cross-encoder score (descending)
            reranking_results.sort(key=lambda result: result.reranking_score, reverse=True)
            
            # Update new ranks
            for new_rank, result in enumerate(reranking_results):
                result.new_rank = new_rank
            
            # Return top-k
            return reranking_results[:self.config.top_k_final]
            
        except Exception as e:
            print(f"✗ Cross-encoder reranking failed: {e}")
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score, 'error': str(e)}
                )
                for i, result in enumerate(results[:self.config.top_k_final])
            ]


class BM25Reranker(RerankingStrategy):
    """BM25 based reranking using lexical matching."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.bm25 = None
    
    def _prepare_bm25(self, documents: List[str]):
        """Prepare BM25 index from documents."""
        try:
            # Simple tokenization (could be improved with proper tokenizer)
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"✗ BM25 preparation failed: {e}")
            self.bm25 = None
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank using BM25 scores."""
        if not results:
            return []
        
        try:
            # Prepare documents and BM25 index with metadata enrichment
            documents = []
            for result in results:
                # Enrich document text with metadata for better contextual scoring
                metadata_text = " ".join(str(v) for v in result.metadata.values() if v and str(v).strip())
                enriched_text = result.content
                if metadata_text:
                    enriched_text += " " + metadata_text
                documents.append(enriched_text)
            self._prepare_bm25(documents)
            
            if not self.bm25:
                # Fallback to original ordering
                return [
                    RerankingResult(
                        search_result=result,
                        original_rank=i,
                        new_rank=i,
                        reranking_score=result.score,
                        score_breakdown={'original': result.score}
                    )
                    for i, result in enumerate(results)
                ]
            
            # Get BM25 scores
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Create reranking results
            reranking_results = []
            for i, (result, bm25_score) in enumerate(zip(results, bm25_scores)):
                rerank_result = RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=0,  # Will be set after sorting
                    reranking_score=float(bm25_score),
                    score_breakdown={
                        'bm25': float(bm25_score),
                        'original': result.score
                    }
                )
                reranking_results.append(rerank_result)
            
            # Sort by BM25 score (descending)
            reranking_results.sort(key=lambda result: result.reranking_score, reverse=True)
            
            # Update new ranks
            for new_rank, result in enumerate(reranking_results):
                result.new_rank = new_rank
            
            # Return top-k
            return reranking_results[:self.config.top_k_final]
            
        except Exception as e:
            print(f"✗ BM25 reranking failed: {e}")
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score, 'error': str(e)}
                )
                for i, result in enumerate(results[:self.config.top_k_final])
            ]


class CohereReranker(RerankingStrategy):
    """Cohere reranking using Cohere's rerank API."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client."""
        if not COHERE_AVAILABLE:
            print("✗ Cohere library not available. Install with: pip install cohere")
            return
        
        try:
            # Get API key from config or environment
            api_key = self.config.cohere_api_key or os.getenv("COHERE_API_KEY")
            
            if not api_key:
                print("✗ Cohere API key not found. Set COHERE_API_KEY environment variable or provide in config")
                return
            
            self.client = cohere.Client(api_key)
            print(f"✓ Initialized Cohere client with model: {self.config.cohere_model}")
            
        except Exception as e:
            print(f"✗ Failed to initialize Cohere client: {e}")
            self.client = None
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank using Cohere's rerank API."""
        if not self.client or not results:
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score}
                )
                for i, result in enumerate(results)
            ]
        
        try:
            # Prepare documents for Cohere API
            documents = [result.content for result in results]
            
            # Call Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.config.cohere_model,
                top_n=min(self.config.top_k_final, len(results))
            )
            
            # Create reranking results
            reranking_results = []
            
            # Process reranked results
            for cohere_result in response.results:
                original_index = cohere_result.index
                original_result = results[original_index]
                
                rerank_result = RerankingResult(
                    search_result=original_result,
                    original_rank=original_index,
                    new_rank=len(reranking_results),  # New rank based on Cohere ordering
                    reranking_score=float(cohere_result.relevance_score),
                    score_breakdown={
                        'cohere': float(cohere_result.relevance_score),
                        'original': original_result.score
                    }
                )
                reranking_results.append(rerank_result)
            
            return reranking_results
            
        except Exception as e:
            print(f"✗ Cohere reranking failed: {e}")
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score, 'error': str(e)}
                )
                for i, result in enumerate(results[:self.config.top_k_final])
            ]


class HybridReranker(RerankingStrategy):
    """Hybrid reranking combining cross-encoder, BM25, semantic scores, and optionally Cohere."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.cross_encoder_reranker = CrossEncoderReranker(config)
        self.bm25_reranker = BM25Reranker(config)
        
        # Initialize Cohere reranker if weight > 0
        self.cohere_reranker = None
        if config.cohere_weight > 0:
            self.cohere_reranker = CohereReranker(config)
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank using hybrid scoring."""
        if not results:
            return []
        
        try:
            # Get cross-encoder results (but don't limit to top-k yet)
            config_copy = RerankingConfig(**asdict(self.config))
            config_copy.top_k_final = len(results)  # Get all results
            
            ce_reranker = CrossEncoderReranker(config_copy)
            ce_results = ce_reranker.rerank(query, results)
            
            # Get BM25 results
            bm25_reranker = BM25Reranker(config_copy)
            bm25_results = bm25_reranker.rerank(query, results)
            
            # Get Cohere results if enabled
            cohere_results = []
            if self.cohere_reranker and self.config.cohere_weight > 0:
                cohere_results = self.cohere_reranker.rerank(query, results)
            
            # Combine scores using weighted average
            combined_results = []
            
            for i, result in enumerate(results):
                # Find corresponding scores from each method
                ce_score = 0.0
                bm25_score = 0.0
                cohere_score = 0.0
                
                # Find cross-encoder score
                for ce_result in ce_results:
                    if ce_result.search_result.chunk_id == result.chunk_id:
                        ce_score = ce_result.reranking_score
                        break
                
                # Find BM25 score
                for bm25_result in bm25_results:
                    if bm25_result.search_result.chunk_id == result.chunk_id:
                        bm25_score = bm25_result.reranking_score
                        break
                
                # Find Cohere score
                if cohere_results:
                    for cohere_result in cohere_results:
                        if cohere_result.search_result.chunk_id == result.chunk_id:
                            cohere_score = cohere_result.reranking_score
                            break
                
                # Normalize scores (simple min-max normalization)
                semantic_score = result.score  # Original semantic similarity
                
                # Combine scores with weights
                combined_score = (
                    self.config.cross_encoder_weight * ce_score +
                    self.config.bm25_weight * bm25_score +
                    self.config.semantic_weight * semantic_score +
                    self.config.cohere_weight * cohere_score
                )
                
                score_breakdown = {
                    'cross_encoder': ce_score,
                    'bm25': bm25_score,
                    'semantic': semantic_score,
                    'combined': combined_score
                }
                
                # Add Cohere score to breakdown if used
                if self.config.cohere_weight > 0:
                    score_breakdown['cohere'] = cohere_score
                
                combined_result = RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=0,  # Will be set after sorting
                    reranking_score=combined_score,
                    score_breakdown=score_breakdown
                )
                combined_results.append(combined_result)
            
            # Sort by combined score (descending)
            combined_results.sort(key=lambda result: result.reranking_score, reverse=True)
            
            # Update new ranks
            for new_rank, result in enumerate(combined_results):
                result.new_rank = new_rank
            
            # Return top-k
            return combined_results[:self.config.top_k_final]
            
        except Exception as e:
            print(f"✗ Hybrid reranking failed: {e}")
            # Fallback to original ordering
            return [
                RerankingResult(
                    search_result=result,
                    original_rank=i,
                    new_rank=i,
                    reranking_score=result.score,
                    score_breakdown={'original': result.score, 'error': str(e)}
                )
                for i, result in enumerate(results[:self.config.top_k_final])
            ]


class RerankingPipeline:
    """Complete reranking pipeline with multiple strategies."""
    
    def __init__(self, config: RerankingConfig = None):
        self.config = config or RerankingConfig()
        self.reranker = self._create_reranker()
        self.reranking_history = []
    
    def _create_reranker(self) -> RerankingStrategy:
        """Create reranker based on configuration."""
        if self.config.strategy == "cross_encoder":
            return CrossEncoderReranker(self.config)
        elif self.config.strategy == "bm25":
            return BM25Reranker(self.config)
        elif self.config.strategy == "cohere":
            return CohereReranker(self.config)
        elif self.config.strategy == "hybrid":
            return HybridReranker(self.config)
        else:
            raise ValueError(f"Unknown reranking strategy: {self.config.strategy}")
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[RerankingResult]:
        """Rerank search results."""
        start_time = time.time()
        
        # Perform reranking
        reranked_results = self.reranker.rerank(query, results)
        
        reranking_time = time.time() - start_time
        
        # Store in history
        self.reranking_history.append({
            'query': query,
            'original_count': len(results),
            'reranked_count': len(reranked_results),
            'reranking_time': reranking_time,
            'strategy': self.config.strategy,
            'timestamp': time.time()
        })
        
        print(f"Reranked {len(results)} → {len(reranked_results)} results in {reranking_time:.2f}s")
        
        return reranked_results
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking performance statistics."""
        if not self.reranking_history:
            return {"total_rerankings": 0}
        
        total_time = sum(h['reranking_time'] for h in self.reranking_history)
        avg_time = total_time / len(self.reranking_history)
        
        return {
            'total_rerankings': len(self.reranking_history),
            'avg_reranking_time': avg_time,
            'total_reranking_time': total_time,
            'strategy': self.config.strategy,
            'config': asdict(self.config)
        }
    
    def analyze_reranking_impact(self, reranked_results: List[RerankingResult]) -> Dict[str, Any]:
        """Analyze the impact of reranking on result ordering."""
        if not reranked_results:
            return {}
        
        # Calculate rank changes
        rank_changes = []
        score_improvements = []
        
        for result in reranked_results:
            rank_change = result.original_rank - result.new_rank
            rank_changes.append(rank_change)
            
            # Score improvement (if available)
            original_score = result.score_breakdown.get('original', 0)
            if original_score > 0:
                score_improvement = (result.reranking_score - original_score) / original_score
                score_improvements.append(score_improvement)
        
        analysis = {
            'total_results': len(reranked_results),
            'avg_rank_change': np.mean(rank_changes) if rank_changes else 0,
            'max_rank_improvement': max(rank_changes) if rank_changes else 0,
            'max_rank_degradation': min(rank_changes) if rank_changes else 0,
            'results_improved': sum(1 for change in rank_changes if change > 0),
            'results_degraded': sum(1 for change in rank_changes if change < 0),
            'results_unchanged': sum(1 for change in rank_changes if change == 0),
        }
        
        if score_improvements:
            analysis.update({
                'avg_score_improvement': np.mean(score_improvements),
                'max_score_improvement': max(score_improvements),
                'min_score_improvement': min(score_improvements)
            })
        
        return analysis


def create_reranking_pipeline(
    strategy: str = "cross_encoder",
    top_k_retrieve: int = 50,
    top_k_final: int = 10,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> RerankingPipeline:
    """Factory function to create reranking pipeline."""
    
    config = RerankingConfig(
        strategy=strategy,
        top_k_retrieve=top_k_retrieve,
        top_k_final=top_k_final,
        cross_encoder_model=cross_encoder_model
    )
    
    return RerankingPipeline(config)


def create_production_reranker() -> RerankingPipeline:
    """Create production-ready hybrid reranker."""
    
    config = RerankingConfig(
        strategy="hybrid",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Better model
        top_k_retrieve=100,
        top_k_final=10,
        cross_encoder_weight=0.6,
        bm25_weight=0.3,
        semantic_weight=0.1
    )
    
    return RerankingPipeline(config)


def create_fast_reranker() -> RerankingPipeline:
    """Create fast reranker for development."""
    
    config = RerankingConfig(
        strategy="bm25",  # Fastest option
        top_k_retrieve=20,
        top_k_final=10
    )
    
    return RerankingPipeline(config)


def create_cohere_reranker(
    cohere_model: str = "rerank-english-v3.0",
    cohere_api_key: Optional[str] = None,
    top_k_retrieve: int = 50,
    top_k_final: int = 10
) -> RerankingPipeline:
    """Create Cohere reranker."""
    
    config = RerankingConfig(
        strategy="cohere",
        cohere_model=cohere_model,
        cohere_api_key=cohere_api_key,
        top_k_retrieve=top_k_retrieve,
        top_k_final=top_k_final
    )
    
    return RerankingPipeline(config)


def create_hybrid_with_cohere_reranker(
    cohere_weight: float = 0.4,
    cross_encoder_weight: float = 0.3,
    bm25_weight: float = 0.2,
    semantic_weight: float = 0.1,
    cohere_model: str = "rerank-english-v3.0",
    cohere_api_key: Optional[str] = None
) -> RerankingPipeline:
    """Create hybrid reranker including Cohere."""
    
    config = RerankingConfig(
        strategy="hybrid",
        cohere_model=cohere_model,
        cohere_api_key=cohere_api_key,
        cross_encoder_weight=cross_encoder_weight,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        cohere_weight=cohere_weight,
        top_k_retrieve=100,
        top_k_final=10
    )
    
    return RerankingPipeline(config)


def create_precision_hybrid_reranker(
    cross_encoder_weight: float = 0.8,
    bm25_weight: float = 0.2,
    semantic_weight: float = 0.0,
    cohere_weight: float = 0.0
) -> RerankingPipeline:
    """
    Create high-precision hybrid reranker that minimizes topic bleed.
    
    This configuration:
    - Heavily weights cross-encoder for semantic precision (0.8)
    - Uses BM25 for lexical matching (0.2) 
    - Disables semantic embedding score to prevent weak similarity bleed (0.0)
    - Optimal for academic/technical documents with clear domain boundaries
    """
    
    config = RerankingConfig(
        strategy="hybrid",
        cross_encoder_weight=cross_encoder_weight,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        cohere_weight=cohere_weight,
        top_k_retrieve=50,  # Reduced breadth for higher precision
        top_k_final=10
    )
    
    return RerankingPipeline(config)