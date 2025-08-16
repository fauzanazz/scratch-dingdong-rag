try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import re
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import openai
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import SearchResult, VectorStore
from ..embeddings.embedding_config import EmbeddingFunction


class CRAGTriggerMode(Enum):
    """Modes for triggering CRAG refinement."""
    ALWAYS = "always"
    SCORE = "score"
    TOKEN_COVERAGE = "token_coverage"
    NLI = "nli"
    HYBRID = "hybrid"


@dataclass
class CRAGRefinementConfig:
    """Configuration for CRAG refinement process."""
    
    # Core enable/disable
    enable_crag: bool = False
    
    # Trigger configuration
    trigger_mode: CRAGTriggerMode = CRAGTriggerMode.HYBRID
    min_mean_similarity: float = 0.22
    min_token_coverage: float = 0.35
    min_score_variance_threshold: float = 0.05  # Low variance indicates poor retrieval
    
    # Query reformulation
    max_reformulations: int = 3
    enable_subquestions: bool = True
    reformulation_model: str = "gpt-4o-mini"
    preserve_domain_terms: bool = True  # Preserve technical terms
    
    # Retrieval expansion
    max_expanded_results: int = 120  # 3x typical top_k
    dedup_similarity_threshold: float = 0.92
    
    # Optional NLI filtering
    enable_nli_filter: bool = False
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_min_entailment: float = 0.6
    
    # Optional web search (disabled by default)
    enable_web_fetch: bool = False
    web_provider: str = "tavily"  # tavily, serpapi
    web_api_key: Optional[str] = None
    max_web_snippets: int = 5
    
    # Safety and budget controls
    max_iterations: int = 1
    max_latency_ms: int = 10000  # 10 seconds max
    max_llm_calls: int = 5
    max_tokens_per_call: int = 1000
    
    # Indonesian-specific configuration
    indonesian_synonyms: Dict[str, List[str]] = field(default_factory=lambda: {
        'matematika': ['math', 'mathematics', 'ilmu hitung'],
        'algoritma': ['algorithm', 'prosedur', 'langkah'],
        'graf': ['graph', 'diagram', 'jaringan'],
        'fungsi': ['function', 'pemetaan'],
        'himpunan': ['set', 'kumpulan'],
        'komputer': ['computer', 'sistem', 'mesin'],
        'data': ['information', 'informasi', 'datum'],
    })
    
    # Cache configuration
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour


@dataclass
class EvidenceQualityMetrics:
    """Metrics for assessing evidence quality."""
    mean_similarity: float
    max_similarity: float
    min_similarity: float
    score_variance: float
    token_coverage: float
    unique_sources: int
    total_results: int
    trigger_refinement: bool
    trigger_reasons: List[str]


@dataclass
class CRAGRefinementResult:
    """Result from CRAG refinement process."""
    original_query: str
    reformulations: List[str]
    initial_results: List[SearchResult]
    expanded_results: List[SearchResult]
    final_results: List[SearchResult]
    evidence_metrics: EvidenceQualityMetrics
    refinement_applied: bool
    latency_ms: float
    llm_calls_used: int
    tokens_used: int
    debug_info: Dict[str, Any]


class CRAGRefinement:
    """
    CRAG refinement engine that improves retrieval quality through:
    1. Evidence quality assessment
    2. Query reformulation and expansion
    3. Expanded retrieval with deduplication
    4. Optional NLI filtering and web search
    """
    
    def __init__(self, 
                 config: CRAGRefinementConfig,
                 vector_store: VectorStore,
                 embedding_function: EmbeddingFunction):
        self.config = config
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.cache = {}
        self._setup_reformulation_client()
        
    def _setup_reformulation_client(self):
        """Setup OpenAI client for query reformulation."""
        if 'gpt' in self.config.reformulation_model.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and self.config.enable_crag:
                raise ValueError("OpenAI API key required for CRAG query reformulation")
            openai.api_key = api_key
    
    def refine(self, query: str, initial_results: List[SearchResult], top_k: int = 10) -> CRAGRefinementResult:
        """
        Main CRAG refinement pipeline.
        
        Args:
            query: Original user query
            initial_results: Results from initial vector search
            top_k: Final number of results to return
            
        Returns:
            CRAGRefinementResult with refined results and metrics
        """
        start_time = time.time()
        debug_info = {"stages": []}
        
        if not self.config.enable_crag:
            return CRAGRefinementResult(
                original_query=query,
                reformulations=[],
                initial_results=initial_results,
                expanded_results=initial_results,
                final_results=initial_results[:top_k],
                evidence_metrics=self._assess_evidence_quality(query, initial_results),
                refinement_applied=False,
                latency_ms=0,
                llm_calls_used=0,
                tokens_used=0,
                debug_info={"reason": "CRAG disabled"}
            )
        
        # Step 1: Assess evidence quality
        debug_info["stages"].append("evidence_assessment")
        evidence_metrics = self._assess_evidence_quality(query, initial_results)
        
        # Step 2: Decide if refinement is needed
        if not evidence_metrics.trigger_refinement:
            latency_ms = (time.time() - start_time) * 1000
            debug_info["skip_reason"] = f"Quality sufficient: {evidence_metrics.trigger_reasons}"
            return CRAGRefinementResult(
                original_query=query,
                reformulations=[],
                initial_results=initial_results,
                expanded_results=initial_results,
                final_results=initial_results[:top_k],
                evidence_metrics=evidence_metrics,
                refinement_applied=False,
                latency_ms=latency_ms,
                llm_calls_used=0,
                tokens_used=0,
                debug_info=debug_info
            )
        
        # Step 3: Query reformulation
        debug_info["stages"].append("query_reformulation")
        reformulations = self._reformulate_query(query)
        debug_info["reformulations"] = reformulations
        
        # Step 4: Expanded retrieval
        debug_info["stages"].append("expanded_retrieval")
        expanded_results = self._expanded_retrieval(query, reformulations, initial_results)
        
        # Step 5: Deduplication
        debug_info["stages"].append("deduplication")
        deduplicated_results = self._deduplicate_results(expanded_results)
        
        # Step 6: Optional NLI filtering
        if self.config.enable_nli_filter:
            debug_info["stages"].append("nli_filtering")
            deduplicated_results = self._nli_filter(query, deduplicated_results)
        
        # Step 7: Optional web search (if still insufficient)
        if self.config.enable_web_fetch and len(deduplicated_results) < top_k:
            debug_info["stages"].append("web_search")
            web_results = self._web_search_fallback(query)
            deduplicated_results.extend(web_results)
        
        # Final results
        final_results = deduplicated_results[:top_k]
        latency_ms = (time.time() - start_time) * 1000
        
        return CRAGRefinementResult(
            original_query=query,
            reformulations=reformulations,
            initial_results=initial_results,
            expanded_results=expanded_results,
            final_results=final_results,
            evidence_metrics=evidence_metrics,
            refinement_applied=True,
            latency_ms=latency_ms,
            llm_calls_used=len(reformulations) + 1,  # Rough estimate
            tokens_used=self._estimate_tokens_used(query, reformulations),
            debug_info=debug_info
        )
    
    def _assess_evidence_quality(self, query: str, results: List[SearchResult]) -> EvidenceQualityMetrics:
        """
        Assess quality of initial retrieval results to determine if refinement is needed.
        
        Uses multiple signals:
        1. Score statistics (mean, variance)
        2. Token coverage (query terms present in results)
        3. Source diversity
        """
        if not results:
            return EvidenceQualityMetrics(
                mean_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                score_variance=0.0,
                token_coverage=0.0,
                unique_sources=0,
                total_results=0,
                trigger_refinement=True,
                trigger_reasons=["no_results"]
            )
        
        # Score statistics
        scores = [r.score for r in results]
        mean_sim = np.mean(scores)
        max_sim = np.max(scores)
        min_sim = np.min(scores)
        score_variance = np.var(scores)
        
        # Token coverage analysis
        token_coverage = self._calculate_token_coverage(query, results)
        
        # Source diversity
        unique_sources = len(set(r.metadata.get('source', 'unknown') for r in results))
        
        # Determine if refinement should be triggered
        trigger_reasons = []
        trigger_refinement = False
        
        if self.config.trigger_mode == CRAGTriggerMode.ALWAYS:
            trigger_refinement = True
            trigger_reasons.append("always_mode")
        else:
            # Score-based triggers
            if self.config.trigger_mode in [CRAGTriggerMode.SCORE, CRAGTriggerMode.HYBRID]:
                if mean_sim < self.config.min_mean_similarity:
                    trigger_refinement = True
                    trigger_reasons.append(f"low_mean_similarity_{mean_sim:.3f}")
                
                if score_variance < self.config.min_score_variance_threshold:
                    trigger_refinement = True
                    trigger_reasons.append(f"low_score_variance_{score_variance:.3f}")
            
            # Token coverage trigger
            if self.config.trigger_mode in [CRAGTriggerMode.TOKEN_COVERAGE, CRAGTriggerMode.HYBRID]:
                if token_coverage < self.config.min_token_coverage:
                    trigger_refinement = True
                    trigger_reasons.append(f"low_token_coverage_{token_coverage:.3f}")
        
        return EvidenceQualityMetrics(
            mean_similarity=mean_sim,
            max_similarity=max_sim,
            min_similarity=min_sim,
            score_variance=score_variance,
            token_coverage=token_coverage,
            unique_sources=unique_sources,
            total_results=len(results),
            trigger_refinement=trigger_refinement,
            trigger_reasons=trigger_reasons
        )
    
    def _calculate_token_coverage(self, query: str, results: List[SearchResult]) -> float:
        """Calculate what fraction of query tokens appear in the top results."""
        if not results:
            return 0.0
        
        # Simple tokenization (could be enhanced with proper tokenizer)
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        if not query_tokens:
            return 1.0
        
        # Check top 5 results for token presence
        top_results_text = " ".join([r.content for r in results[:5]]).lower()
        
        covered_tokens = 0
        for token in query_tokens:
            if token in top_results_text:
                covered_tokens += 1
        
        return covered_tokens / len(query_tokens)
    
    def _reformulate_query(self, query: str) -> List[str]:
        """
        Generate query reformulations using LLM and rule-based approaches.
        
        Strategies:
        1. LLM-based rewriting for semantic variants
        2. Indonesian synonym expansion
        3. Subquestion decomposition
        """
        reformulations = []
        
        # LLM-based reformulation
        if 'gpt' in self.config.reformulation_model.lower():
            llm_reformulations = self._llm_reformulate(query)
            reformulations.extend(llm_reformulations)
        
        # Rule-based Indonesian expansion
        indonesian_variants = self._indonesian_synonym_expansion(query)
        reformulations.extend(indonesian_variants)
        
        # Remove duplicates and limit count
        unique_reformulations = list(dict.fromkeys(reformulations))  # Preserve order
        return unique_reformulations[:self.config.max_reformulations]
    
    def _llm_reformulate(self, query: str) -> List[str]:
        """Use LLM to generate semantic reformulations of the query."""
        
        system_prompt = """You are an expert at reformulating search queries for academic documents, particularly in Indonesian and English. Your task is to generate alternative phrasings that capture the same semantic intent.

Guidelines:
1. Preserve technical terms and domain-specific vocabulary
2. Generate variants that might match different document styles
3. Include both Indonesian and English variants when appropriate
4. Keep reformulations concise and focused
5. Avoid completely changing the meaning or scope"""

        user_prompt = f"""Original query: "{query}"

Generate 2-3 alternative reformulations that would help find relevant academic content. Focus on:
- Different ways to phrase the same question
- Alternative technical terminology
- Both formal and informal variants

Return only the reformulations, one per line, without numbering or explanation."""

        try:
            response = openai.chat.completions.create(
                model=self.config.reformulation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens_per_call,
                temperature=0.7
            )
            
            reformulations_text = response.choices[0].message.content.strip()
            reformulations = [line.strip() for line in reformulations_text.split('\n') if line.strip()]
            
            # Filter out reformulations that are too similar to original
            filtered_reformulations = []
            for ref in reformulations:
                if ref.lower() != query.lower() and len(ref.strip()) > 5:
                    filtered_reformulations.append(ref)
            
            return filtered_reformulations[:2]  # Limit to 2 LLM reformulations
            
        except Exception as e:
            print(f"Warning: LLM reformulation failed: {e}")
            return []
    
    def _indonesian_synonym_expansion(self, query: str) -> List[str]:
        """Generate Indonesian synonym-based query variants."""
        variants = []
        query_lower = query.lower()
        
        # Simple synonym replacement
        for term, synonyms in self.config.indonesian_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    variant = query_lower.replace(term, synonym)
                    if variant != query_lower:
                        variants.append(variant)
        
        return variants[:2]  # Limit Indonesian variants
    
    def _expanded_retrieval(self, original_query: str, reformulations: List[str], 
                          initial_results: List[SearchResult]) -> List[SearchResult]:
        """
        Perform expanded retrieval using reformulated queries.
        """
        all_results = list(initial_results)  # Start with initial results
        seen_chunk_ids = set(r.chunk_id for r in initial_results)
        
        # Retrieve for each reformulation
        for reformulation in reformulations:
            try:
                # Use a higher top_k for reformulations to get more diversity
                reformulation_results = self.vector_store.search(
                    reformulation, 
                    top_k=min(50, self.config.max_expanded_results // len(reformulations) + 20)
                )
                
                # Add new results (avoiding duplicates by chunk_id)
                for result in reformulation_results:
                    if result.chunk_id not in seen_chunk_ids:
                        all_results.append(result)
                        seen_chunk_ids.add(result.chunk_id)
                        
                        # Respect the max expanded results limit
                        if len(all_results) >= self.config.max_expanded_results:
                            break
                            
            except Exception as e:
                print(f"Warning: Expanded retrieval failed for '{reformulation}': {e}")
                continue
        
        return all_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove near-duplicate results using embedding similarity.
        """
        if len(results) <= 1:
            return results
        
        # Get embeddings for all result contents
        contents = [r.content for r in results]
        try:
            embeddings = self.embedding_function.embed_text(contents)
            if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 1:
                embeddings = [embeddings]
        except Exception as e:
            print(f"Warning: Could not embed for deduplication: {e}")
            return results
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Mark duplicates
        keep_indices = []
        for i in range(len(results)):
            is_duplicate = False
            for j in range(i):
                if similarity_matrix[i][j] > self.config.dedup_similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
        
        return [results[i] for i in keep_indices]
    
    def _nli_filter(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Optional NLI-based filtering (placeholder for now)."""
        # This would require implementing or importing an NLI model
        # For now, just return the results as-is
        return results
    
    def _web_search_fallback(self, query: str) -> List[SearchResult]:
        """Optional web search fallback (placeholder for now)."""
        # This would integrate with web search APIs
        # For now, return empty list
        return []
    
    def _estimate_tokens_used(self, query: str, reformulations: List[str]) -> int:
        """Rough estimate of tokens used in LLM calls."""
        total_text = query + " ".join(reformulations)
        return len(total_text.split()) * 1.3  # Rough approximation


# Factory functions for creating CRAG configurations

def create_minimal_crag_config() -> CRAGRefinementConfig:
    """Create minimal CRAG configuration for basic refinement."""
    return CRAGRefinementConfig(
        enable_crag=True,
        trigger_mode=CRAGTriggerMode.HYBRID,
        max_reformulations=2,
        max_expanded_results=60,  # 2x typical top_k
        enable_nli_filter=False,
        enable_web_fetch=False,
        reformulation_model="gpt-4o-mini"
    )


def create_production_crag_config() -> CRAGRefinementConfig:
    """Create production CRAG configuration with all features."""
    return CRAGRefinementConfig(
        enable_crag=True,
        trigger_mode=CRAGTriggerMode.HYBRID,
        max_reformulations=3,
        max_expanded_results=120,
        enable_nli_filter=True,
        enable_web_fetch=False,  # Disable by default for security
        reformulation_model="gpt-4o",
        max_latency_ms=15000,  # 15 seconds for production
        max_llm_calls=8
    )


def create_fast_crag_config() -> CRAGRefinementConfig:
    """Create fast CRAG configuration optimized for low latency."""
    return CRAGRefinementConfig(
        enable_crag=True,
        trigger_mode=CRAGTriggerMode.SCORE,  # Faster than hybrid
        max_reformulations=1,
        max_expanded_results=40,
        enable_nli_filter=False,
        enable_web_fetch=False,
        reformulation_model="gpt-4o-mini",
        max_latency_ms=5000,  # 5 seconds max
        max_llm_calls=2
    )


def create_crag_refinement_engine(
    config: CRAGRefinementConfig,
    vector_store: VectorStore,
    embedding_function: EmbeddingFunction
) -> CRAGRefinement:
    """Factory function to create CRAG refinement engine."""
    return CRAGRefinement(config, vector_store, embedding_function)