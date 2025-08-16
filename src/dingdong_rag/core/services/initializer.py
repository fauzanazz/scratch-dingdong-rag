from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...embeddings.embedding_config import EmbeddingFunction, create_embedding_function
from ...retrieval.vector_store import VectorStore, VectorStoreConfig, create_vector_store
from ...retrieval.reranking import RerankingPipeline, create_reranking_pipeline
from ...chat.chat_completion import ChatCompletionEngine, create_chat_completion_engine


@dataclass
class PipelineComponents:
    embedding_function: Optional[EmbeddingFunction] = None
    vector_store: Optional[VectorStore] = None
    crag_refinement: Optional[Any] = None
    reranking_pipeline: Optional[RerankingPipeline] = None
    chat_engine: Optional[ChatCompletionEngine] = None


def initialize_components(config: Any) -> Tuple[PipelineComponents, Dict[str, Any]]:
    steps_completed = []
    steps_failed = []
    start_time = time.time()

    components = PipelineComponents()

    # 1) Embedding function
    try:
        components.embedding_function = create_embedding_function(
            config.embedding_model,
            api_key=config.openai_api_key,
        )
        steps_completed.append("embedding_function")
    except Exception as e:
        steps_failed.append(f"embedding_function: {e}")
        return components, _summary(steps_completed, steps_failed, start_time)

    # 2) Vector store
    try:
        vs_config = VectorStoreConfig(
            store_type=config.vector_store_type,
            collection_name=config.collection_name,
            persist_directory=f"{config.working_dir}/vector_db",
            pinecone_api_key=config.pinecone_api_key,
            index_name=config.pinecone_index,
        )
        components.vector_store = create_vector_store(vs_config, components.embedding_function)
        if not components.vector_store.initialize():
            raise RuntimeError("Vector store initialization failed")
        steps_completed.append("vector_store")
    except Exception as e:
        steps_failed.append(f"vector_store: {e}")
        return components, _summary(steps_completed, steps_failed, start_time)

    # 3) CRAG refinement (optional)
    if getattr(config, "enable_crag", False):
        try:
            from ...retrieval.crag import CRAGRefinementConfig, CRAGRefinement, CRAGTriggerMode

            crag_config = CRAGRefinementConfig(
                enable_crag=True,
                trigger_mode=CRAGTriggerMode(config.crag_trigger_mode),
                min_mean_similarity=config.crag_min_similarity,
                min_token_coverage=config.crag_min_token_coverage,
                max_reformulations=config.crag_max_reformulations,
                reformulation_model=config.crag_reformulation_model,
                max_expanded_results=config.retrieval_top_k * 2,
            )
            components.crag_refinement = CRAGRefinement(
                crag_config, components.vector_store, components.embedding_function
            )
            steps_completed.append("crag_refinement")
        except Exception as e:
            steps_failed.append(f"crag_refinement: {e}")

    else:
        steps_completed.append("crag_refinement_skipped")

    # 4) Reranking pipeline
    try:
        components.reranking_pipeline = create_reranking_pipeline(
            strategy=config.reranking_strategy,
            top_k_retrieve=config.retrieval_top_k,
            top_k_final=config.reranking_top_k,
        )
        steps_completed.append("reranking_pipeline")
    except Exception as e:
        steps_failed.append(f"reranking_pipeline: {e}")
        return components, _summary(steps_completed, steps_failed, start_time)

    # 5) Chat engine
    try:
        components.chat_engine = create_chat_completion_engine(
            model=config.llm_model,
            max_tokens=config.max_response_tokens,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key,
        )
        steps_completed.append("chat_engine")
    except Exception as e:
        steps_failed.append(f"chat_engine: {e}")

    return components, _summary(steps_completed, steps_failed, start_time)


def _summary(steps_completed: list, steps_failed: list, start_time: float) -> Dict[str, Any]:
    return {
        "steps_completed": steps_completed,
        "steps_failed": steps_failed,
        "total_time": time.time() - start_time,
    }


