from __future__ import annotations

import time
from typing import Any, Dict, List


def execute_query(
    query: str,
    config: Any,
    vector_store: Any,
    reranking_pipeline: Any,
    chat_engine: Any,
    crag_refinement: Any = None,
    conversation_id: str | None = None,
    save_results: bool = True,
) -> Dict[str, Any]:
    start_time = time.time()

    # 1) Retrieval
    retrieval_start = time.time()
    search_results = vector_store.search(query, top_k=config.retrieval_top_k)
    retrieval_time = time.time() - retrieval_start
    if not search_results:
        return {
            "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
            "conversation_id": conversation_id or "",
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "sources_used": [],
            "retrieval_time": retrieval_time,
            "reranking_time": 0.0,
            "completion_time": 0.0,
            "total_time": time.time() - start_time,
            "token_usage": {},
            "metadata": {"no_results": True},
        }

    # 2) Optional CRAG
    crag_time = 0.0
    if crag_refinement and getattr(config, "enable_crag", False):
        crag_start = time.time()
        try:
            crag_result = crag_refinement.refine(
                query=query, initial_results=search_results, top_k=config.retrieval_top_k * 2
            )
            search_results = crag_result.final_results
        except Exception:
            pass
        crag_time = time.time() - crag_start

    # 3) Reranking
    reranking_start = time.time()
    reranked_results = reranking_pipeline.rerank(query, search_results)
    reranking_time = time.time() - reranking_start

    # 4) Chat completion
    completion_start = time.time()
    chat_response = chat_engine.complete_chat(query=query, reranked_results=reranked_results, conversation_id=conversation_id)
    completion_time = time.time() - completion_start

    # 5) Persist
    result_id = None
    if save_results:
        try:
            result_id = vector_store.save_results(
                query=query,
                results=search_results,
                metadata={
                    "reranked_results": len(reranked_results),
                    "response": chat_response.message,
                    "sources_used": chat_response.sources_used,
                    "processing_time": time.time() - start_time,
                },
            )
        except Exception:
            pass

    total_time = time.time() - start_time
    return {
        "answer": getattr(chat_response, "message", ""),
        "conversation_id": getattr(chat_response, "conversation_id", "") or (conversation_id or ""),
        "retrieved_chunks": len(search_results),
        "reranked_chunks": len(reranked_results),
        "sources_used": getattr(chat_response, "sources_used", []),
        "retrieval_time": retrieval_time,
        "reranking_time": reranking_time,
        "completion_time": completion_time,
        "total_time": total_time,
        "token_usage": getattr(chat_response, "token_usage", {}),
        "metadata": {
            "result_id": result_id,
            "reranking_strategy": config.reranking_strategy,
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
            "crag_enabled": getattr(config, "enable_crag", False),
            "crag_time": crag_time,
        },
    }


