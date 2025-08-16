try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently fail if dotenv is not available

import os
import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from ..parsing.pdf_parser import process_documents_folder, ValidationConfig, GeminiConfig
from ..parsing.docling_adapter import DoclingConfig
from ..chunking.chunking import get_chunking_strategy, ChunkingConfig
from ..embeddings.embedding_config import create_embedding_function
from ..retrieval.vector_store import create_vector_store, VectorStoreConfig
from ..retrieval.reranking import create_reranking_pipeline
from ..chat.chat_completion import create_chat_completion_engine
from ..core.complete_rag_pipeline import CompleteRAGConfig, CompleteRAGPipeline, create_complete_rag_pipeline, create_production_rag_pipeline


def print_banner():
    """Print the DingDong RAG banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                           DingDong RAG Pipeline                              ║
║                    Hybrid Document Intelligence System                       ║
║                                                                              ║
║  PDF → Chunking → Vector DB → Reranking → Knowledge Graph → Chat             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_step_header(step_number: int, step_name: str, description: str = ""):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_name.upper()}")
    if description:
        print(f"Description: {description}")
    print(f"{'=' * 80}")


def print_success(message: str):
    """Print a success message."""
    print(f"SUCCESS: {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"ERROR: {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"WARNING: {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"INFO: {message}")


def step1_document_processing(
    documents_dir: str,
    working_dir: str,
    fallback_to_ocr: bool = True,
    enable_metadata: bool = True,
    ocr_method: str = "tesseract",
    gemini_config: Optional[GeminiConfig] = None,
    parser_method: str = "pymupdf",
    docling_config: Optional[DoclingConfig] = None,
) -> Dict[str, Any]:
    """
    Step 1: Process PDF documents with OCR fallback.
    """
    print_step_header(1, "Document Processing", "Extract text from PDF files with OCR fallback")
    
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        print_error(f"Documents directory does not exist: {documents_dir}")
        return {}
    
    print_info(f"Scanning for PDF files in {documents_dir} (recursive)")
    
    # Quick check for PDF files (recursive)
    pdf_files = list(documents_path.glob("**/*.pdf"))
    if not pdf_files:
        print_warning(f"No PDF files found recursively in {documents_dir}")
        return {}
    
    print_info(f"Found {len(pdf_files)} PDF files to process")
    
    # Configure validation
    validation_config = ValidationConfig(
        min_chars_per_page=50,
        min_word_ratio=0.3,
        max_symbol_ratio=0.3,
        min_alpha_ratio=0.5
    )
    
    start_time = time.time()
    
    try:
        results = process_documents_folder(
            documents_dir=documents_dir,
            fallback_to_ocr=fallback_to_ocr,
            validation_config=validation_config,
            extract_metadata=enable_metadata,
            ocr_method=ocr_method if ocr_method in ("tesseract", "gemini") else "tesseract",
            gemini_config=gemini_config,
            parser_method=parser_method,
            docling_config=docling_config,
        )
        
        processing_time = time.time() - start_time
        
        if results:
            print_success(f"Processed {len(results)} documents in {processing_time:.2f} seconds")
            
            # Save results
            output_path = Path(working_dir) / "processed_documents.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert results to serializable format
            serializable_results = {}
            for filename, doc_result in results.items():
                serializable_results[filename] = {
                    'content': doc_result.get('content', ''),
                    'metadata': doc_result.get('metadata', {}),
                    'processing_method': doc_result.get('processing_method', 'unknown'),
                    'word_count': len(doc_result.get('content', '').split()),
                    'char_count': len(doc_result.get('content', ''))
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
            print_info(f"Results saved to: {output_path}")
            
            # Print summary
            total_words = sum(doc['word_count'] for doc in serializable_results.values())
            total_chars = sum(doc['char_count'] for doc in serializable_results.values())
            print_info(f"Total extracted: {total_words} words, {total_chars} characters")
            
            return results
        else:
            print_error("No documents were successfully processed")
            return {}
            
    except Exception as e:
        print_error(f"Document processing failed: {str(e)}")
        return {}


def step2_chunking(documents: Dict[str, Any], working_dir: str, strategy: str = "recursive", 
                  chunk_size: int = 1200, chunk_overlap: int = 300,
                  chonkie_chunker_type: str = "word",
                  chonkie_tokenizer: str = "gpt2",
                  chonkie_embedding_model: str = "all-MiniLM-L6-v2",
                  chonkie_similarity_threshold: float = 0.5,
                  chonkie_initial_sentences: int = 1) -> List[Any]:
    """
    Step 2: Chunk documents using specified strategy.
    """
    print_step_header(2, "Document Chunking", f"Split documents into chunks using {strategy} strategy")
    
    if not documents:
        print_error("No documents to chunk")
        return []
    
    print_info(f"Chunking {len(documents)} documents with strategy: {strategy}")
    print_info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    if strategy == "chonkie":
        print_info(f"Chonkie chunker type: {chonkie_chunker_type}")
        if chonkie_chunker_type in ["semantic", "sdpm"]:
            print_info(f"Chonkie embedding model: {chonkie_embedding_model}")
            print_info(f"Chonkie similarity threshold: {chonkie_similarity_threshold}")
        if chonkie_chunker_type == "token":
            print_info(f"Chonkie tokenizer: {chonkie_tokenizer}")
        if chonkie_chunker_type == "sdpm":
            print_info(f"Chonkie initial sentences: {chonkie_initial_sentences}")
    
    # Configure chunking
    chunking_config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chonkie_chunker_type=chonkie_chunker_type,
        chonkie_tokenizer=chonkie_tokenizer,
        chonkie_embedding_model=chonkie_embedding_model,
        chonkie_similarity_threshold=chonkie_similarity_threshold,
        chonkie_initial_sentences=chonkie_initial_sentences
    )
    
    chunking_strategy = get_chunking_strategy(strategy, chunking_config)
    
    start_time = time.time()
    all_chunks = []
    
    try:
        for filename, doc_data in documents.items():
            content = doc_data.get('content', '')
            if not content.strip():
                print_warning(f"Skipping empty document: {filename}")
                continue
                
            print_info(f"Chunking: {filename}")
            chunks = chunking_strategy.chunk_text(content, filename)
            all_chunks.extend(chunks)
            print_info(f"  → Created {len(chunks)} chunks")
        
        chunking_time = time.time() - start_time
        
        if all_chunks:
            print_success(f"Created {len(all_chunks)} total chunks in {chunking_time:.2f} seconds")
            
            # Save chunks
            output_path = Path(working_dir) / "chunks.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to serializable format
            serializable_chunks = []
            for chunk in all_chunks:
                chunk_data = {
                    'content': getattr(chunk, 'content', str(chunk)),
                    'metadata': getattr(chunk, 'metadata', {}),
                    'word_count': len(getattr(chunk, 'content', str(chunk)).split()),
                    'char_count': len(getattr(chunk, 'content', str(chunk)))
                }
                serializable_chunks.append(chunk_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
                
            print_info(f"Chunks saved to: {output_path}")
            
            # Print statistics
            avg_chunk_size = sum(chunk['word_count'] for chunk in serializable_chunks) / len(serializable_chunks)
            print_info(f"Average chunk size: {avg_chunk_size:.1f} words")
            
            return all_chunks
        else:
            print_error("No chunks were created")
            return []
            
    except Exception as e:
        print_error(f"Chunking failed: {str(e)}")
        return []


def step3_embeddings_and_vector_store(chunks: List[Any], working_dir: str, 
                                     embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                     vector_store_type: str = "chroma",
                                     collection_name: str = "rag_collection") -> Any:
    """
    Step 3: Generate embeddings and store in vector database.
    """
    print_step_header(3, "Embeddings & Vector Storage", 
                      f"Generate embeddings using {embedding_model} and store in {vector_store_type}")
    
    if not chunks:
        print_error("No chunks to embed")
        return None
    
    print_info(f"Processing {len(chunks)} chunks")
    print_info(f"Embedding model: {embedding_model}")
    print_info(f"Vector store: {vector_store_type}")
    
    start_time = time.time()
    
    try:
        # Create embedding function
        print_info("Initializing embedding function...")
        embedding_function = create_embedding_function(
            model_name=embedding_model,
            api_key=os.getenv('OPENAI_API_KEY') if 'openai' in embedding_model else None
        )
        
        # Create vector store
        print_info("Initializing vector store...")
        vector_config = VectorStoreConfig(
            store_type=vector_store_type,
            collection_name=collection_name,
            persist_directory=str(Path(working_dir) / "vector_db"),
            pinecone_api_key=os.getenv('PINECONE_API_KEY') if vector_store_type == 'pinecone' else None
        )
        
        vector_store = create_vector_store(vector_config, embedding_function)
        # Initialize if not already initialized (handles fallback cases)
        if not (getattr(vector_store, 'initialized', False) or getattr(vector_store, 'collection', None)):
            init_success = vector_store.initialize()
            if not init_success:
                print_error("Vector store initialization failed")
                return None
        
        # Clear existing chunks to avoid duplicates
        print_info("Clearing existing chunks from vector store...")
        vector_store.clear_collection()
        
        # Add chunks to vector store
        print_info("Adding chunks to vector store...")
        vector_store.add_chunks(chunks)
        
        processing_time = time.time() - start_time
        print_success(f"Embedded and stored {len(chunks)} chunks in {processing_time:.2f} seconds")
        
        # Get vector store statistics
        try:
            stats = vector_store.get_stats()
            print_info(f"Vector store stats: {stats.total_vectors} vectors, {stats.storage_size_mb:.2f} MB")
        except:
            print_info("Vector store statistics not available")
        
        return vector_store
        
    except Exception as e:
        print_error(f"Embeddings and vector storage failed: {str(e)}")
        return None


def step4_setup_reranking(working_dir: str, strategy: str = "hybrid", top_k: int = 15, 
                          cohere_model: str = "rerank-english-v3.0", 
                          cohere_api_key: Optional[str] = None,
                          precision_mode: bool = False,
                          cross_encoder_weight: float = 0.7,
                          bm25_weight: float = 0.2,
                          semantic_weight: float = 0.1) -> Any:
    """
    Step 4: Setup reranking pipeline.
    """
    print_step_header(4, "Reranking Setup", f"Initialize {strategy} reranking with top-{top_k} results")
    
    print_info(f"Reranking strategy: {strategy}")
    print_info(f"Top-K results: {top_k}")
    
    if strategy == "cohere":
        print_info(f"Cohere model: {cohere_model}")
    elif strategy == "hybrid":
        if precision_mode:
            print_info("Using precision-tuned hybrid reranking")
            print_info("Weights: cross_encoder=0.8, bm25=0.2, semantic=0.0")
        else:
            print_info(f"Weights: cross_encoder={cross_encoder_weight}, bm25={bm25_weight}, semantic={semantic_weight}")
    
    try:
        if strategy == "cohere":
            from ..retrieval.reranking import create_cohere_reranker
            reranking_pipeline = create_cohere_reranker(
                cohere_model=cohere_model,
                cohere_api_key=cohere_api_key,
                top_k_retrieve=50,
                top_k_final=top_k
            )
        elif strategy == "hybrid" and precision_mode:
            from ..retrieval.reranking import create_precision_hybrid_reranker
            reranking_pipeline = create_precision_hybrid_reranker()
            reranking_pipeline.config.top_k_final = top_k
        elif strategy == "hybrid":
            from ..retrieval.reranking import RerankingConfig, RerankingPipeline
            config = RerankingConfig(
                strategy="hybrid",
                cross_encoder_weight=cross_encoder_weight,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                top_k_retrieve=50,
                top_k_final=top_k
            )
            reranking_pipeline = RerankingPipeline(config)
        else:
            reranking_pipeline = create_reranking_pipeline(
                strategy=strategy,
                top_k_retrieve=50,
                top_k_final=top_k
            )
        
        print_success("Reranking pipeline initialized successfully")
        return reranking_pipeline
        
    except Exception as e:
        print_error(f"Reranking setup failed: {str(e)}")
        return None


def step5_setup_chat_completion(working_dir: str, model: str = "gpt-4o", 
                               max_tokens: int = 2000, temperature: float = 0.6,
                               enable_query_enhancement: bool = True,
                               enable_context_compression: bool = True) -> Any:
    """
    Step 5: Setup chat completion engine.
    """
    print_step_header(5, "Chat Completion Setup", f"Initialize {model} with max_tokens={max_tokens}")
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key and 'gpt' in model:
        print_warning("OpenAI API key not found. Chat completion may not work for OpenAI models.")
    
    print_info(f"Model: {model}")
    print_info(f"Max tokens: {max_tokens}")
    print_info(f"Temperature: {temperature}")
    print_info(f"Query enhancement: {'enabled' if enable_query_enhancement else 'disabled'}")
    print_info(f"Context compression: {'enabled' if enable_context_compression else 'disabled'}")
    
    try:
        chat_engine = create_chat_completion_engine(
            model=model,
            openai_api_key=openai_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_query_enhancement=enable_query_enhancement,
            enable_auto_compression=enable_context_compression
        )
        
        print_success("Chat completion engine initialized successfully")
        return chat_engine
        
    except Exception as e:
        print_error(f"Chat completion setup failed: {str(e)}")
        return None




def interactive_query_session(vector_store: Any, reranking_pipeline: Any, chat_engine: Any,
                             retrieval_top_k: int = 75, reranking_top_k: int = 20, 
                             enable_crag: bool = False, crag_config: Optional[Dict] = None,
                             similarity_threshold: float = 0.0, domain_filter: Optional[str] = None,
                             auto_domain_detection: bool = True, domain_confidence_threshold: float = 0.3):
    """
    Interactive query session for testing the pipeline.
    """
    step_number = 6
    pipeline_type = "Vector RAG"
    print_step_header(step_number, "Interactive Query Session", f"Test the complete {pipeline_type} pipeline")
    
    print_info("You can now ask questions about your documents!")
    print_info("Type 'quit', 'exit', or 'q' to stop the session.")
    print_info("Type 'help' for available commands.")
    if auto_domain_detection:
        print_info("🎯 Automatic domain detection enabled - queries will be filtered by detected subject area")
    print("-" * 80)
    
    # Initialize domain detector
    domain_detector = None
    if auto_domain_detection:
        try:
            from ..retrieval.domain_detection import create_domain_detector
            domain_detector = create_domain_detector()
            print_info("Domain detector initialized successfully")
        except Exception as e:
            print_warning(f"Failed to initialize domain detector: {e}")
            auto_domain_detection = False
    
    query_count = 0
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print_info("Goodbye!")
                break
                
            if query.lower() == 'help':
                print_info("Available commands:")
                print("  - Ask any question about your documents")
                print("  - 'stats' - Show pipeline statistics")
                print("  - 'help' - Show this help")
                print("  - 'quit' or 'exit' or 'q' - Exit the session")
                continue
                
            if query.lower() == 'stats':
                print_info(f"Queries processed: {query_count}")
                try:
                    stats = vector_store.get_stats()
                    print_info(f"Vector store: {stats.total_vectors} vectors, {stats.storage_size_mb:.2f} MB")
                except:
                    print_info("Vector store statistics not available")
                continue
                
            if not query:
                print_warning("Please enter a question.")
                continue
            
            print_info(f"Processing query: {query}")
            start_time = time.time()
            
            # Step 1: Vector search
            # Determine domain filter: manual override or automatic detection
            active_domain_filter = domain_filter  # Start with manual filter
            detected_domain = None
            
            # Try automatic domain detection if enabled and no manual filter
            if auto_domain_detection and domain_detector and not domain_filter:
                detected_domain = domain_detector.detect_domain(query, domain_confidence_threshold)
                if detected_domain:
                    active_domain_filter = domain_detector.get_domain_filter(query, domain_confidence_threshold)
                    print_info(f"🎯 Detected domain: {detected_domain.course_code} ({detected_domain.confidence:.2f} confidence)")
                    print_info(f"📚 Keywords matched: {', '.join(detected_domain.matched_keywords[:3])}...")
            
            # Apply domain filtering
            search_kwargs = {"top_k": retrieval_top_k}
            if active_domain_filter:
                search_kwargs["filter_metadata"] = {"source_path": {"$contains": active_domain_filter}}
                filter_source = "detected" if detected_domain else "manual"
                print_info(f"🔍 Applying {filter_source} domain filter: {active_domain_filter}")
            
            search_results = vector_store.search(query, **search_kwargs)
            print_info(f"Found {len(search_results)} initial matches")
            
            # Apply similarity threshold filtering
            if similarity_threshold > 0.0:
                original_count = len(search_results)
                search_results = [r for r in search_results if r.score >= similarity_threshold]
                filtered_count = len(search_results)
                if filtered_count < original_count:
                    print_info(f"Filtered by similarity threshold ({similarity_threshold:.2f}): {original_count} → {filtered_count} results")
            
            # Step 1.5: CRAG refinement (if enabled)
            crag_applied = False
            if enable_crag and search_results:
                try:
                    from ..retrieval.crag import CRAGRefinement, CRAGRefinementConfig, CRAGTriggerMode
                    
                    # Create CRAG config from provided parameters
                    if crag_config is None:
                        crag_config = {}
                    
                    from ..retrieval.crag import CRAGRefinementConfig, CRAGTriggerMode
                    final_crag_config = CRAGRefinementConfig(
                        enable_crag=True,
                        trigger_mode=CRAGTriggerMode(crag_config.get('trigger_mode', 'hybrid')),
                        min_mean_similarity=crag_config.get('min_similarity', 0.22),
                        min_token_coverage=crag_config.get('min_token_coverage', 0.35),
                        max_reformulations=crag_config.get('max_reformulations', 2),
                        reformulation_model=crag_config.get('reformulation_model', 'gpt-4o-mini'),
                        max_expanded_results=100  # 2x the initial search
                    )
                    
                    # Create CRAG refinement engine - we need embedding function, skip CRAG if not available
                    try:
                        # Try to get embedding function from vector store
                        embedding_function = getattr(vector_store, 'embedding_function', None)
                        if not embedding_function:
                            print_warning("CRAG skipped: embedding function not available")
                            raise AttributeError("No embedding function")
                        crag_engine = CRAGRefinement(final_crag_config, vector_store, embedding_function)
                    except Exception:
                        print_warning("CRAG skipped: failed to initialize")
                        raise
                    
                    # Apply CRAG refinement
                    crag_result = crag_engine.refine(query, search_results, top_k=75)
                    
                    if crag_result.refinement_applied:
                        search_results = crag_result.final_results
                        crag_applied = True
                        print_info(f"CRAG applied: {len(crag_result.initial_results)} → {len(search_results)} results")
                        print_info(f"Reformulations: {', '.join(crag_result.reformulations[:2])}...")
                    else:
                        print_info(f"CRAG skipped: {', '.join(crag_result.evidence_metrics.trigger_reasons)}")
                        
                except Exception as e:
                    print_warning(f"CRAG refinement failed: {e}")
            
            # Step 2: Reranking
            if reranking_pipeline and search_results:
                reranked_results = reranking_pipeline.rerank(query, search_results)
                print_info(f"Reranked to top {len(reranked_results)} results")
                # Use the reranked results directly for chat completion
                final_reranked_results = reranked_results[:reranking_top_k]  # Use configured top-k
                final_results = [rr.search_result for rr in final_reranked_results]  # For display purposes
            else:
                # Create dummy RerankingResult objects if no reranking
                from ..retrieval.reranking import RerankingResult
                final_reranked_results = [
                    RerankingResult(
                        search_result=result,
                        original_rank=i,
                        new_rank=i,
                        reranking_score=result.score,
                        score_breakdown={'original': result.score}
                    )
                    for i, result in enumerate(search_results[:reranking_top_k])
                ]
                final_results = search_results[:reranking_top_k]
            
            # Step 3: Generate response
            if chat_engine and final_reranked_results:
                response = chat_engine.complete_chat(query, final_reranked_results)
                
                processing_time = time.time() - start_time
                query_count += 1
                
                print("\n" + "=" * 80)
                print("RESPONSE:")
                print("=" * 80)
                print(response.message)
                print("\n" + "-" * 80)
                
                # Build status line with enhancement info
                status_parts = [
                    f"Response time: {processing_time:.2f}s",
                    f"Sources: {len(final_results)}",
                    f"Query #{query_count}"
                ]
                
                if crag_applied:
                    status_parts.append("CRAG: ✓")
                else:
                    status_parts.append("CRAG: ✗")
                
                # Add enhancement information
                if hasattr(response, 'query_enhancement') and response.query_enhancement:
                    enh = response.query_enhancement
                    if enh.original_query != enh.enhanced_query:
                        status_parts.append(f"Query Enhanced: ✓ ({enh.enhancement_type})")
                    else:
                        status_parts.append("Query Enhanced: ✗")
                
                if hasattr(response, 'compression_result') and response.compression_result:
                    comp = response.compression_result
                    if comp.original_message_count > 0:
                        status_parts.append(f"Context Compressed: ✓ (-{comp.tokens_saved} tokens)")
                    else:
                        status_parts.append("Context Compressed: ✗")
                
                print(" | ".join(status_parts))
                
                # Show enhancement details if available
                if hasattr(response, 'query_enhancement') and response.query_enhancement:
                    enh = response.query_enhancement
                    if enh.original_query != enh.enhanced_query and enh.confidence_score > 0.5:
                        print(f"\n🔍 Enhanced query ({enh.confidence_score:.2f} confidence): {enh.enhanced_query}")
                
                if hasattr(response, 'query_validation') and response.query_validation:
                    val = response.query_validation
                    if not val.is_valid and val.issues:
                        print(f"\n⚠️  Query issues: {', '.join(val.issues[:2])}")
                
                if hasattr(response, 'sources_used') and response.sources_used:
                    print("\nSources:")
                    for i, source in enumerate(response.sources_used[:3], 1):
                        print(f"  {i}. {source}")
                        
                if hasattr(response, 'context_snippets') and response.context_snippets:
                    print(f"\nContext snippets used: {len(response.context_snippets)}")
                        
            else:
                print_warning("No relevant results found or chat engine not available.")
                
        except KeyboardInterrupt:
            print_info("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print_error(f"Query processing failed: {str(e)}")


def run_selfrag_pipeline(args: argparse.Namespace) -> bool:
    print_error("Self-RAG is no longer supported in the library version.")
    return False


def run_end_to_end_pipeline(args: argparse.Namespace) -> bool:
    """
    Run the complete end-to-end RAG pipeline.
    """
    print_banner()
    print_info("Starting End-to-End RAG Pipeline")
    print_info(f"Documents: {args.documents_dir}")
    print_info(f"Working Directory: {args.working_dir}")
    print("-" * 80)
    
    overall_start_time = time.time()
    
    # Step 1: Document Processing
    # Build Gemini config if requested
    gemini_cfg = None
    if args.ocr_method == "gemini":
        gemini_cfg = GeminiConfig(
            model=args.gemini_model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=args.gemini_retries,
            daily_limit=0 if args.disable_gemini_limit else args.gemini_daily_limit,
            usage_dir=args.gemini_usage_dir,
            estimated_cost_per_page_usd=args.gemini_estimated_cost_per_page,
        )

    # Build Docling config
    docling_cfg = DoclingConfig(
        extract_tables=args.docling_tables,
        extract_images=args.docling_images,
        extract_figures=args.docling_figures,
        table_extraction_mode=args.docling_table_mode,
        ocr_enabled=False,  # We handle OCR separately in hybrid mode
        offline_mode=args.docling_offline_mode,
        download_timeout=args.docling_download_timeout,
        max_download_retries=args.docling_max_retries,
        cache_dir=args.docling_cache_dir,
        suppress_warnings=not args.no_suppress_pytorch_warnings,
        force_cpu=not args.docling_use_gpu,
    )

    documents = step1_document_processing(
        documents_dir=args.documents_dir,
        working_dir=args.working_dir,
        fallback_to_ocr=args.ocr_fallback,
        enable_metadata=args.enable_metadata,
        ocr_method=args.ocr_method,
        gemini_config=gemini_cfg,
        parser_method=args.parser_method,
        docling_config=docling_cfg,
    )
    
    if not documents:
        print_error("Pipeline failed at document processing step")
        return False
    
    # Step 2: Chunking
    chunks = step2_chunking(
        documents=documents,
        working_dir=args.working_dir,
        strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chonkie_chunker_type=args.chonkie_chunker_type,
        chonkie_tokenizer=args.chonkie_tokenizer,
        chonkie_embedding_model=args.chonkie_embedding_model,
        chonkie_similarity_threshold=args.chonkie_similarity_threshold,
        chonkie_initial_sentences=args.chonkie_initial_sentences
    )
    
    if not chunks:
        print_error("Pipeline failed at chunking step")
        return False
    
    # Step 3: Embeddings and Vector Store
    vector_store = step3_embeddings_and_vector_store(
        chunks=chunks,
        working_dir=args.working_dir,
        embedding_model=args.embedding_model,
        vector_store_type=args.vector_store,
        collection_name=args.collection_name
    )
    
    if not vector_store:
        print_error("Pipeline failed at embeddings/vector storage step")
        return False
    
    # Step 4: Reranking Setup
    reranking_pipeline = step4_setup_reranking(
        working_dir=args.working_dir,
        strategy=args.reranking_strategy,
        top_k=args.reranking_top_k,
        cohere_model=args.cohere_model,
        cohere_api_key=args.cohere_api_key,
        precision_mode=getattr(args, 'precision_reranking', False),
        cross_encoder_weight=getattr(args, 'cross_encoder_weight', 0.7),
        bm25_weight=getattr(args, 'bm25_weight', 0.2),
        semantic_weight=getattr(args, 'semantic_weight', 0.1)
    )
    
    # Step 5: Chat Completion Setup
    chat_engine = step5_setup_chat_completion(
        working_dir=args.working_dir,
        model=args.llm_model,
        max_tokens=args.max_response_tokens,
        temperature=args.temperature,
        enable_query_enhancement=args.enable_query_enhancement,
        enable_context_compression=args.enable_context_compression
    )
    
    total_time = time.time() - overall_start_time
    print_success(f"Pipeline setup completed in {total_time:.2f} seconds")
    
    # Step 6: Interactive Query Session (if requested)
    if args.interactive:
        # Prepare CRAG config if enabled
        crag_config = None
        if getattr(args, 'enable_crag', False):
            crag_config = {
                'trigger_mode': getattr(args, 'crag_trigger_mode', 'hybrid'),
                'min_similarity': getattr(args, 'crag_min_similarity', 0.22),
                'min_token_coverage': getattr(args, 'crag_min_token_coverage', 0.35),
                'max_reformulations': getattr(args, 'crag_max_reformulations', 2),
                'reformulation_model': getattr(args, 'crag_reformulation_model', 'gpt-4o-mini'),
            }
        
        interactive_query_session(
            vector_store, 
            reranking_pipeline, 
            chat_engine,
            retrieval_top_k=args.retrieval_top_k,
            reranking_top_k=args.reranking_top_k,
            enable_crag=getattr(args, 'enable_crag', False),
            crag_config=crag_config,
            similarity_threshold=getattr(args, 'similarity_threshold', 0.25),
            domain_filter=getattr(args, 'domain_filter', None),
            auto_domain_detection=getattr(args, 'auto_domain_detection', True),
            domain_confidence_threshold=getattr(args, 'domain_confidence_threshold', 0.3)
        )

    # Optional: print Gemini usage summary
    try:
        from ..api.api_usage import get_gemini_usage_tracker
        if args.print_gemini_usage and args.ocr_method == "gemini" and get_gemini_usage_tracker:
            get_gemini_usage_tracker().print_summary()
    except Exception:
        pass
    
    return True


def run_chat_only(args: argparse.Namespace) -> bool:
    """
    Load existing pipeline data and jump directly to chat.
    """
    print_banner()
    print_info("Loading existing pipeline data for chat session")
    print_info(f"Working Directory: {args.working_dir}")
    print("-" * 80)
    
    working_dir = Path(args.working_dir)
    
    # Check if working directory exists
    if not working_dir.exists():
        print_error(f"Working directory does not exist: {args.working_dir}")
        print_info("Please run the pipeline first with --e2e or --step-by-step")
        return False
    
    # Check for required files
    required_files = [
        "processed_documents.json",
        "chunks.json",
    ]
    
    for file_name in required_files:
        file_path = working_dir / file_name
        if not file_path.exists():
            print_error(f"Required file not found: {file_path}")
            print_info("Please run the pipeline first to generate the required data")
            return False
    
    # Load configuration from saved data
    try:
        # Load chunks to get configuration info
        chunks_file = working_dir / "chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Infer chunking info from the chunks themselves
        if isinstance(chunks_data, list) and chunks_data:
            # Check first chunk for metadata clues
            first_chunk = chunks_data[0]
            if 'metadata' in first_chunk and 'recursive_level' in first_chunk['metadata']:
                chunking_strategy = 'recursive'
            else:
                chunking_strategy = 'recursive'  # default
            
            # Estimate chunk size from existing chunks
            if 'char_count' in first_chunk:
                chunk_size = first_chunk['char_count']
            else:
                chunk_size = len(first_chunk.get('content', ''))
        else:
            chunking_strategy = 'recursive'
            chunk_size = 1000
        
        print_info(f"Detected chunking strategy: {chunking_strategy}")
        print_info(f"Estimated chunk size: {chunk_size}")
        
    except Exception as e:
        print_warning(f"Could not load chunking config: {e}")
        chunking_strategy = 'recursive'
        chunk_size = 1000
    
    start_time = time.time()
    
    # Step 3: Load Vector Store
    print_step_header(3, "Vector Store Loading", f"Loading existing vector database")
    
    try:
        # Try to detect the existing collection name
        collection_name = "rag_collection"  # Use the detected collection name
        print_info(f"Using collection: {collection_name}")
        
        # Use the same defaults as the main pipeline for consistency
        vector_store_config = VectorStoreConfig(
            store_type="chroma",
            persist_directory=str(working_dir / "vector_db"),
            collection_name=collection_name
        )
        
        # Create embedding function (needed for retrieval)
        embedding_function = create_embedding_function(
            model_name=getattr(args, 'embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        )
        
        vector_store = create_vector_store(vector_store_config, embedding_function)
        
        # Initialize the vector store
        if not vector_store.initialize():
            print_error("Failed to initialize vector store")
            return False
        
        # Check if vector store has data
        try:
            if hasattr(vector_store, 'collection') and vector_store.collection:
                doc_count = vector_store.collection.count()
                print_info(f"Loaded vector store with {doc_count} documents")
            else:
                print_warning("Could not access vector store collection")
                doc_count = -1
            
            if doc_count == 0:
                print_error("Vector store is empty. Please run the pipeline first.")
                return False
                
        except Exception as e:
            print_warning(f"Could not verify vector store contents: {e}")
        
        print_success("Vector store loaded successfully")
        
    except Exception as e:
        print_error(f"Failed to load vector store: {e}")
        print_info("Please ensure the pipeline has been run successfully")
        return False
    
    # Step 4: Setup Reranking
    print_step_header(4, "Reranking Setup", "Initialize reranking pipeline")
    
    try:
        reranking_strategy = getattr(args, 'reranking_strategy', 'hybrid')
        top_k = getattr(args, 'reranking_top_k', 20)
        
        print_info(f"Reranking strategy: {reranking_strategy}")
        print_info(f"Top-K results: {top_k}")
        
        reranking_pipeline = create_reranking_pipeline(
            strategy=reranking_strategy,
            top_k_final=top_k
        )
        
        print_success("Reranking pipeline initialized successfully")
        
    except Exception as e:
        print_error(f"Failed to setup reranking: {e}")
        return False
    
    # Step 5: Setup Chat Completion
    print_step_header(5, "Chat Completion Setup", "Initialize chat engine")
    
    try:
        llm_model = getattr(args, 'llm_model', 'gpt-4o')
        max_tokens = getattr(args, 'llm_max_tokens', 2000)
        temperature = getattr(args, 'llm_temperature', 0.6)
        
        print_info(f"Model: {llm_model}")
        print_info(f"Max tokens: {max_tokens}")
        print_info(f"Temperature: {temperature}")
        
        chat_engine = create_chat_completion_engine(
            model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        print_success("Chat completion engine initialized successfully")
        
    except Exception as e:
        print_error(f"Failed to setup chat completion: {e}")
        return False
    
    total_time = time.time() - start_time
    print_success(f"Chat setup completed in {total_time:.2f} seconds")
    
    # Start Interactive Chat Session
    interactive_query_session(
        vector_store, 
        reranking_pipeline, 
        chat_engine,
        retrieval_top_k=getattr(args, 'retrieval_top_k', 75),
        reranking_top_k=getattr(args, 'reranking_top_k', 20),
        enable_crag=False,  # CRAG not supported in chat-only mode
        crag_config=None,
        similarity_threshold=getattr(args, 'similarity_threshold', 0.0),  # Conservative default for chat-only
        domain_filter=getattr(args, 'domain_filter', None),
        auto_domain_detection=getattr(args, 'auto_domain_detection', True),
        domain_confidence_threshold=getattr(args, 'domain_confidence_threshold', 0.3)
    )
    
    return True


def run_step_by_step_pipeline(args: argparse.Namespace) -> bool:
    """
    Run the pipeline step by step with user confirmation.
    """
    print_banner()
    print_info("Starting Step-by-Step RAG Pipeline")
    print_info("You'll be prompted before each step.")
    print("-" * 80)
    
    # Step 1: Document Processing
    # Prepare optional Gemini config if selected
    gemini_cfg = None
    if args.ocr_method == "gemini":
        try:
            # Propagate usage dir so worker processes/readers share the same location
            os.environ["GEMINI_USAGE_DIR"] = args.gemini_usage_dir
        except Exception:
            pass
        gemini_cfg = GeminiConfig(
            model=args.gemini_model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=args.gemini_retries,
            daily_limit=0 if args.disable_gemini_limit else args.gemini_daily_limit,
            usage_dir=args.gemini_usage_dir,
            estimated_cost_per_page_usd=args.gemini_estimated_cost_per_page,
        )

    # Prepare Docling config
    docling_cfg = DoclingConfig(
        extract_tables=args.docling_tables,
        extract_images=args.docling_images,
        extract_figures=args.docling_figures,
        table_extraction_mode=args.docling_table_mode,
        ocr_enabled=False,  # We handle OCR separately in hybrid mode
        offline_mode=args.docling_offline_mode,
        download_timeout=args.docling_download_timeout,
        max_download_retries=args.docling_max_retries,
        cache_dir=args.docling_cache_dir,
        suppress_warnings=not args.no_suppress_pytorch_warnings,
        force_cpu=not args.docling_use_gpu,
    )

    if input("\nRun Step 1: Document Processing? (y/n): ").lower().startswith('y'):
        documents = step1_document_processing(
            documents_dir=args.documents_dir,
            working_dir=args.working_dir,
            fallback_to_ocr=args.ocr_fallback,
            enable_metadata=args.enable_metadata,
            ocr_method=args.ocr_method,
            gemini_config=gemini_cfg,
            parser_method=args.parser_method,
            docling_config=docling_cfg,
        )
        if not documents:
            return False
    else:
        print_info("Skipping Step 1. Loading existing results...")
        try:
            with open(Path(args.working_dir) / "processed_documents.json", 'r', encoding='utf-8') as f:
                documents = json.load(f)
            print_success(f"Loaded {len(documents)} processed documents")
        except FileNotFoundError:
            print_error("No existing processed documents found. Please run Step 1.")
            return False
    
    # Step 2: Chunking
    if input("\nRun Step 2: Document Chunking? (y/n): ").lower().startswith('y'):
        chunks = step2_chunking(
            documents=documents,
            working_dir=args.working_dir,
            strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chonkie_chunker_type=args.chonkie_chunker_type,
            chonkie_tokenizer=args.chonkie_tokenizer,
            chonkie_embedding_model=args.chonkie_embedding_model,
            chonkie_similarity_threshold=args.chonkie_similarity_threshold,
            chonkie_initial_sentences=args.chonkie_initial_sentences
        )
        if not chunks:
            return False
    else:
        print_info("Skipping Step 2. You'll need to provide chunks for the next steps.")
        chunks = []  # Would need to implement chunk loading here
    
    # Continue with remaining steps...
    if chunks:
        if input("\nRun Step 3: Embeddings & Vector Storage? (y/n): ").lower().startswith('y'):
            vector_store = step3_embeddings_and_vector_store(
                chunks=chunks,
                working_dir=args.working_dir,
                embedding_model=args.embedding_model,
                vector_store_type=args.vector_store,
                collection_name=args.collection_name
            )
        
        if input("\nRun Step 4: Setup Reranking? (y/n): ").lower().startswith('y'):
            reranking_pipeline = step4_setup_reranking(
                working_dir=args.working_dir,
                strategy=args.reranking_strategy,
                top_k=args.reranking_top_k,
                cohere_model=args.cohere_model,
                cohere_api_key=args.cohere_api_key,
                precision_mode=getattr(args, 'precision_reranking', False),
                cross_encoder_weight=getattr(args, 'cross_encoder_weight', 0.7),
                bm25_weight=getattr(args, 'bm25_weight', 0.2),
                semantic_weight=getattr(args, 'semantic_weight', 0.1)
            )
        
        if input("\nRun Step 5: Setup Chat Completion? (y/n): ").lower().startswith('y'):
            chat_engine = step5_setup_chat_completion(
                working_dir=args.working_dir,
                model=args.llm_model,
                max_tokens=args.max_response_tokens,
                temperature=args.temperature,
                enable_query_enhancement=args.enable_query_enhancement,
                enable_context_compression=args.enable_context_compression
            )
        
        
        if input("\nRun Step 6: Interactive Query Session? (y/n): ").lower().startswith('y'):
            # Prepare CRAG config if enabled
            crag_config = None
            if getattr(args, 'enable_crag', False):
                crag_config = {
                    'trigger_mode': getattr(args, 'crag_trigger_mode', 'hybrid'),
                    'min_similarity': getattr(args, 'crag_min_similarity', 0.22),
                    'min_token_coverage': getattr(args, 'crag_min_token_coverage', 0.35),
                    'max_reformulations': getattr(args, 'crag_max_reformulations', 2),
                    'reformulation_model': getattr(args, 'crag_reformulation_model', 'gpt-4o-mini'),
                }
            
            interactive_query_session(
                vector_store if 'vector_store' in locals() else None,
                reranking_pipeline if 'reranking_pipeline' in locals() else None,
                chat_engine if 'chat_engine' in locals() else None,
                retrieval_top_k=args.retrieval_top_k,
                reranking_top_k=args.reranking_top_k,
                enable_crag=getattr(args, 'enable_crag', False),
                crag_config=crag_config,
                similarity_threshold=getattr(args, 'similarity_threshold', 0.25),
                domain_filter=getattr(args, 'domain_filter', None),
                auto_domain_detection=getattr(args, 'auto_domain_detection', True),
                domain_confidence_threshold=getattr(args, 'domain_confidence_threshold', 0.3)
            )

    # Optional: print Gemini usage summary
    try:
        from ..api.api_usage import get_gemini_usage_tracker
        if args.print_gemini_usage and args.ocr_method == "gemini" and get_gemini_usage_tracker:
            get_gemini_usage_tracker().print_summary()
    except Exception:
        pass
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DingDong RAG: Complete Document Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # End-to-end pipeline with defaults
  python main.py --documents-dir ./documents --e2e
  
  # Step-by-step pipeline with custom settings
  python main.py --documents-dir ./documents --step-by-step --chunk-size 800
  
  # Jump directly to chat using existing pipeline data
  python main.py --documents-dir ./documents --chat-only
  
  # Use Docling parser instead of PyMuPDF
  python main.py --documents-dir ./documents --e2e --parser-method docling
  
  # Pre-download Docling models (for network issues)
  python main.py --pre-download-docling-models --docling-download-timeout 600
  
  # Use Docling in offline mode (after pre-downloading) - quiet operation
  python main.py --documents-dir ./documents --e2e --parser-method docling --docling-offline-mode
  
  # RECOMMENDED: Optimized setup with best practices (requires OpenAI key)
  python main.py --documents-dir ./documents --e2e --embedding-model text-embedding-3-large --reranking-strategy hybrid --reranking-top-k 20 --retrieval-top-k 75
  
  # Production setup with OpenAI models and Docling
  python main.py --documents-dir ./documents --e2e --parser-method docling --embedding-model text-embedding-3-large --llm-model gpt-4o
  
  # Multilingual setup for Indonesian+English documents (recommended)
  python main.py --documents-dir ./documents --e2e --embedding-model paraphrase-multilingual-MiniLM-L12-v2
  
  # High-quality multilingual setup with LaBSE
  python main.py --documents-dir ./documents --e2e --embedding-model sentence-transformers/LaBSE
  
  # Enable CRAG refinement for improved retrieval quality
  python main.py --documents-dir ./documents --e2e --enable-crag --crag-trigger-mode hybrid
  
   # Self-RAG options have been removed in the library version
  
  # Advanced CRAG setup with custom thresholds
  python main.py --documents-dir ./documents --e2e --enable-crag --crag-min-similarity 0.25 --crag-max-reformulations 3
  
  # HIGH PRECISION: Reduce topic bleed with similarity threshold and precision reranking
  python main.py --documents-dir ./documents --e2e --similarity-threshold 0.3 --precision-reranking
  
  # Domain-specific search (OS documents only - manual override)
  python main.py --documents-dir ./documents --e2e --domain-filter "IF2130 - Sistem Operasi" --similarity-threshold 0.25
  
  # AUTOMATIC domain detection (default) - queries automatically filtered by detected subject
  python main.py --documents-dir ./documents --e2e --similarity-threshold 0.3 --precision-reranking
  
  # Disable automatic domain detection for cross-domain queries
  python main.py --documents-dir ./documents --e2e --disable-auto-domain-detection --similarity-threshold 0.25
  
  # Custom hybrid reranking weights for fine-tuning
  python main.py --documents-dir ./documents --e2e --cross-encoder-weight 0.8 --bm25-weight 0.2 --semantic-weight 0.0
  
  # Disable query enhancement features for faster processing
  python main.py --documents-dir ./documents --e2e --disable-query-enhancement --disable-context-compression
  
  # Custom compression settings for long conversations
  python main.py --documents-dir ./documents --e2e --max-uncompressed-messages 20 --compression-ratio 0.3
  
  # Quick demo run (non-interactive)
  python main.py --documents-dir ./documents --e2e --no-interactive
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--documents-dir", "-d",
        type=str, 
        required=True,
        help="Directory containing PDF documents to process"
    )
    
    # Execution mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--e2e", "--end-to-end",
        action="store_true",
        help="Run complete end-to-end pipeline"
    )
    mode_group.add_argument(
        "--step-by-step", "-s",
        action="store_true",
        help="Run pipeline step by step with confirmations"
    )
    mode_group.add_argument(
        "--chat-only", "-c",
        action="store_true",
        help="Jump directly to chat using existing pipeline data"
    )
    
    # Directory and file options
    parser.add_argument(
        "--working-dir", "-w",
        type=str,
        default="./rag_working_dir",
        help="Working directory for intermediate files (default: ./rag_working_dir)"
    )
    
    # Document processing options
    parser.add_argument(
        "--no-ocr-fallback",
        dest="ocr_fallback",
        action="store_false",
        help="Disable OCR fallback for PDF processing"
    )
    parser.add_argument(
        "--ocr-method",
        type=str,
        default="tesseract",
        choices=["tesseract", "gemini"],
        help="OCR method used by hybrid parser (default: tesseract)"
    )
    parser.add_argument(
        "--parser-method",
        type=str,
        default="pymupdf",
        choices=["pymupdf", "docling"],
        help="Primary PDF parser method: pymupdf (PyMuPDF4LLM) or docling (default: pymupdf)"
    )
    parser.add_argument(
        "--docling-tables",
        action="store_true",
        default=True,
        help="Enable table extraction in Docling (default: True)"
    )
    parser.add_argument(
        "--no-docling-tables",
        action="store_false",
        dest="docling_tables",
        help="Disable table extraction in Docling"
    )
    parser.add_argument(
        "--docling-figures",
        action="store_true",
        default=True,
        help="Enable figure extraction in Docling (default: True)"
    )
    parser.add_argument(
        "--no-docling-figures",
        action="store_false",
        dest="docling_figures",
        help="Disable figure extraction in Docling"
    )
    parser.add_argument(
        "--docling-images",
        action="store_true",
        default=False,
        help="Enable image extraction in Docling (default: False)"
    )
    parser.add_argument(
        "--docling-table-mode",
        type=str,
        default="fast",
        choices=["fast", "accurate", "hybrid"],
        help="Table extraction mode for Docling (default: fast)"
    )
    parser.add_argument(
        "--docling-offline-mode",
        action="store_true",
        default=False,
        help="Use Docling in offline mode (skip model downloads, use only local models)"
    )
    parser.add_argument(
        "--docling-download-timeout",
        type=int,
        default=300,
        help="Timeout for Docling model downloads in seconds (default: 300)"
    )
    parser.add_argument(
        "--docling-max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed Docling model downloads (default: 3)"
    )
    parser.add_argument(
        "--docling-cache-dir",
        type=str,
        help="Custom cache directory for Docling models (default: system default)"
    )
    parser.add_argument(
        "--pre-download-docling-models",
        action="store_true",
        help="Pre-download Docling models and exit (useful for network issues)"
    )
    parser.add_argument(
        "--no-suppress-pytorch-warnings",
        action="store_true",
        default=False,
        help="Show PyTorch warnings (default: warnings are suppressed for cleaner output)"
    )
    parser.add_argument(
        "--docling-use-gpu",
        action="store_true",
        default=False,
        help="Allow Docling to use GPU if available (default: force CPU mode)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash-lite",
        help="Gemini model to use when --ocr-method gemini (default: gemini-2.0-flash-lite)"
    )
    parser.add_argument(
        "--gemini-retries",
        type=int,
        default=2,
        help="Max retries for Gemini OCR calls (default: 2)"
    )
    parser.add_argument(
        "--gemini-daily-limit",
        type=int,
        default=1500,
        help="Daily Gemini request cap; set 0 to disable (default: 1500)"
    )
    parser.add_argument(
        "--disable-gemini-limit",
        action="store_true",
        help="Disable Gemini daily limit enforcement"
    )
    parser.add_argument(
        "--gemini-usage-dir",
        type=str,
        default=".cache/api_usage",
        help="Directory for Gemini API usage logs (default: .cache/api_usage)"
    )
    parser.add_argument(
        "--gemini-estimated-cost-per-page",
        type=float,
        default=0.0,
        help="Optional cost estimate added per Gemini page request in USD (default: 0.0)"
    )
    parser.add_argument(
        "--print-gemini-usage",
        action="store_true",
        help="Print Gemini API usage summary at the end of the run"
    )
    
    # Chunking options
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="recursive",
        choices=["fixed", "sentence", "recursive", "semantic", "chonkie"],
        help="Text chunking strategy (default: recursive)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters (default: 1200)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="Chunk overlap in characters (default: 300)"
    )
    
    # Chonkie-specific options
    parser.add_argument(
        "--chonkie-chunker-type",
        type=str,
        default="word",
        choices=["word", "sentence", "token", "semantic", "sdpm"],
        help="Chonkie chunker type (default: word)"
    )
    parser.add_argument(
        "--chonkie-tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer for Chonkie token-based chunkers (default: gpt2)"
    )
    parser.add_argument(
        "--chonkie-embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model for Chonkie semantic chunkers (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--chonkie-similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for Chonkie semantic chunkers (default: 0.5)"
    )
    parser.add_argument(
        "--chonkie-initial-sentences",
        type=int,
        default=1,
        help="Initial sentences for Chonkie SDPM chunker (default: 1)"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        choices=[
            # English-focused models
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            # Multilingual models (recommended for Indonesian+English)
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased", 
            "sentence-transformers/LaBSE",
            # OpenAI models (require API key)
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ],
        help="Embedding model name. RECOMMENDED: text-embedding-3-large (best quality, requires OpenAI key). For free/offline: paraphrase-multilingual-MiniLM-L12-v2 or LaBSE (default: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    
    # Vector store options
    parser.add_argument(
        "--vector-store",
        type=str,
        default="chroma",
        choices=["chroma", "pinecone"],
        help="Vector store type (default: chroma)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rag_collection",
        help="Vector store collection name (default: rag_collection)"
    )
    
    # Reranking options
    parser.add_argument(
        "--reranking-strategy",
        type=str,
        default="hybrid",
        choices=["cross_encoder", "bm25", "hybrid", "cohere"],
        help="Reranking strategy (default: hybrid)"
    )
    parser.add_argument(
        "--reranking-top-k",
        type=int,
        default=20,
        help="Number of top results after reranking (default: 20, optimal for most use cases)"
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=75,
        help="Number of initial results to retrieve before reranking (default: 75, higher values improve hybrid reranking)"
    )
    parser.add_argument(
        "--cohere-model",
        type=str,
        default="rerank-english-v3.0",
        help="Cohere reranking model (default: rerank-english-v3.0)"
    )
    parser.add_argument(
        "--cohere-api-key",
        type=str,
        help="Cohere API key (alternatively use COHERE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score to keep results before reranking (0.0-1.0, default: 0.0 = disabled, recommended: 0.25-0.35 for precision)"
    )
    parser.add_argument(
        "--domain-filter", 
        type=str,
        help="Filter results to specific domain/course (e.g., 'IF2130 - Sistem Operasi' for OS-only results)"
    )
    parser.add_argument(
        "--precision-reranking",
        action="store_true",
        help="Use precision-tuned hybrid reranking (cross_encoder=0.8, bm25=0.2, semantic=0.0) to reduce topic bleed"
    )
    parser.add_argument(
        "--cross-encoder-weight",
        type=float,
        default=0.7,
        help="Weight for cross-encoder scores in hybrid reranking (default: 0.7, precision mode: 0.8)"
    )
    parser.add_argument(
        "--bm25-weight", 
        type=float,
        default=0.2,
        help="Weight for BM25 scores in hybrid reranking (default: 0.2)"
    )
    parser.add_argument(
        "--semantic-weight",
        type=float, 
        default=0.1,
        help="Weight for semantic embedding scores in hybrid reranking (default: 0.1, precision mode: 0.0)"
    )
    parser.add_argument(
        "--disable-auto-domain-detection",
        dest="auto_domain_detection",
        action="store_false",
        help="Disable automatic domain detection from query keywords"
    )
    parser.add_argument(
        "--domain-confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for automatic domain detection (0.0-1.0, default: 0.3)"
    )
    
    # Chat completion options
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="Language model for chat completion (default: gpt-4o)"
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=2000,
        help="Maximum response tokens (default: 2000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="LLM temperature (default: 0.6)"
    )
    
    # Chat enhancement options
    parser.add_argument(
        "--enable-query-enhancement",
        action="store_true",
        default=True,
        help="Enable LLM-based query enhancement and rephrasing for better retrieval (default: enabled)"
    )
    parser.add_argument(
        "--disable-query-enhancement",
        action="store_false",
        dest="enable_query_enhancement",
        help="Disable query enhancement features"
    )
    parser.add_argument(
        "--enable-context-compression",
        action="store_true", 
        default=True,
        help="Enable automatic context compression for long conversations (default: enabled)"
    )
    parser.add_argument(
        "--disable-context-compression",
        action="store_false",
        dest="enable_context_compression",
        help="Disable automatic context compression"
    )
    parser.add_argument(
        "--max-uncompressed-messages",
        type=int,
        default=15,
        help="Maximum number of messages before triggering compression (default: 15)"
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.4,
        help="Target compression ratio (0.0-1.0, default: 0.4 means compress to 40%% of original)"
    )
    
    # CRAG refinement options
    parser.add_argument(
        "--enable-crag",
        action="store_true",
        help="Enable CRAG (Corrective RAG) refinement for improved retrieval quality"
    )
    parser.add_argument(
        "--crag-trigger-mode",
        type=str,
        default="hybrid",
        choices=["always", "score", "token_coverage", "hybrid"],
        help="CRAG trigger mode: when to apply refinement (default: hybrid)"
    )
    parser.add_argument(
        "--crag-min-similarity",
        type=float,
        default=0.22,
        help="Minimum mean similarity threshold for CRAG trigger (default: 0.22)"
    )
    parser.add_argument(
        "--crag-min-token-coverage",
        type=float,
        default=0.35,
        help="Minimum token coverage threshold for CRAG trigger (default: 0.35)"
    )
    parser.add_argument(
        "--crag-max-reformulations",
        type=int,
        default=2,
        help="Maximum number of query reformulations in CRAG (default: 2)"
    )
    parser.add_argument(
        "--crag-reformulation-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for CRAG query reformulation (default: gpt-4o-mini)"
    )
    
    # Self-RAG options removed in library version
    
    # Interactive options
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Skip interactive query session"
    )
    
    # Metadata options
    parser.add_argument(
        "--metadata",
        dest="enable_metadata",
        action="store_true",
        default=True,
        help="Enable metadata extraction (default: True)"
    )
    parser.add_argument(
        "--no-metadata",
        dest="enable_metadata",
        action="store_false",
        help="Disable metadata extraction"
    )
    
    # Utility options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle pre-download models option
    if args.pre_download_docling_models:
        from ..parsing.docling_adapter import pre_download_models, DoclingConfig
        
        print_banner()
        print("Pre-downloading Docling models...")
        
        config = DoclingConfig(
            offline_mode=False,
            download_timeout=args.docling_download_timeout,
            max_download_retries=args.docling_max_retries,
            cache_dir=args.docling_cache_dir,
            suppress_warnings=not args.no_suppress_pytorch_warnings,
            force_cpu=not args.docling_use_gpu,
        )
        
        success = pre_download_models(config)
        if success:
            print_success("Docling models downloaded successfully!")
            print_info("You can now run with --docling-offline-mode to use cached models")
        else:
            print_error("Failed to download Docling models")
            sys.exit(1)
        
        return 0
    
    # Validate arguments
    if not Path(args.documents_dir).exists():
        print_error(f"Documents directory does not exist: {args.documents_dir}")
        sys.exit(1)
    
    # Check for required API keys
    if 'openai' in args.embedding_model.lower() or 'gpt' in args.llm_model.lower():
        if not os.getenv('OPENAI_API_KEY'):
            print_warning("OpenAI API key not found in environment variables.")
            print_info("Set OPENAI_API_KEY environment variable for OpenAI models.")
    
    if args.vector_store == 'pinecone' and not os.getenv('PINECONE_API_KEY'):
        print_warning("Pinecone API key not found in environment variables.")
        print_info("Set PINECONE_API_KEY environment variable for Pinecone vector store.")
    
    # Run the selected pipeline mode
    try:
        if args.e2e:
            success = run_end_to_end_pipeline(args)
        elif args.step_by_step:
            success = run_step_by_step_pipeline(args)
        else:  # chat-only
            success = run_chat_only(args)
        
        if success:
            print_success("Pipeline completed successfully!")
            sys.exit(0)
        else:
            print_error("Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_info("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Pipeline failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()