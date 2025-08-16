import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from ..core.complete_rag_pipeline import create_complete_rag_pipeline, create_production_rag_pipeline
from ..evaluation.rag_evaluation import run_comprehensive_rag_evaluation, compare_rag_configurations
from ..embeddings.embedding_config import compare_embedding_models


def cmd_initialize(args):
    """Initialize RAG system."""
    
    print("Initializing RAG System")
    print("=" * 40)
    
    # Check if documents directory exists
    if not Path(args.documents_dir).exists():
        print(f"ERROR: Documents directory not found: {args.documents_dir}")
        return 1
    
    # Create RAG system
    if args.production:
        if not args.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OpenAI API key required for production mode")
            print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
            return 1
        
        rag_system = create_production_rag_pipeline(
            documents_dir=args.documents_dir,
            working_dir=args.working_dir,
            openai_api_key=args.openai_api_key
        )
    else:
        rag_system = create_complete_rag_pipeline(
            documents_dir=args.documents_dir,
            working_dir=args.working_dir,
            embedding_model=args.embedding_model,
            auto_optimize=not args.no_optimize,
            openai_api_key=args.openai_api_key
        )
    
    # Initialize system
    try:
        results = rag_system.initialize_pipeline()
        
        if 'error' in results:
            print(f"ERROR: Initialization failed: {results['error']}")
            return 1
        
        # Save system state
        rag_system.save_pipeline_state()
        
        print("\nRAG system initialized successfully.")
        print(f"Working directory: {args.working_dir}")
        print(f"Embedding model: {rag_system.config.embedding_model}")
        print(f"Documents processed: {results.get('ingestion_stats', {}).get('metrics', {}).get('documents_ingested', 0)}")
        
        # Show next steps
        print("\nNext steps:")
        print(f"  Query: python rag_cli.py query --working-dir {args.working_dir} \"Your question here\"")
        print(f"  Evaluate: python rag_cli.py evaluate --working-dir {args.working_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nERROR: Initialization interrupted by user")
        return 1
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        return 1


def cmd_query(args):
    """Query the RAG system."""
    
    # Check if system is initialized
    if not Path(args.working_dir).exists():
        print(f"ERROR: RAG system not found at {args.working_dir}")
        print("Initialize first with: python rag_cli.py init --documents-dir <path>")
        return 1
    
    # Load system configuration
    try:
        config_file = Path(args.working_dir) / "system_state.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                state = json.load(f)
                config = state['config']
        else:
            # Use defaults
            config = {
                'documents_dir': './documents',
                'embedding_model': 'all-MiniLM-L6-v2',
                'openai_api_key': args.openai_api_key
            }
        
        # Recreate RAG system
        rag_system = create_complete_rag_pipeline(
            documents_dir=config.get('documents_dir', './documents'),
            working_dir=args.working_dir,
            embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2'),
            openai_api_key=config.get('openai_api_key') or args.openai_api_key
        )
        
        # Skip initialization since system already exists
        # RAG system ready for use
        
    except Exception as e:
        print(f"ERROR: Failed to load RAG system: {e}")
        return 1
    
    # Process query
    print(f"Querying RAG pipeline: \"{args.question}\"")
    print("=" * 60)
    
    try:
        result = rag_system.query_pipeline(args.question)
        
        if hasattr(result, 'answer'):
            print(f"\nAnswer:\n{result.answer}")
            print(f"\nQuery processing time: {result.total_time:.2f}s")
            print(f"Retrieved chunks: {result.retrieved_chunks}")
            print(f"Reranked chunks: {result.reranked_chunks}")
            if result.sources_used:
                print(f"Sources used: {len(result.sources_used)}")
        else:
            print("ERROR: Invalid response from pipeline")
            return 1
            
    except Exception as e:
        print(f"ERROR: Query failed: {e}")
        return 1
    
    return 0


def cmd_evaluate(args):
    """Evaluate RAG system performance."""
    
    # Check if system exists
    if not Path(args.working_dir).exists():
        print(f"ERROR: RAG system not found at {args.working_dir}")
        return 1
    
    try:
        # Load system
        config_file = Path(args.working_dir) / "system_state.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                state = json.load(f)
                config = state['config']
        else:
            config = {'documents_dir': './documents', 'embedding_model': 'all-MiniLM-L6-v2'}
        
        rag_system = create_complete_rag_pipeline(
            documents_dir=config.get('documents_dir', './documents'),
            working_dir=args.working_dir,
            embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Run evaluation
        print("Running comprehensive RAG evaluation...")
        
        results = run_comprehensive_rag_evaluation(
            rag_system=rag_system,
            output_dir=args.output_dir
        )
        
        print("Evaluation complete.")
        return 0
        
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return 1


def cmd_benchmark(args):
    """Benchmark embedding models."""
    
    print("Benchmarking embedding models...")
    
    sample_texts = [
        "What are the main concepts in machine learning?",
        "How does artificial intelligence work?",
        "Explain the difference between supervised and unsupervised learning.",
        "What are the applications of deep learning in computer vision?",
        "How do neural networks process information?"
    ]
    
    models = ["all-MiniLM-L6-v2"]
    if args.openai_api_key or os.getenv("OPENAI_API_KEY"):
        models.extend(["text-embedding-3-small", "text-embedding-3-large"])
    
    results = compare_embedding_models(
        texts=sample_texts,
        models=models,
        api_key=args.openai_api_key
    )
    
    print("\nEMBEDDING MODEL BENCHMARK RESULTS")
    print("=" * 50)
    
    for model, metrics in results.items():
        print(f"\n{model}:")
        if 'error' in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            print(f"  Dimension: {metrics.get('embedding_dimension', 'Unknown')}")
            print(f"  Time: Test time: {metrics.get('test_time', 0):.2f}s")
            print(f"  Estimated cost: ${metrics.get('metrics', {}).get('total_cost', 0):.4f}")
            print(f"  Texts processed: {metrics.get('texts_processed', 0)}")
    
    return 0


def cmd_upgrade(args):
    """Upgrade to production models."""
    
    # Check if system exists
    if not Path(args.working_dir).exists():
        print(f"ERROR: RAG system not found at {args.working_dir}")
        return 1
    
    if not args.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key required for production upgrade")
        return 1
    
    try:
        # Load and upgrade system
        config_file = Path(args.working_dir) / "system_state.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                state = json.load(f)
                config = state['config']
        else:
            config = {'documents_dir': './documents'}
        
        rag_system = create_complete_rag_pipeline(
            documents_dir=config.get('documents_dir', './documents'),
            working_dir=args.working_dir,
            openai_api_key=args.openai_api_key
        )
        
        print("Creating production pipeline...")
        
        # Create a new production pipeline
        production_rag = create_production_rag_pipeline(
            documents_dir=rag_system.config.documents_dir,
            working_dir=rag_system.config.working_dir,
            openai_api_key=args.openai_api_key
        )
        
        # Initialize the production pipeline
        results = production_rag.initialize_pipeline()
        
        if 'steps_failed' in results and results['steps_failed']:
            print(f"ERROR: Production upgrade failed")
            return 1
        else:
            print("Successfully created production pipeline.")
            print(f"Embedding model: {production_rag.config.embedding_model}")
            production_rag.save_pipeline_state()
            return 0
            
    except Exception as e:
        print(f"ERROR: Upgrade failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Complete RAG CLI - Optimal chunking + SelfRAG + Flexible embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with auto-optimization
  python rag_cli.py init --documents-dir ./documents
  
  # Initialize for production (requires OpenAI API key)
  python rag_cli.py init --documents-dir ./documents --production
  
  # Query the system
  python rag_cli.py query "What are the main concepts in machine learning?"
  
  # Compare all query modes
  python rag_cli.py query "How does AI work?" --compare-modes
  
  # Evaluate system performance
  python rag_cli.py evaluate
  
  # Benchmark embedding models
  python rag_cli.py benchmark --openai-api-key sk-...
  
  # Upgrade to production models
  python rag_cli.py upgrade --openai-api-key sk-...
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--working-dir", "-w",
        type=str,
        default="./rag_system",
        help="Working directory for RAG system (default: ./rag_system)"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize RAG system')
    init_parser.add_argument(
        "--documents-dir", "-d",
        type=str,
        required=True,
        help="Directory containing PDF documents"
    )
    init_parser.add_argument(
        "--embedding-model", "-e",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "text-embedding-3-small", "text-embedding-3-large"],
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )
    init_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip chunking optimization"
    )
    init_parser.add_argument(
        "--production",
        action="store_true",
        help="Initialize for production with high-quality models"
    )
    init_parser.set_defaults(func=cmd_initialize)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument(
        "question",
        type=str,
        help="Question to ask the RAG system"
    )
    query_parser.add_argument(
        "--mode", "-m",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid"],
        help="Query mode (default: hybrid)"
    )
    query_parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare all query modes"
    )
    query_parser.set_defaults(func=cmd_query)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate RAG system performance')
    eval_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./rag_evaluation_results",
        help="Output directory for evaluation results"
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark embedding models')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade to production models')
    upgrade_parser.set_defaults(func=cmd_upgrade)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())