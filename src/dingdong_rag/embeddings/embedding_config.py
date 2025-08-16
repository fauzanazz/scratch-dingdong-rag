try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently fail if dotenv is not available

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import openai


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class EmbeddingModel(Enum):
    """Supported embedding models."""
    # Fast local models for testing (English-focused)
    MINILM_L6_V2 = "all-MiniLM-L6-v2"
    MINILM_L12_V2 = "all-MiniLM-L12-v2"
    
    # Multilingual local models (recommended for Indonesian+English)
    PARAPHRASE_MULTILINGUAL_MINILM_L12_V2 = "paraphrase-multilingual-MiniLM-L12-v2"
    DISTILUSE_BASE_MULTILINGUAL_CASED = "distiluse-base-multilingual-cased"
    LABSE = "sentence-transformers/LaBSE"
    
    # OpenAI models for production (multilingual by default)
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations."""
    total_tokens: int = 0
    total_cost: float = 0.0
    total_requests: int = 0
    total_time: float = 0.0
    avg_latency: float = 0.0
    
    def add_request(self, tokens: int, cost: float, latency: float):
        """Add metrics from a single request."""
        self.total_tokens += tokens
        self.total_cost += cost
        self.total_requests += 1
        self.total_time += latency
        self.avg_latency = self.total_time / self.total_requests


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model: EmbeddingModel
    provider: EmbeddingProvider
    batch_size: int = 32
    max_tokens_per_request: int = 8192
    cache_embeddings: bool = True
    api_key: Optional[str] = None
    
    # Cost per 1M tokens (as of 2024)
    cost_per_million_tokens: Dict[EmbeddingModel, float] = None
    
    def __post_init__(self):
        if self.cost_per_million_tokens is None:
            self.cost_per_million_tokens = {
                # OpenAI pricing (USD per 1M tokens)
                EmbeddingModel.TEXT_EMBEDDING_3_SMALL: 0.02,
                EmbeddingModel.TEXT_EMBEDDING_3_LARGE: 0.13,
                EmbeddingModel.TEXT_EMBEDDING_ADA_002: 0.10,
                # Local models are free
                EmbeddingModel.MINILM_L6_V2: 0.0,
                EmbeddingModel.MINILM_L12_V2: 0.0,
                EmbeddingModel.PARAPHRASE_MULTILINGUAL_MINILM_L12_V2: 0.0,
                EmbeddingModel.DISTILUSE_BASE_MULTILINGUAL_CASED: 0.0,
                EmbeddingModel.LABSE: 0.0,
            }
    
    def get_cost_per_token(self) -> float:
        """Get cost per token for this model."""
        return self.cost_per_million_tokens.get(self.model, 0.0) / 1_000_000
    
    def estimate_cost(self, num_tokens: int) -> float:
        """Estimate cost for given number of tokens."""
        return num_tokens * self.get_cost_per_token()


class EmbeddingFunction:
    """Unified embedding function interface."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.metrics = EmbeddingMetrics()
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        if self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            self._model = SentenceTransformer(self.config.model.value)
        elif self.config.provider == EmbeddingProvider.OPENAI:
            if not self.config.api_key:
                self.config.api_key = os.getenv("OPENAI_API_KEY")
            if not self.config.api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            openai.api_key = self.config.api_key
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def embed_text(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Embed text(s) and return embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        start_time = time.time()
        
        if self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            embeddings = self._embed_sentence_transformers(texts)
        elif self.config.provider == EmbeddingProvider.OPENAI:
            embeddings = self._embed_openai(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # Update metrics
        latency = time.time() - start_time
        total_tokens = sum(len(text.split()) for text in texts)  # Rough estimate
        cost = self.config.estimate_cost(total_tokens)
        self.metrics.add_request(total_tokens, cost, latency)
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    def _embed_sentence_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using SentenceTransformers."""
        embeddings = self._model.encode(
            texts, 
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return [embeddings[i] for i in range(len(embeddings))]
    
    def _embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using OpenAI API."""
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                response = openai.embeddings.create(
                    model=self.config.model.value,
                    input=batch
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"OpenAI embedding error: {e}")
                # Fallback to zero embeddings for failed requests
                embeddings.extend([np.zeros(1536) for _ in batch])  # Assuming 1536 dims
        
        return embeddings
    
    async def embed_text_async(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Async version of embed_text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self._model.get_sentence_embedding_dimension()
        elif self.config.model == EmbeddingModel.TEXT_EMBEDDING_3_LARGE:
            return 3072
        elif self.config.model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL:
            return 1536
        elif self.config.model == EmbeddingModel.TEXT_EMBEDDING_ADA_002:
            return 1536
        else:
            return 1536  # Default
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'model': self.config.model.value,
            'provider': self.config.provider.value,
            'metrics': asdict(self.metrics),
            'estimated_monthly_cost': self.metrics.total_cost * 30 if self.metrics.total_cost > 0 else 0
        }


def create_embedding_function(
    model_name: str, 
    api_key: Optional[str] = None,
    batch_size: int = 32
) -> EmbeddingFunction:
    """Factory function to create embedding function by name."""
    
    # Map string names to enums
    model_mapping = {
        # English-focused local models
        'all-MiniLM-L6-v2': (EmbeddingModel.MINILM_L6_V2, EmbeddingProvider.SENTENCE_TRANSFORMERS),
        'all-MiniLM-L12-v2': (EmbeddingModel.MINILM_L12_V2, EmbeddingProvider.SENTENCE_TRANSFORMERS),
        # Multilingual local models
        'paraphrase-multilingual-MiniLM-L12-v2': (EmbeddingModel.PARAPHRASE_MULTILINGUAL_MINILM_L12_V2, EmbeddingProvider.SENTENCE_TRANSFORMERS),
        'distiluse-base-multilingual-cased': (EmbeddingModel.DISTILUSE_BASE_MULTILINGUAL_CASED, EmbeddingProvider.SENTENCE_TRANSFORMERS),
        'sentence-transformers/LaBSE': (EmbeddingModel.LABSE, EmbeddingProvider.SENTENCE_TRANSFORMERS),
        # OpenAI models (multilingual by default)
        'text-embedding-3-small': (EmbeddingModel.TEXT_EMBEDDING_3_SMALL, EmbeddingProvider.OPENAI),
        'text-embedding-3-large': (EmbeddingModel.TEXT_EMBEDDING_3_LARGE, EmbeddingProvider.OPENAI),
        'text-embedding-ada-002': (EmbeddingModel.TEXT_EMBEDDING_ADA_002, EmbeddingProvider.OPENAI),
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_mapping.keys())}")
    
    model, provider = model_mapping[model_name]
    
    config = EmbeddingConfig(
        model=model,
        provider=provider,
        batch_size=batch_size,
        api_key=api_key
    )
    
    return EmbeddingFunction(config)


def compare_embedding_models(
    texts: List[str], 
    models: List[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Compare performance of different embedding models."""
    
    if models is None:
        # Default to multilingual models for Indonesian+English corpus
        models = ['paraphrase-multilingual-MiniLM-L12-v2', 'all-MiniLM-L6-v2']
    
    results = {}
    
    for model_name in models:
        print(f"Testing {model_name}...")
        
        try:
            embedding_func = create_embedding_function(model_name, api_key)
            
            # Test embedding
            start_time = time.time()
            embeddings = embedding_func.embed_text(texts)
            end_time = time.time()
            
            # Collect results
            metrics = embedding_func.get_metrics()
            metrics['test_time'] = end_time - start_time
            metrics['embedding_dimension'] = embedding_func.get_dimension()
            metrics['texts_processed'] = len(texts)
            
            results[model_name] = metrics
            
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    return results


# Pre-configured embedding functions for common use cases
def get_fast_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get fast local embedding function for testing."""
    return create_embedding_function('all-MiniLM-L6-v2', api_key)


def get_production_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get high-quality OpenAI embedding function for production."""
    return create_embedding_function('text-embedding-3-large', api_key)


def get_balanced_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get balanced OpenAI embedding function (good quality, lower cost)."""
    return create_embedding_function('text-embedding-3-small', api_key)


def get_multilingual_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get fast multilingual embedding function for Indonesian+English (recommended)."""
    return create_embedding_function('paraphrase-multilingual-MiniLM-L12-v2', api_key)


def get_best_multilingual_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get high-quality multilingual embedding function using LaBSE."""
    return create_embedding_function('sentence-transformers/LaBSE', api_key)


def get_production_multilingual_embedding_function(api_key: Optional[str] = None) -> EmbeddingFunction:
    """Get production-grade multilingual embedding using OpenAI (requires API key)."""
    return create_embedding_function('text-embedding-3-large', api_key)