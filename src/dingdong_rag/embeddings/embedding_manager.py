import gc
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingModelManager:
    """Singleton manager for embedding models to prevent memory duplication."""
    
    _instance: Optional['EmbeddingModelManager'] = None
    _models: Dict[str, SentenceTransformer] = {}
    
    def __new__(cls) -> 'EmbeddingModelManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        """Get or create embedding model instance."""
        if model_name not in self._models:
            print(f"Loading embedding model: {model_name}")
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]
    
    def clear_model(self, model_name: str) -> None:
        """Remove specific model from memory."""
        if model_name in self._models:
            del self._models[model_name]
            gc.collect()
    
    def clear_all_models(self) -> None:
        """Clear all models from memory."""
        self._models.clear()
        gc.collect()
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names."""
        return list(self._models.keys())
    
    def get_memory_usage(self) -> dict:
        """Get approximate memory usage info."""
        return {
            'loaded_models': len(self._models),
            'model_names': list(self._models.keys())
        }


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Convenience function to get embedding model through singleton."""
    manager = EmbeddingModelManager()
    return manager.get_model(model_name)