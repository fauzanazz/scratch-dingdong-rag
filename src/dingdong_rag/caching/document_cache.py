import os
import json
import hashlib
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DocumentCacheEntry:
    """Cached document entry with checksum validation."""
    file_path: str
    checksum: str
    checksum_type: str  # 'md5', 'sha256', etc.
    content: str
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]
    cached_at: str  # ISO timestamp
    file_size: int
    file_mtime: float  # File modification time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentCacheEntry':
        """Create from dictionary."""
        return cls(**data)


class DocumentCache:
    """Document cache with checksum-based validation."""
    def __init__(self, cache_dir: str = ".cache/documents", 
                 checksum_type: str = "md5", compression: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_type = checksum_type.lower()
        self.compression = compression
        self.index_file = self.cache_dir / "cache_index.json"
        self.entries_dir = self.cache_dir / "entries"
        self.entries_dir.mkdir(exist_ok=True)
        self.index = self._load_cache_index()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'size_saved_mb': 0.0
        }
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Cache index corrupted, rebuilding: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache index: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_func = hashlib.md5() if self.checksum_type == 'md5' else hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                while chunk := f.read(8192):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            raise RuntimeError(f"Failed to calculate checksum for {file_path}: {e}")
    
    def _get_cache_entry_path(self, file_path: str) -> Path:
        """Get cache entry file path."""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()
        extension = ".pkl.gz" if self.compression else ".pkl"
        return self.entries_dir / f"{path_hash}{extension}"
    
    def _save_cache_entry(self, entry: DocumentCacheEntry):
        """Save cache entry to disk."""
        entry_path = self._get_cache_entry_path(entry.file_path)
        
        try:
            data = entry.to_dict()
            
            if self.compression:
                with gzip.open(entry_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(entry_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
        except Exception as e:
            print(f"Failed to save cache entry for {entry.file_path}: {e}")
    
    def _load_cache_entry(self, file_path: str) -> Optional[DocumentCacheEntry]:
        """Load cache entry from disk."""
        entry_path = self._get_cache_entry_path(file_path)
        
        if not entry_path.exists():
            return None
            
        try:
            if self.compression:
                with gzip.open(entry_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(entry_path, 'rb') as f:
                    data = pickle.load(f)
                    
            return DocumentCacheEntry.from_dict(data)
            
        except Exception as e:
            print(f"Failed to load cache entry for {file_path}: {e}")
            try:
                entry_path.unlink()
            except:
                pass
            return None
    
    def _is_file_changed(self, file_path: str, cached_entry: DocumentCacheEntry) -> bool:
        """Check if file has changed since caching."""
        try:
            stat = os.stat(file_path)
            if stat.st_size != cached_entry.file_size:
                return True
            if abs(stat.st_mtime - cached_entry.file_mtime) > 1:
                return True
            current_checksum = self._calculate_checksum(file_path)
            return current_checksum != cached_entry.checksum
            
        except Exception:
            return True
    
    def get_cached_result(self, file_path: str) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Get cached parsing result if valid."""
        abs_path = os.path.abspath(file_path)
        
        if abs_path not in self.index:
            self.stats['misses'] += 1
            return None
        cached_entry = self._load_cache_entry(abs_path)
        if not cached_entry:
            del self.index[abs_path]
            self.stats['misses'] += 1
            return None
        if self._is_file_changed(abs_path, cached_entry):
            print(f"Cache invalidated: {Path(file_path).name} (file changed)")
            self._invalidate_cache(abs_path)
            self.stats['invalidations'] += 1
            return None
        print(f"Cache hit: {Path(file_path).name} ({cached_entry.checksum[:8]}...)")
        self.stats['hits'] += 1
        
        # Estimate size saved (rough approximation)
        content_size_mb = len(cached_entry.content.encode('utf-8')) / 1024 / 1024
        self.stats['size_saved_mb'] += content_size_mb
        
        return cached_entry.content, cached_entry.metadata, cached_entry.processing_stats
    
    def cache_result(self, file_path: str, content: str, metadata: Dict[str, Any], 
                    processing_stats: Optional[Dict[str, Any]] = None):
        """Cache parsing result with checksum."""
        abs_path = os.path.abspath(file_path)
        
        try:
            # Calculate checksum and file stats
            checksum = self._calculate_checksum(abs_path)
            stat = os.stat(abs_path)
            
            # Create cache entry
            entry = DocumentCacheEntry(
                file_path=abs_path,
                checksum=checksum,
                checksum_type=self.checksum_type,
                content=content,
                metadata=metadata,
                processing_stats=processing_stats or {},
                cached_at=datetime.now().isoformat(),
                file_size=stat.st_size,
                file_mtime=stat.st_mtime
            )
            
            # Save entry to disk
            self._save_cache_entry(entry)
            
            # Update index
            self.index[abs_path] = {
                'checksum': checksum,
                'cached_at': entry.cached_at,
                'file_size': stat.st_size
            }
            
            self._save_cache_index()
            
            print(f"  💾 Cached: {Path(file_path).name} ({checksum[:8]}...)")
            
        except Exception as e:
            print(f"Failed to cache {file_path}: {e}")
    
    def _invalidate_cache(self, file_path: str):
        """Remove file from cache."""
        abs_path = os.path.abspath(file_path)
        
        # Remove from index
        if abs_path in self.index:
            del self.index[abs_path]
        
        # Remove cache entry file
        entry_path = self._get_cache_entry_path(abs_path)
        try:
            entry_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    def clear_cache(self):
        """Clear all cached entries."""
        try:
            # Remove all entry files
            for entry_file in self.entries_dir.glob("*"):
                entry_file.unlink()
            
            # Clear index
            self.index.clear()
            self._save_cache_index()
            
            print("🧹 Cache cleared")
            
        except Exception as e:
            print(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Calculate cache size
            cache_size_mb = 0
            entry_count = 0
            
            for entry_file in self.entries_dir.glob("*"):
                cache_size_mb += entry_file.stat().st_size / 1024 / 1024
                entry_count += 1
            
            return {
                'total_entries': len(self.index),
                'cache_size_mb': cache_size_mb,
                'cache_directory': str(self.cache_dir),
                'compression_enabled': self.compression,
                'checksum_type': self.checksum_type,
                'session_stats': self.stats.copy()
            }
            
        except Exception as e:
            return {'error': str(e), 'session_stats': self.stats.copy()}
    
    def print_cache_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        
        print(f"\n📊 DOCUMENT CACHE STATISTICS")
        print("=" * 40)
        print(f"Cache directory: {stats.get('cache_directory', 'Unknown')}")
        print(f"Total entries: {stats.get('total_entries', 0)}")
        print(f"Cache size: {stats.get('cache_size_mb', 0):.1f} MB")
        print(f"Compression: {'Enabled' if stats.get('compression_enabled') else 'Disabled'}")
        print(f"Checksum type: {stats.get('checksum_type', 'Unknown')}")
        
        session_stats = stats.get('session_stats', {})
        print(f"\nSession statistics:")
        print(f"  Cache hits: {session_stats.get('hits', 0)}")
        print(f"  Cache misses: {session_stats.get('misses', 0)}")
        print(f"  Cache invalidations: {session_stats.get('invalidations', 0)}")
        print(f"  Processing saved: {session_stats.get('size_saved_mb', 0):.1f} MB")
        
        hit_rate = 0
        total_requests = session_stats.get('hits', 0) + session_stats.get('misses', 0)
        if total_requests > 0:
            hit_rate = session_stats.get('hits', 0) / total_requests * 100
            print(f"  Hit rate: {hit_rate:.1f}%")
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove cache entries older than specified days."""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        entries_to_remove = []
        
        for file_path, index_data in self.index.items():
            try:
                cached_at = datetime.fromisoformat(index_data['cached_at'])
                if cached_at.timestamp() < cutoff_time:
                    entries_to_remove.append(file_path)
            except Exception:
                # Remove entries with invalid timestamps
                entries_to_remove.append(file_path)
        
        for file_path in entries_to_remove:
            self._invalidate_cache(file_path)
            removed_count += 1
        
        if removed_count > 0:
            self._save_cache_index()
            print(f"🧹 Cleaned up {removed_count} old cache entries (>{max_age_days} days)")
        
        return removed_count


# Global cache instance
_default_cache: Optional[DocumentCache] = None

def get_document_cache(cache_dir: str = ".cache/documents", 
                      checksum_type: str = "md5",
                      compression: bool = True) -> DocumentCache:
    """Get or create the default document cache instance."""
    global _default_cache
    
    if _default_cache is None:
        _default_cache = DocumentCache(cache_dir, checksum_type, compression)
    
    return _default_cache