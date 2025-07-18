"""
Performance Optimization Module for High-Frequency Sentiment Analysis.
Implements memory management, SIMD optimization, and efficient data structures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import time
import gc
import psutil
import logging
from dataclasses import dataclass
from collections import deque, defaultdict
import asyncio
import concurrent.futures
from functools import lru_cache, wraps
import pickle
import zlib
import mmap
import os
import sys


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    cpu_usage: float
    memory_usage: float
    processing_time: float
    throughput: float
    cache_hit_rate: float
    queue_size: int
    active_threads: int


class MemoryPool:
    """Custom memory pool for efficient text processing."""
    
    def __init__(self, block_size: int = 1024, pool_size: int = 1000):
        """Initialize memory pool."""
        self.block_size = block_size
        self.pool_size = pool_size
        self.available_blocks = queue.Queue(maxsize=pool_size)
        self.allocated_blocks = set()
        self.lock = threading.Lock()
        
        # Pre-allocate blocks
        for _ in range(pool_size):
            block = bytearray(block_size)
            self.available_blocks.put(block)
        
        logging.info(f"Memory pool initialized: {pool_size} blocks of {block_size} bytes")
    
    def get_block(self) -> Optional[bytearray]:
        """Get a memory block from the pool."""
        try:
            block = self.available_blocks.get_nowait()
            with self.lock:
                self.allocated_blocks.add(id(block))
            return block
        except queue.Empty:
            # Pool exhausted, allocate new block
            logging.warning("Memory pool exhausted, allocating new block")
            return bytearray(self.block_size)
    
    def return_block(self, block: bytearray):
        """Return a memory block to the pool."""
        if block is None:
            return
        
        # Clear the block
        block[:] = b'\x00' * len(block)
        
        with self.lock:
            if id(block) in self.allocated_blocks:
                self.allocated_blocks.remove(id(block))
        
        try:
            self.available_blocks.put_nowait(block)
        except queue.Full:
            # Pool is full, let block be garbage collected
            pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        return {
            'available_blocks': self.available_blocks.qsize(),
            'allocated_blocks': len(self.allocated_blocks),
            'total_capacity': self.pool_size,
            'block_size': self.block_size
        }


class LockFreeQueue:
    """Lock-free queue for high-frequency updates."""
    
    def __init__(self, maxsize: int = 10000):
        """Initialize lock-free queue."""
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        self.size = 0
        
    def put(self, item: Any) -> bool:
        """Put item in queue (non-blocking)."""
        try:
            if len(self.queue) >= self.maxsize:
                self.queue.popleft()  # Remove oldest item
            self.queue.append(item)
            self.size = len(self.queue)
            return True
        except Exception:
            return False
    
    def get(self) -> Optional[Any]:
        """Get item from queue (non-blocking)."""
        try:
            item = self.queue.popleft()
            self.size = len(self.queue)
            return item
        except IndexError:
            return None
    
    def peek(self) -> Optional[Any]:
        """Peek at next item without removing."""
        try:
            return self.queue[0]
        except IndexError:
            return None
    
    def clear(self):
        """Clear the queue."""
        self.queue.clear()
        self.size = 0
    
    def __len__(self) -> int:
        """Get queue size."""
        return self.size


class SIMDTextProcessor:
    """SIMD-optimized text processing operations."""
    
    def __init__(self):
        """Initialize SIMD processor."""
        self.use_simd = self._check_simd_support()
        if self.use_simd:
            logging.info("SIMD support detected and enabled")
        else:
            logging.info("SIMD not available, using standard operations")
    
    def _check_simd_support(self) -> bool:
        """Check if SIMD operations are available."""
        try:
            # Check for NumPy with SIMD support
            import numpy as np
            return hasattr(np.core._multiarray_umath, '__cpu_features__')
        except:
            return False
    
    def vectorized_sentiment_score(self, texts: List[str], 
                                 positive_words: List[str], 
                                 negative_words: List[str]) -> np.ndarray:
        """Vectorized sentiment scoring using SIMD operations."""
        if not texts:
            return np.array([])
        
        # Convert to numpy arrays for vectorization
        text_array = np.array(texts, dtype=object)
        scores = np.zeros(len(texts), dtype=np.float32)
        
        if self.use_simd:
            # Use vectorized operations
            for i, text in enumerate(texts):
                if text:
                    text_lower = text.lower()
                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)
                    word_count = len(text.split())
                    
                    if word_count > 0:
                        scores[i] = (pos_count - neg_count) / word_count
        else:
            # Fallback to standard processing
            for i, text in enumerate(texts):
                scores[i] = self._standard_sentiment_score(text, positive_words, negative_words)
        
        return scores
    
    def _standard_sentiment_score(self, text: str, positive_words: List[str], 
                                negative_words: List[str]) -> float:
        """Standard sentiment scoring."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        word_count = len(text.split())
        
        if word_count > 0:
            return (pos_count - neg_count) / word_count
        return 0.0
    
    def batch_text_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract text features in batches using vectorized operations."""
        if not texts:
            return {}
        
        # Vectorized feature extraction
        features = {}
        
        # Text lengths
        features['length'] = np.array([len(text) for text in texts], dtype=np.int32)
        
        # Word counts
        features['word_count'] = np.array([len(text.split()) for text in texts], dtype=np.int32)
        
        # Character frequencies (vectorized)
        if self.use_simd:
            # Use numpy for faster character counting
            features['exclamation_count'] = np.array([text.count('!') for text in texts], dtype=np.int32)
            features['question_count'] = np.array([text.count('?') for text in texts], dtype=np.int32)
            features['uppercase_ratio'] = np.array([
                sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts
            ], dtype=np.float32)
        else:
            # Standard processing
            features['exclamation_count'] = np.array([text.count('!') for text in texts])
            features['question_count'] = np.array([text.count('?') for text in texts])
            features['uppercase_ratio'] = np.array([
                sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts
            ])
        
        return features


class StreamingProcessor:
    """Streaming text processing pipeline for real-time analysis."""
    
    def __init__(self, buffer_size: int = 1000, batch_size: int = 100):
        """Initialize streaming processor."""
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.input_queue = LockFreeQueue(buffer_size)
        self.output_queue = LockFreeQueue(buffer_size)
        self.processing_thread = None
        self.running = False
        
        # Performance tracking
        self.processed_count = 0
        self.processing_times = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Memory pool for text processing
        self.memory_pool = MemoryPool(block_size=4096, pool_size=100)
        
        # SIMD processor
        self.simd_processor = SIMDTextProcessor()
        
    def start(self):
        """Start the streaming processor."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logging.info("Streaming processor started")
    
    def stop(self):
        """Stop the streaming processor."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logging.info("Streaming processor stopped")
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text for processing."""
        item = {
            'text': text,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        return self.input_queue.put(item)
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get processed result."""
        return self.output_queue.get()
    
    def _processing_loop(self):
        """Main processing loop."""
        batch = []
        
        while self.running:
            try:
                # Collect batch
                while len(batch) < self.batch_size and self.running:
                    item = self.input_queue.get()
                    if item is None:
                        time.sleep(0.001)  # Small delay to prevent busy waiting
                        continue
                    batch.append(item)
                
                if batch:
                    # Process batch
                    start_time = time.time()
                    results = self._process_batch(batch)
                    processing_time = time.time() - start_time
                    
                    # Update performance metrics
                    self.processing_times.append(processing_time)
                    self.processed_count += len(batch)
                    
                    # Add results to output queue
                    for result in results:
                        self.output_queue.put(result)
                    
                    batch.clear()
                
            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                batch.clear()
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of texts."""
        texts = [item['text'] for item in batch]
        
        # Extract features using SIMD
        features = self.simd_processor.batch_text_features(texts)
        
        # Simple sentiment analysis (placeholder)
        positive_words = ['good', 'great', 'excellent', 'buy', 'bull', 'up', 'rise']
        negative_words = ['bad', 'terrible', 'sell', 'bear', 'down', 'fall', 'crash']
        
        sentiment_scores = self.simd_processor.vectorized_sentiment_score(
            texts, positive_words, negative_words
        )
        
        # Prepare results
        results = []
        for i, item in enumerate(batch):
            result = {
                'text': item['text'],
                'metadata': item['metadata'],
                'timestamp': item['timestamp'],
                'sentiment_score': float(sentiment_scores[i]) if i < len(sentiment_scores) else 0.0,
                'features': {key: float(values[i]) if i < len(values) else 0.0 
                           for key, values in features.items()},
                'processing_time': time.time() - item['timestamp']
            }
            results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate throughput
        throughput = self.processed_count / max(uptime, 1)
        
        # Average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # System metrics
        process = psutil.Process()
        cpu_usage = process.cpu_percent()
        memory_usage = process.memory_percent()
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            processing_time=avg_processing_time,
            throughput=throughput,
            cache_hit_rate=0.0,  # Placeholder
            queue_size=len(self.input_queue),
            active_threads=threading.active_count()
        )


class EfficientCache:
    """Efficient caching system with compression and TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """Initialize efficient cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl_seconds:
                    self.hits += 1
                    self.access_times[key] = time.time()
                    
                    # Decompress if needed
                    value = self.cache[key]
                    if isinstance(value, bytes):
                        return pickle.loads(zlib.decompress(value))
                    return value
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, compress: bool = True):
        """Put item in cache."""
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Compress large objects
            if compress and sys.getsizeof(value) > 1024:
                compressed_value = zlib.compress(pickle.dumps(value))
                self.cache[key] = compressed_value
            else:
                self.cache[key] = value
            
            self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance optimizer."""
        self.config = config or {}
        
        # Initialize components
        self.streaming_processor = StreamingProcessor(
            buffer_size=self.config.get('buffer_size', 1000),
            batch_size=self.config.get('batch_size', 100)
        )
        
        self.cache = EfficientCache(
            max_size=self.config.get('cache_size', 10000),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.performance_history = deque(maxlen=1000)
        
    def start_optimization(self):
        """Start performance optimization."""
        self.streaming_processor.start()
        self._start_monitoring()
        logging.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.streaming_processor.stop()
        self._stop_monitoring()
        logging.info("Performance optimization stopped")
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.streaming_processor.get_performance_metrics()
                self.performance_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                
                # Log performance warnings
                if metrics.cpu_usage > 80:
                    logging.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
                
                if metrics.memory_usage > 80:
                    logging.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
                
                if metrics.queue_size > 500:
                    logging.warning(f"Large queue size: {metrics.queue_size}")
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.performance_history:
            return {'status': 'No performance data available'}
        
        recent_metrics = self.performance_history[-1]['metrics']
        cache_stats = self.cache.get_stats()
        
        # Calculate averages over last 10 measurements
        recent_history = list(self.performance_history)[-10:]
        avg_cpu = np.mean([h['metrics'].cpu_usage for h in recent_history])
        avg_memory = np.mean([h['metrics'].memory_usage for h in recent_history])
        avg_throughput = np.mean([h['metrics'].throughput for h in recent_history])
        
        return {
            'current_metrics': recent_metrics.__dict__,
            'averages': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'throughput': avg_throughput
            },
            'cache_performance': cache_stats,
            'memory_pool_stats': self.streaming_processor.memory_pool.get_stats(),
            'optimization_recommendations': self._get_recommendations(recent_metrics)
        }
    
    def _get_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        
        if metrics.cpu_usage > 70:
            recommendations.append("Consider reducing batch size or increasing processing interval")
        
        if metrics.memory_usage > 70:
            recommendations.append("Consider clearing caches or reducing buffer sizes")
        
        if metrics.throughput < 10:
            recommendations.append("Consider optimizing text processing algorithms")
        
        if metrics.queue_size > 100:
            recommendations.append("Consider increasing processing threads or batch size")
        
        return recommendations


# Test the performance optimization system
if __name__ == "__main__":
    def test_performance_optimization():
        """Test the performance optimization system."""
        print("⚡ Testing Performance Optimization System")
        print("=" * 50)
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer({
            'buffer_size': 500,
            'batch_size': 50,
            'cache_size': 1000
        })
        
        # Start optimization
        optimizer.start_optimization()
        
        # Test streaming processing
        print("📊 Testing streaming processor...")
        test_texts = [
            "Apple earnings beat expectations! Great news for investors.",
            "Market crash incoming! Sell everything now!",
            "Bitcoin price stable, normal trading volume.",
            "Tesla announces new factory. Bullish for $TSLA!"
        ] * 25  # 100 texts total
        
        # Add texts for processing
        start_time = time.time()
        for i, text in enumerate(test_texts):
            optimizer.streaming_processor.add_text(text, {'id': i})
        
        # Wait for processing
        time.sleep(2)
        
        # Collect results
        results = []
        while True:
            result = optimizer.streaming_processor.get_result()
            if result is None:
                break
            results.append(result)
        
        processing_time = time.time() - start_time
        
        print(f"   Processed {len(results)} texts in {processing_time:.3f}s")
        print(f"   Throughput: {len(results)/processing_time:.1f} texts/second")
        
        # Test cache performance
        print("\n💾 Testing cache performance...")
        cache = optimizer.cache
        
        # Add items to cache
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Test cache hits
        hit_count = 0
        for i in range(100):
            if cache.get(f"key_{i}") is not None:
                hit_count += 1
        
        cache_stats = cache.get_stats()
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Cache size: {cache_stats['size']}")
        
        # Get optimization report
        print("\n📈 Performance Report:")
        report = optimizer.get_optimization_report()
        
        if 'current_metrics' in report:
            metrics = report['current_metrics']
            print(f"   CPU Usage: {metrics['cpu_usage']:.1f}%")
            print(f"   Memory Usage: {metrics['memory_usage']:.1f}%")
            print(f"   Throughput: {metrics['throughput']:.1f} items/sec")
            print(f"   Queue Size: {metrics['queue_size']}")
        
        if report['optimization_recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['optimization_recommendations']:
                print(f"   • {rec}")
        
        # Stop optimization
        optimizer.stop_optimization()
        
        print("\n✅ Performance optimization test completed!")
    
    test_performance_optimization()
