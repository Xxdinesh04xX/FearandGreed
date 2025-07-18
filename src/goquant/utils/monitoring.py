"""
System monitoring and health check utilities.
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .logger import get_logger


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float
    timestamp: datetime


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: str  # 'healthy', 'warning', 'error'
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class SystemMonitor:
    """
    System monitoring and health checking utility.
    
    Monitors system resources, component health, and performance metrics.
    """
    
    def __init__(self):
        """Initialize the system monitor."""
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self.component_checks = {}
        self.metrics_history = []
        self.max_history_size = 1000
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                process_count=process_count,
                uptime_seconds=uptime_seconds,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_io={},
                process_count=0,
                uptime_seconds=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def check_component_health(self, 
                                   component_name: str,
                                   check_function) -> ComponentHealth:
        """
        Check the health of a system component.
        
        Args:
            component_name: Name of the component
            check_function: Async function that returns health status
            
        Returns:
            ComponentHealth object
        """
        start_time = time.time()
        
        try:
            # Run the health check
            result = await check_function()
            response_time_ms = (time.time() - start_time) * 1000
            
            if result.get('status') == 'healthy':
                health = ComponentHealth(
                    name=component_name,
                    status='healthy',
                    last_check=datetime.utcnow(),
                    response_time_ms=response_time_ms,
                    metrics=result.get('metrics')
                )
            else:
                health = ComponentHealth(
                    name=component_name,
                    status='warning',
                    last_check=datetime.utcnow(),
                    response_time_ms=response_time_ms,
                    error_message=result.get('error'),
                    metrics=result.get('metrics')
                )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            health = ComponentHealth(
                name=component_name,
                status='error',
                last_check=datetime.utcnow(),
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
        
        # Store the health check result
        self.component_checks[component_name] = health
        
        return health
    
    async def check_database_health(self, db_manager) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Simple connectivity test
            async with db_manager.get_async_session() as session:
                # Execute a simple query
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'metrics': {
                    'response_time_ms': response_time,
                    'connection_pool_size': getattr(db_manager.async_engine.pool, 'size', 0),
                    'checked_out_connections': getattr(db_manager.async_engine.pool, 'checkedout', 0)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_sentiment_model_health(self, sentiment_analyzer) -> Dict[str, Any]:
        """Check sentiment analysis model health."""
        try:
            if not sentiment_analyzer._initialized:
                return {
                    'status': 'error',
                    'error': 'Sentiment analyzer not initialized'
                }
            
            start_time = time.time()
            
            # Test sentiment analysis
            test_result = await sentiment_analyzer.analyze_text("The market is performing well today.")
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'metrics': {
                    'response_time_ms': response_time,
                    'model_name': sentiment_analyzer.sentiment_model.model_name,
                    'device': sentiment_analyzer.sentiment_model.device,
                    'test_confidence': test_result.confidence
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_data_collector_health(self, data_collector) -> Dict[str, Any]:
        """Check data collector health."""
        try:
            status = data_collector.get_collection_status()
            
            # Check if any collectors have errors
            total_errors = sum(source.get('error_count', 0) for source in status.values())
            active_collectors = sum(1 for source in status.values() if source.get('initialized', False))
            
            if total_errors > 10:  # Threshold for warnings
                return {
                    'status': 'warning',
                    'error': f'High error count: {total_errors}',
                    'metrics': {
                        'active_collectors': active_collectors,
                        'total_errors': total_errors,
                        'collectors': status
                    }
                }
            else:
                return {
                    'status': 'healthy',
                    'metrics': {
                        'active_collectors': active_collectors,
                        'total_errors': total_errors,
                        'collectors': status
                    }
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.component_checks:
            return {
                'overall_status': 'unknown',
                'components': {},
                'last_check': None
            }
        
        # Determine overall status
        statuses = [check.status for check in self.component_checks.values()]
        
        if 'error' in statuses:
            overall_status = 'error'
        elif 'warning' in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # Get latest check time
        last_check = max(check.last_check for check in self.component_checks.values())
        
        return {
            'overall_status': overall_status,
            'components': {name: {
                'status': check.status,
                'last_check': check.last_check.isoformat(),
                'response_time_ms': check.response_time_ms,
                'error_message': check.error_message
            } for name, check in self.component_checks.items()},
            'last_check': last_check.isoformat(),
            'system_metrics': self.get_system_metrics().__dict__
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in recent_metrics],
            'cpu_percent': [m.cpu_percent for m in recent_metrics],
            'memory_percent': [m.memory_percent for m in recent_metrics],
            'disk_usage_percent': [m.disk_usage_percent for m in recent_metrics],
            'response_times': [
                check.response_time_ms for check in self.component_checks.values()
                if check.response_time_ms is not None
            ]
        }


# Global monitor instance
system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return system_monitor
