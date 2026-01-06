import atexit
import threading
from threading import Lock
from typing import Optional, Dict, Set, Callable

from .http_utils import http_post
from .utils import TWINKLE_SERVER_URL


class HeartbeatManager:
    """Manages heartbeat threads for processors, models, and samplers.
    
    This class provides automatic heartbeat management with these features:
    - Global thread for processor heartbeats (sent every 30 seconds)
    - Per-adapter threads for model/sampler heartbeats (sent every 30 seconds)
    - Batch processor heartbeats to reduce network load
    - Automatic cleanup on object destruction
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.server_url = TWINKLE_SERVER_URL
        
        # Processor heartbeat management
        self.processor_ids: Set[str] = set()
        self.processor_lock = Lock()
        self.processor_thread: Optional[threading.Thread] = None
        self.processor_stop_event = threading.Event()
        
        # Adapter heartbeat management (for models/samplers)
        self.adapter_threads: Dict[str, threading.Thread] = {}
        self.adapter_stop_events: Dict[str, threading.Event] = {}
        self.adapter_heartbeat_funcs: Dict[str, Callable] = {}
        self.adapter_lock = Lock()
        
        # Register cleanup on exit
        atexit.register(self.shutdown_all)

    def processor_heartbeat_func(self, processor_id_list: str):
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'processor_id': processor_id_list}
        )
        response.raise_for_status()

    def register_processor(self, processor_id: str):
        """Register a processor for heartbeat monitoring.
        
        Args:
            processor_id: The processor ID to monitor
        """
        with self.processor_lock:
            self.processor_ids.add(processor_id)
            
            # Start processor heartbeat thread if not running
            if self.processor_thread is None or not self.processor_thread.is_alive():
                self.processor_stop_event.clear()
                self.processor_thread = threading.Thread(
                    target=self._processor_heartbeat_loop,
                    daemon=True,
                    name="ProcessorHeartbeatThread"
                )
                self.processor_thread.start()
    
    def unregister_processor(self, processor_id: str):
        """Unregister a processor from heartbeat monitoring.
        
        Args:
            processor_id: The processor ID to remove
        """
        with self.processor_lock:
            self.processor_ids.discard(processor_id)
            
            # Stop thread if no more processors
            if not self.processor_ids and self.processor_thread:
                self.processor_stop_event.set()
    
    def register_adapter(self, adapter_key: str, heartbeat_func: Callable):
        """Register an adapter for heartbeat monitoring.
        
        Args:
            adapter_key: Unique key for the adapter (e.g., "model:adapter_name")
            heartbeat_func: Function to call for heartbeat (no arguments)
        """
        with self.adapter_lock:
            # Stop existing thread if any
            if adapter_key in self.adapter_threads:
                self.adapter_stop_events[adapter_key].set()
                self.adapter_threads[adapter_key].join(timeout=1)
            
            # Create new thread
            self.adapter_heartbeat_funcs[adapter_key] = heartbeat_func
            stop_event = threading.Event()
            self.adapter_stop_events[adapter_key] = stop_event
            
            thread = threading.Thread(
                target=self._adapter_heartbeat_loop,
                args=(adapter_key, stop_event),
                daemon=True,
                name=f"AdapterHeartbeat-{adapter_key}"
            )
            self.adapter_threads[adapter_key] = thread
            thread.start()
    
    def unregister_adapter(self, adapter_key: str):
        """Unregister an adapter from heartbeat monitoring.
        
        Args:
            adapter_key: Unique key for the adapter
        """
        with self.adapter_lock:
            if adapter_key in self.adapter_stop_events:
                self.adapter_stop_events[adapter_key].set()
            
            if adapter_key in self.adapter_threads:
                self.adapter_threads[adapter_key].join(timeout=1)
                del self.adapter_threads[adapter_key]
            
            self.adapter_stop_events.pop(adapter_key, None)
            self.adapter_heartbeat_funcs.pop(adapter_key, None)
    
    def _processor_heartbeat_loop(self):
        """Heartbeat loop for processors (runs every 30 seconds)."""
        while not self.processor_stop_event.wait(timeout=30):
            with self.processor_lock:
                if not self.processor_ids or not self.processor_heartbeat_func:
                    continue
                
                # Batch send processor IDs as comma-separated string
                processor_id_list = ','.join(self.processor_ids)
                
                try:
                    self.processor_heartbeat_func(processor_id_list)
                except Exception as e:
                    print(f"Processor heartbeat error: {e}")
    
    def _adapter_heartbeat_loop(self, adapter_key: str, stop_event: threading.Event):
        """Heartbeat loop for a specific adapter (runs every 30 seconds).
        
        Args:
            adapter_key: Unique key for the adapter
            stop_event: Event to signal thread shutdown
        """
        while not stop_event.wait(timeout=30):
            heartbeat_func = self.adapter_heartbeat_funcs.get(adapter_key)
            if heartbeat_func:
                try:
                    heartbeat_func()
                except Exception as e:
                    print(f"Adapter heartbeat error for {adapter_key}: {e}")
    
    def shutdown_all(self):
        """Shutdown all heartbeat threads."""
        # Stop processor thread
        if self.processor_thread:
            self.processor_stop_event.set()
            self.processor_thread.join(timeout=1)
        
        # Stop all adapter threads
        with self.adapter_lock:
            for stop_event in self.adapter_stop_events.values():
                stop_event.set()
            
            for thread in self.adapter_threads.values():
                thread.join(timeout=1)


# Global heartbeat manager instance
heartbeat_manager = HeartbeatManager()
