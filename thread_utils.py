import threading
import queue
import time
from typing import Callable, Any, Dict, Optional, List, Tuple
import traceback
import cv2
import io
import struct
from PIL import Image
import numpy as np
import subprocess


class ThreadCommand:
    """Command object used to communicate between threads."""
    def __init__(self, command_type: str, data: Any = None, callback: Callable = None):
        self.command_type = command_type
        self.data = data
        self.callback = callback
        self.result = None
        self.exception = None
        self.completed = threading.Event()
    
    def complete(self, result=None, exception=None):
        """Mark the command as completed with optional result or exception."""
        self.result = result
        self.exception = exception
        self.completed.set()
        
    def wait(self, timeout=None) -> bool:
        """Wait for the command to complete and return if it succeeded."""
        if self.completed.wait(timeout):
            return self.exception is None
        return False


class ThreadPool:
    """A simple thread pool for managing worker threads."""
    def __init__(self, num_workers: int = 2, name_prefix: str = "Worker"):
        self.command_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.name_prefix = name_prefix
        
        # Create and start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"{name_prefix}-{i+1}",
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
    
    def _worker_loop(self):
        """Worker thread main loop."""
        thread_name = threading.current_thread().name
        print(f"Thread {thread_name} started")
        
        while not self.stop_event.is_set():
            try:
                # Get a command from the queue with a timeout
                # This allows the thread to check the stop event periodically
                try:
                    cmd = self.command_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process the command
                try:
                    result = None
                    if callable(cmd.data):
                        # If data is a callable, execute it
                        result = cmd.data()
                    
                    # Command completed successfully
                    cmd.complete(result=result)
                    
                    # If there's a callback, execute it in the same worker thread
                    if cmd.callback and callable(cmd.callback):
                        try:
                            cmd.callback(result)
                        except Exception as callback_error:
                            print(f"Error in callback: {callback_error}")
                            traceback.print_exc()
                except Exception as e:
                    # Command failed with an exception
                    print(f"Error in worker thread {thread_name}: {e}")
                    traceback.print_exc()
                    cmd.complete(exception=e)
                
                # Mark task as done
                self.command_queue.task_done()
                
            except Exception as e:
                print(f"Unexpected error in worker thread {thread_name}: {e}")
                traceback.print_exc()
        
        print(f"Thread {thread_name} stopped")
    
    def submit(self, command_type: str, data: Any = None, callback: Callable = None) -> ThreadCommand:
        """Submit a command to be executed by a worker thread."""
        cmd = ThreadCommand(command_type, data, callback)
        self.command_queue.put(cmd)
        return cmd
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        self.stop_event.set()
        
        if wait:
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=2.0)  # Give threads time to finish
        
        # Clear the queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
            except queue.Empty:
                break


class ProgressTracker:
    """A thread-safe progress tracking utility."""
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.completed_time = None
        self.status_message = ""
        self.lock = threading.RLock()
        self.callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable):
        """Register a callback to be called when progress updates."""
        with self.lock:
            self.callbacks.append(callback)
    
    def start(self, total_steps: int = None):
        """Start or restart the progress tracker."""
        with self.lock:
            if total_steps is not None:
                self.total_steps = total_steps
            self.current_step = 0
            self.start_time = time.time()
            self.completed_time = None
            self.status_message = "Started"
            self._notify_callbacks()
    
    def update(self, step: int = None, increment: int = None, message: str = None):
        """Update the progress."""
        with self.lock:
            if step is not None:
                self.current_step = min(step, self.total_steps)
            elif increment is not None:
                self.current_step = min(self.current_step + increment, self.total_steps)
            
            if message is not None:
                self.status_message = message
            
            self._notify_callbacks()
    
    def complete(self, message: str = "Completed"):
        """Mark the progress as complete."""
        with self.lock:
            self.current_step = self.total_steps
            self.completed_time = time.time()
            self.status_message = message
            self._notify_callbacks()
    
    def get_progress(self) -> Tuple[int, int, float, str]:
        """Get the current progress as (current_step, total_steps, percentage, message)."""
        with self.lock:
            percentage = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
            return (self.current_step, self.total_steps, percentage, self.status_message)
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get the elapsed time in seconds."""
        with self.lock:
            if self.start_time is None:
                return None
            
            end_time = self.completed_time if self.completed_time else time.time()
            return end_time - self.start_time
    
    def is_complete(self) -> bool:
        """Check if the progress is complete."""
        with self.lock:
            return self.current_step >= self.total_steps
    
    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        progress_data = self.get_progress()
        for callback in self.callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                print(f"Error in progress callback: {e}")






class CameraThread:
    """A dedicated thread for managing ffmpeg-based camera operations."""
    def __init__(self, camera_init_func, frame_callback):
        self.camera_init_func = camera_init_func
        self.frame_callback = frame_callback
        self.camera = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.frame_interval = 1 / 30  # ~30 FPS

    def start(self):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.thread = threading.Thread(target=self._camera_loop, name="CameraThread", daemon=True)
            self.thread.start()
            return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._terminate_camera()

    def _terminate_camera(self):
        if self.camera and hasattr(self.camera, 'terminate'):
            try:
                print("Terminating ffmpeg...")
                self.camera.terminate()
                try:
                    self.camera.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("FFmpeg did not exit in time. Forcing kill...")
                    self.camera.kill()
                    self.camera.wait(timeout=1)
                print("? ffmpeg terminated")
            except Exception as e:
                print(f"Error terminating ffmpeg: {e}")
            self.camera = None

    def _drain_stderr(self, stream):
        try:
            for _ in iter(stream.readline, b''):
                pass  # You can log this if you want
        except Exception as e:
            print(f"Error draining stderr: {e}")

    def _camera_loop(self):
        buffer = b""
        retries = 0
        max_retries = 5
    
        while self.running:
            try:
                if not self.camera or self.camera.poll() is not None:
                    print("?? ffmpeg process dead or not running. Restarting...")
                    self._terminate_camera()
                    self.camera = self.camera_init_func()
                    retries += 1
                    if retries >= max_retries:
                        print("? Too many camera init failures.")
                        break
                    time.sleep(1)
                    continue
    
                chunk = self.camera.stdout.read(4096)
                if not chunk:
                    print("?? No data from ffmpeg")
                    retries += 1
                    time.sleep(1)
                    continue
    
                buffer += chunk
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start)
                    if start != -1 and end != -1 and end > start:
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
    
                        # Decode JPEG
                        image_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
                        if frame is not None and self.frame_callback:
                            self.frame_callback(frame)
                    else:
                        # Wait for more data
                        break
    
                retries = 0
                time.sleep(self.frame_interval)
    
            except Exception as e:
                print(f"Camera thread error: {e}")
                traceback.print_exc()
                self._terminate_camera()
                retries += 1
                time.sleep(1)
    
    
    def is_running(self):
        with self.lock:
            return self.running and self.thread and self.thread.is_alive()
