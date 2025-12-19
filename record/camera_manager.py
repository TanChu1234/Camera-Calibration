import cv2
import multiprocessing
from camera_process import CameraProcess

class CameraManager:
    def __init__(self):
        self.cameras = {}       # cam_id -> process
        self.queues = {}        # cam_id -> Queue
        self.stop_events = {}   # cam_id -> Event
        self.record_command_queues = {}  # cam_id -> Queue for recording commands
        self.windows_visible = {}  # show/hide
        self.rtsp_urls = {}     # Store RTSP URLs for restart
        self.camera_configs = {}  # Store FPS settings
        self.recording_status = {}  # cam_id -> bool (for UI display)
        self.last_frames = {}  # Cache last frame to avoid blocking
    
    def add_camera(self, cam_id, rtsp_url, target_fps=30):
        """
        Add a camera to the manager
        
        Args:
            cam_id: Unique camera identifier
            rtsp_url: RTSP stream URL
            target_fps: Target frames per second (default: 30)
        """
        self.rtsp_urls[cam_id] = rtsp_url
        self.camera_configs[cam_id] = {
            'target_fps': target_fps
        }
        
        # Larger queue to prevent blocking
        self.queues[cam_id] = multiprocessing.Queue(maxsize=2)
        self.stop_events[cam_id] = multiprocessing.Event()
        self.record_command_queues[cam_id] = multiprocessing.Queue()
        
        self.cameras[cam_id] = CameraProcess(
            cam_id,
            rtsp_url,
            self.queues[cam_id],
            self.stop_events[cam_id],
            target_fps,
            self.record_command_queues[cam_id]
        )
        self.windows_visible[cam_id] = False
        self.recording_status[cam_id] = False
        self.last_frames[cam_id] = None
        print(f"[Manager] Camera {cam_id} added (FPS: {target_fps})")
    
    def start_camera(self, cam_id):
        cam = self.cameras[cam_id]
        # If process is dead, create a new one
        if not cam.is_alive():
            # If it was previously started and stopped, recreate it
            if cam.exitcode is not None:
                print(f"[Manager] Recreating process for camera {cam_id}")
                config = self.camera_configs[cam_id]
                self.cameras[cam_id] = CameraProcess(
                    cam_id,
                    self.rtsp_urls[cam_id],
                    self.queues[cam_id],
                    self.stop_events[cam_id],
                    config['target_fps'],
                    self.record_command_queues[cam_id]
                )
                cam = self.cameras[cam_id]
            
            self.stop_events[cam_id].clear()
            cam.start()
            print(f"[Manager] Camera {cam_id} started")
        
        # Create window immediately (safe)
        cv2.namedWindow(f"Camera {cam_id}", cv2.WINDOW_NORMAL)
        self.windows_visible[cam_id] = True
    
    def stop_camera(self, cam_id):
        print(f"[Manager] Stopping camera {cam_id}")
        
        # Stop recording if active
        if self.recording_status.get(cam_id, False):
            self.stop_recording(cam_id)
        
        self.windows_visible[cam_id] = False
        window_name = f"Camera {cam_id}"
        
        # Check if window exists
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
        except cv2.error:
            pass
        
        # Stop the process
        if cam_id in self.stop_events:
            self.stop_events[cam_id].set()
        if cam_id in self.cameras and self.cameras[cam_id].is_alive():
            self.cameras[cam_id].join(timeout=2)
        
        # Clear the queue
        while not self.queues[cam_id].empty():
            try:
                self.queues[cam_id].get_nowait()
            except:
                break
        
        print(f"[Manager] Camera {cam_id} stopped")
    
    def start_recording(self, cam_id):
        """Start recording for a camera"""
        if cam_id not in self.record_command_queues:
            print(f"[Manager] Camera {cam_id} not found")
            return
        
        try:
            self.record_command_queues[cam_id].put_nowait("start")
            print(f"[Manager] Sent start recording command to camera {cam_id}")
        except:
            print(f"[Manager] Failed to send start recording command to camera {cam_id}")
    
    def stop_recording(self, cam_id):
        """Stop recording for a camera"""
        if cam_id not in self.record_command_queues:
            print(f"[Manager] Camera {cam_id} not found")
            return
        
        try:
            self.record_command_queues[cam_id].put_nowait("stop")
            print(f"[Manager] Sent stop recording command to camera {cam_id}")
        except:
            print(f"[Manager] Failed to send stop recording command to camera {cam_id}")
    
    def is_recording(self, cam_id):
        """Check if camera is currently recording"""
        return self.recording_status.get(cam_id, False)
    
    def show_frames(self):
        """
        Call this from PySide6 QTimer every ~30ms
        Optimized to be non-blocking and fast
        """
        for cam_id, q in self.queues.items():
            if not self.windows_visible[cam_id]:
                continue
            
            # Get all available frames (use latest, discard old)
            latest_data = None
            frames_retrieved = 0
            
            while not q.empty() and frames_retrieved < 5:  # Limit to prevent blocking
                try:
                    latest_data = q.get_nowait()
                    frames_retrieved += 1
                except:
                    break
            
            # Only update display if we got a new frame
            if latest_data is not None:
                try:
                    # Handle both formats
                    if isinstance(latest_data, tuple):
                        frame, is_recording = latest_data
                        self.recording_status[cam_id] = is_recording
                    else:
                        frame = latest_data
                        self.recording_status[cam_id] = False
                    
                    self.last_frames[cam_id] = frame
                    cv2.imshow(f"Camera {cam_id}", frame)
                except Exception as e:
                    print(f"[Manager] Error displaying frame for camera {cam_id}: {e}")
        
        # Single waitKey for all windows (more efficient)
        cv2.waitKey(1)
    
    def stop_all(self):
        print("[Manager] Stopping all cameras...")
        for cam_id in list(self.cameras.keys()):
            self.stop_camera(cam_id)
        cv2.destroyAllWindows()