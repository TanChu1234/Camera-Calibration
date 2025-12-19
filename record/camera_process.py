import cv2
import multiprocessing
import time
import os
from datetime import datetime

class CameraProcess(multiprocessing.Process):
    def __init__(self, cam_id, rtsp_url, frame_queue, stop_event, target_fps=30, 
                 record_command_queue=None):
        """
        Camera process with video recording capability
        
        Args:
            cam_id: Camera identifier
            rtsp_url: RTSP stream URL
            frame_queue: Queue for sending frames to main process
            stop_event: Event to signal process stop
            target_fps: Target frames per second
            record_command_queue: Queue for receiving recording commands (start/stop)
        """
        super().__init__()
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.target_fps = target_fps
        self.record_command_queue = record_command_queue
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recording_filename = None
        
    def start_recording(self, frame_shape):
        """Start video recording"""
        if self.is_recording:
            print(f"[Camera {self.cam_id}] Already recording")
            return
        
        # Create recordings directory if it doesn't exist
        recordings_dir = "recordings"
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"{recordings_dir}/camera_{self.cam_id}_{timestamp}.avi"
        
        # Initialize video writer
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.recording_filename,
            fourcc,
            self.target_fps,
            (width, height)
        )
        
        if self.video_writer.isOpened():
            self.is_recording = True
            self.recording_start_time = time.time()
            print(f"[Camera {self.cam_id}] Started recording: {self.recording_filename}")
        else:
            print(f"[Camera {self.cam_id}] Failed to start recording")
            self.video_writer = None
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            print(f"[Camera {self.cam_id}] Not recording")
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        print(f"[Camera {self.cam_id}] Stopped recording: {self.recording_filename}")
        print(f"[Camera {self.cam_id}] Recording duration: {duration:.1f} seconds")
        
        self.is_recording = False
        self.recording_start_time = None
        self.recording_filename = None
    
    def check_recording_commands(self):
        """Check for recording commands from main process"""
        if self.record_command_queue is None:
            return None
        
        try:
            while not self.record_command_queue.empty():
                command = self.record_command_queue.get_nowait()
                if command == "start":
                    return "start"
                elif command == "stop":
                    return "stop"
        except:
            pass
        return None
        
    def run(self):
        # Ultra-optimized FFMPEG settings for maximum FPS
        
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"[Camera {self.cam_id}] Cannot open RTSP")
            return
        
        # Aggressive optimization settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        
        # Get actual camera FPS
        actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera {self.cam_id}] Camera reports FPS: {actual_cam_fps}")
        print(f"[Camera {self.cam_id}] Process started (Target FPS: {self.target_fps})")
        
        last_display_time = time.time()
        frame_count = 0
        skip_frames = 0
        command = None
        frame_for_recording_start = None
        last_grab_time = time.time()
        
        while not self.stop_event.is_set():
            # Check for recording commands (only every 30 frames to reduce overhead)
            if frame_count % 30 == 0:
                command = self.check_recording_commands()
                if command == "stop" and self.is_recording:
                    self.stop_recording()
                elif command == "start" and not self.is_recording:
                    frame_for_recording_start = True
            
            # Measure grab time
            grab_start = time.time()
            
            # Read frame as fast as possible - NO delays, NO throttling
            ret = cap.grab()  # Grab without decoding (fastest)
            
            if not ret:
                skip_frames += 1
                if skip_frames > 100:
                    print(f"[Camera {self.cam_id}] Too many failed grabs, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    skip_frames = 0
                time.sleep(0.001)  # Tiny delay on failure
                continue
            
            skip_frames = 0
            
            # Decode frame (fast operation)
            ret, frame = cap.retrieve()
            
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            # Start recording if commanded
            if frame_for_recording_start and not self.is_recording:
                self.start_recording(frame.shape)
                frame_for_recording_start = False
            
            # Write frame to video if recording (minimal overhead)
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            
            # Send frame to display - completely non-blocking
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait((frame, self.is_recording))
                except:
                    pass
            
            # Performance logging every 3 seconds (reduced frequency)
            current_time = time.time()
            if current_time - last_display_time >= 3.0:
                elapsed = current_time - last_display_time
                actual_fps = frame_count / elapsed
                recording_status = "RECORDING" if self.is_recording else "NOT RECORDING"
                avg_grab_time = (current_time - last_grab_time) / max(frame_count, 1) * 1000
                print(f"[Camera {self.cam_id}] FPS: {actual_fps:.1f}/{self.target_fps} | "
                      f"Avg grab: {avg_grab_time:.1f}ms | {recording_status}")
                last_display_time = current_time
                last_grab_time = current_time
                frame_count = 0
        
        # Clean up
        if self.is_recording:
            self.stop_recording()
        
        cap.release()
        print(f"[Camera {self.cam_id}] Process stopped")