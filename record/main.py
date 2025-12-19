from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSpinBox)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from camera_manager import CameraManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Manager with Video Recording")
        self.manager = CameraManager()
        
        # Add cameras with optimized configuration
        # Camera 1: 30 FPS (increased from 20)
        self.manager.add_camera(
            cam_id=1,
            rtsp_url="rtsp://admin:infiniq2025@10.29.98.57/Streaming/Channels/101",
            target_fps=30
        )
        
        # Camera 2: 30 FPS (increased from 20)
        self.manager.add_camera(
            cam_id=2,
            rtsp_url="rtsp://admin:infiniq2025@10.29.98.58/Streaming/Channels/101",
            target_fps=30
        )
        
        # Camera 3: 30 FPS (increased from 20)
        self.manager.add_camera(
            cam_id=3,
            rtsp_url="rtsp://admin:infiniq2025@10.29.98.59/Streaming/Channels/101",
            target_fps=30
        )
        
        # --- UI ---
        main_layout = QVBoxLayout()
        
        # Camera 1 Controls
        cam1_layout = QHBoxLayout()
        cam1_label = QLabel("Camera 1:")
        cam1_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.btnStartCam1 = QPushButton("Start")
        self.btnStopCam1 = QPushButton("Stop")
        self.btnRecordCam1 = QPushButton("🔴 Record")
        self.btnRecordCam1.setEnabled(False)
        self.lblRecord1 = QLabel("⚪ Not Recording")
        
        # FPS control for Camera 1
        fps1_label = QLabel("FPS:")
        self.spinFps1 = QSpinBox()
        self.spinFps1.setRange(1, 60)
        self.spinFps1.setValue(30)
        self.spinFps1.setToolTip("Frames per second")
        
        cam1_layout.addWidget(cam1_label)
        cam1_layout.addWidget(self.btnStartCam1)
        cam1_layout.addWidget(self.btnStopCam1)
        cam1_layout.addWidget(fps1_label)
        cam1_layout.addWidget(self.spinFps1)
        cam1_layout.addWidget(self.btnRecordCam1)
        cam1_layout.addWidget(self.lblRecord1)
        cam1_layout.addStretch()
        
        # Camera 2 Controls
        cam2_layout = QHBoxLayout()
        cam2_label = QLabel("Camera 2:")
        cam2_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.btnStartCam2 = QPushButton("Start")
        self.btnStopCam2 = QPushButton("Stop")
        self.btnRecordCam2 = QPushButton("🔴 Record")
        self.btnRecordCam2.setEnabled(False)
        self.lblRecord2 = QLabel("⚪ Not Recording")
        
        # FPS control for Camera 2
        fps2_label = QLabel("FPS:")
        self.spinFps2 = QSpinBox()
        self.spinFps2.setRange(1, 60)
        self.spinFps2.setValue(30)
        self.spinFps2.setToolTip("Frames per second")
        
        cam2_layout.addWidget(cam2_label)
        cam2_layout.addWidget(self.btnStartCam2)
        cam2_layout.addWidget(self.btnStopCam2)
        cam2_layout.addWidget(fps2_label)
        cam2_layout.addWidget(self.spinFps2)
        cam2_layout.addWidget(self.btnRecordCam2)
        cam2_layout.addWidget(self.lblRecord2)
        cam2_layout.addStretch()
        
        # Camera 3 Controls
        cam3_layout = QHBoxLayout()
        cam3_label = QLabel("Camera 3:")
        cam3_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.btnStartCam3 = QPushButton("Start")
        self.btnStopCam3 = QPushButton("Stop")
        self.btnRecordCam3 = QPushButton("🔴 Record")
        self.btnRecordCam3.setEnabled(False)
        self.lblRecord3 = QLabel("⚪ Not Recording")
        
        # FPS control for Camera 3
        fps3_label = QLabel("FPS:")
        self.spinFps3 = QSpinBox()
        self.spinFps3.setRange(1, 60)
        self.spinFps3.setValue(30)
        self.spinFps3.setToolTip("Frames per second")
        
        cam3_layout.addWidget(cam3_label)
        cam3_layout.addWidget(self.btnStartCam3)
        cam3_layout.addWidget(self.btnStopCam3)
        cam3_layout.addWidget(fps3_label)
        cam3_layout.addWidget(self.spinFps3)
        cam3_layout.addWidget(self.btnRecordCam3)
        cam3_layout.addWidget(self.lblRecord3)
        cam3_layout.addStretch()
        
        # Info label
        info_label = QLabel("ℹ️ Video recordings will be saved to 'recordings/' folder in AVI format.")
        info_label.setStyleSheet("color: blue; font-style: italic;")
        
        # Add to main layout
        main_layout.addLayout(cam1_layout)
        main_layout.addLayout(cam2_layout)
        main_layout.addLayout(cam3_layout)
        main_layout.addWidget(info_label)
        main_layout.addStretch()
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Connect buttons
        self.btnStartCam1.clicked.connect(lambda: self.start_camera(1))
        self.btnStopCam1.clicked.connect(lambda: self.stop_camera(1))
        self.btnRecordCam1.clicked.connect(lambda: self.toggle_recording(1))
        
        self.btnStartCam2.clicked.connect(lambda: self.start_camera(2))
        self.btnStopCam2.clicked.connect(lambda: self.stop_camera(2))
        self.btnRecordCam2.clicked.connect(lambda: self.toggle_recording(2))
        
        self.btnStartCam3.clicked.connect(lambda: self.start_camera(3))
        self.btnStopCam3.clicked.connect(lambda: self.stop_camera(3))
        self.btnRecordCam3.clicked.connect(lambda: self.toggle_recording(3))
        
        # Timer for showing frames - optimized to 16ms (~60 FPS UI refresh)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16)  # ~60 FPS UI update
    
    def start_camera(self, cam_id):
        """Start camera with updated FPS setting"""
        # Get FPS value from spinbox
        if cam_id == 1:
            fps = self.spinFps1.value()
            btn = self.btnRecordCam1
        elif cam_id == 2:
            fps = self.spinFps2.value()
            btn = self.btnRecordCam2
        else:
            fps = self.spinFps3.value()
            btn = self.btnRecordCam3
        
        # Update camera configuration
        self.manager.camera_configs[cam_id]['target_fps'] = fps
        
        # If camera is already running, stop it first
        if self.manager.cameras[cam_id].is_alive():
            self.manager.stop_camera(cam_id)
            # Give it a moment to stop
            import time
            time.sleep(0.1)
        
        # Start with new FPS
        self.manager.start_camera(cam_id)
        
        # Enable record button
        btn.setEnabled(True)
    
    def stop_camera(self, cam_id):
        """Stop camera"""
        self.manager.stop_camera(cam_id)
        
        # Disable record button
        if cam_id == 1:
            self.btnRecordCam1.setEnabled(False)
        elif cam_id == 2:
            self.btnRecordCam2.setEnabled(False)
        else:
            self.btnRecordCam3.setEnabled(False)
    
    def toggle_recording(self, cam_id):
        """Toggle recording on/off for a camera"""
        if self.manager.is_recording(cam_id):
            # Stop recording
            self.manager.stop_recording(cam_id)
        else:
            # Start recording
            self.manager.start_recording(cam_id)
    
    def update_ui(self):
        """Update frame display and recording indicators"""
        self.manager.show_frames()
        
        # Update Camera 1 status
        if self.manager.is_recording(1):
            self.lblRecord1.setText("🔴 RECORDING")
            self.lblRecord1.setStyleSheet("color: red; font-weight: bold;")
            self.btnRecordCam1.setText("⏹️ Stop")
            self.btnRecordCam1.setStyleSheet("background-color: #ffcccc;")
        else:
            self.lblRecord1.setText("⚪ Not Recording")
            self.lblRecord1.setStyleSheet("color: gray;")
            self.btnRecordCam1.setText("🔴 Record")
            self.btnRecordCam1.setStyleSheet("")
        
        # Update Camera 2 status
        if self.manager.is_recording(2):
            self.lblRecord2.setText("🔴 RECORDING")
            self.lblRecord2.setStyleSheet("color: red; font-weight: bold;")
            self.btnRecordCam2.setText("⏹️ Stop")
            self.btnRecordCam2.setStyleSheet("background-color: #ffcccc;")
        else:
            self.lblRecord2.setText("⚪ Not Recording")
            self.lblRecord2.setStyleSheet("color: gray;")
            self.btnRecordCam2.setText("🔴 Record")
            self.btnRecordCam2.setStyleSheet("")
        
        # Update Camera 3 status
        if self.manager.is_recording(3):
            self.lblRecord3.setText("🔴 RECORDING")
            self.lblRecord3.setStyleSheet("color: red; font-weight: bold;")
            self.btnRecordCam3.setText("⏹️ Stop")
            self.btnRecordCam3.setStyleSheet("background-color: #ffcccc;")
        else:
            self.lblRecord3.setText("⚪ Not Recording")
            self.lblRecord3.setStyleSheet("color: gray;")
            self.btnRecordCam3.setText("🔴 Record")
            self.btnRecordCam3.setStyleSheet("")
    
    def closeEvent(self, event):
        self.manager.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()