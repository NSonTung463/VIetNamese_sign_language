import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QGridLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
import cv2
import mediapipe as mp
import qdarkstyle
import numpy as np


class TrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand and Face Landmark Tracking App")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        button_layout1 = QVBoxLayout()
        self.exit_button = QPushButton("Exit Application", self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.exit_button)

        self.run_webcam_button = QPushButton("Run Webcam", self)
        self.run_webcam_button.clicked.connect(self.run_webcam)
        self.run_webcam_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.run_webcam_button)

        self.import_video_button = QPushButton("Import Video", self)
        self.import_video_button.clicked.connect(self.import_video)
        self.import_video_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.import_video_button)

        self.export_csv_button = QPushButton("Export CSV", self)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.export_csv_button)

        layout = QGridLayout()
        layout.addLayout(button_layout1, 0, 0)

        # Add a line between the buttons and the video label
        line = QFrame(self)
        line.setFrameShape(QFrame.VLine)
        layout.addWidget(line, 0, 1)

        layout.addWidget(self.video_label, 0, 2)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.camera = None
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_face = mp.solutions.face_mesh.FaceMesh()
        self.mp_pose = mp.solutions.pose.Pose()
        self.hands_results = None
        self.face_results = None
        self.pose_results = None
        
        self.landmark_dataframe = pd.DataFrame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.landmark_dataframe = pd.DataFrame(columns=["sequence_id", "frame"])

    def run_webcam(self):
        self.camera = cv2.VideoCapture(0)  # Open the default camera
        self.timer.start(30)  # Update frame every 30 milliseconds

    def import_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video File")
        if video_path:
            self.camera = cv2.VideoCapture(video_path)
            self.timer.start(30)  # Update frame every 30 milliseconds

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            self.hands_results = self.mp_hands.process(frame)
            self.face_results = self.mp_face.process(frame)
            self.pose_results = self.mp_pose.process(frame)

            # Initialize the landmark_data dictionary with NaN values
            landmark_data = {"sequence_id": "default", "frame": self.get_frame_number()}
            for hand_index in range(2):  # assuming two hands
                for idx in range(21):  # assuming 21 hand landmarks
                    for axis in ['x', 'y', 'z']:
                        hand_key = f"{['right', 'left'][hand_index]}_hand_{idx + 1}_{axis}"
                        landmark_data[hand_key] = float('nan')
                    
            for pose_index in range(33):  # assuming 33 pose landmarks
                for axis in ['x', 'y', 'z']:
                    pose_key = f"pose_{pose_index + 1}_{axis}"
                    landmark_data[pose_key] = float('nan')
                    
            for face_index in range(468):  # assuming 468 face landmarks
                for axis in ['x', 'y', 'z']:
                    face_key = f"face_{face_index + 1}_{axis}"
                    landmark_data[face_key] = float('nan')  

            # Fill in hand landmarks if available
            if self.hands_results.multi_hand_landmarks and self.face_results.multi_face_landmarks:
                for hand_index, hand_landmarks in enumerate(self.hands_results.multi_hand_landmarks):
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        z = landmark.z
                        hand_key = f"{['right', 'left'][hand_index]}_hand_{idx + 1}"
                        landmark_data[hand_key + '_x'] = x
                        landmark_data[hand_key + '_y'] = y
                        landmark_data[hand_key + '_z'] = z
                        cv2.circle(frame, (x, y), 1, (99, 255, 78), -1)
                        cv2.putText(frame, f"{hand_key}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 1)

            # Fill in face landmarks if available
            if self.face_results.multi_face_landmarks:
                for face_landmarks in self.face_results.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        z = landmark.z
                        face_key = f"face_{idx + 1}"
                        landmark_data[face_key + '_x'] = x
                        landmark_data[face_key + '_y'] = y
                        landmark_data[face_key + '_z'] = z
                        cv2.circle(frame, (x, y), 1, (125, 148, 195), -1)
                        # cv2.putText(frame, f"{face_key}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 255), 1)

            if self.pose_results.pose_landmarks:
                for idx, landmark in enumerate(self.pose_results.pose_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    pose_key = f"pose_{idx + 1}"
                    landmark_data[pose_key + '_x'] = x
                    landmark_data[pose_key + '_y'] = y
                    landmark_data[pose_key + '_z'] = z
                    cv2.circle(frame, (x, y), 1, (197, 148, 195), -1)
                    # cv2.putText(frame, f"{pose_key}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (0, 0, 255), 1)
            
            # Append the landmark_data to the DataFrame
            self.landmark_dataframe = self.landmark_dataframe._append(landmark_data, ignore_index=True)

            # Display the frame
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.release()
        event.accept()

    def get_frame_number(self):
        if self.camera is None:
            return 0
        else:
            return self.camera.get(cv2.CAP_PROP_POS_FRAMES)

    def export_csv(self):
        file_dialog = QFileDialog()
        csv_path, _ = file_dialog.getSaveFileName(self, "Export CSV File")
        if csv_path:
            self.landmark_dataframe.to_csv(csv_path, index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # Apply the dark theme
    window = TrackingApp()
    window.show()
    sys.exit(app.exec())
