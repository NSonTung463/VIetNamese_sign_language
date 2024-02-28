import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QGridLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
import cv2
import mediapipe as mp
import qdarkstyle

from model.model_1 import *
import torch
import argparse
import importlib
from copy import copy
import sys
import numpy as np
import os 

from model.model_1 import SimpleLSTM
import pandas as pd
import torch



BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')
    
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_1")

parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

model = SimpleLSTM(cfg.input_size, cfg.hidden_size, cfg.output_size)
model.load_state_dict(torch.load('./output/weights/cfg_1/fold0/checkpoint_last_seed-1.pth')['model'])
actions = os.listdir('./dataset/landmarks')
label_map = {label: i for i, label in enumerate(actions)}
model.eval()

def predict(input, threshold=0.2):
    input = input.unsqueeze(0).float()
    
    # Perform inference on the sample
    with torch.no_grad():
        output = model(input)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    if probabilities[0, predicted_class] > threshold:
        predicted_class_string = [key for key, value in label_map.items() if value == predicted_class][0]
    else:
        predicted_class_string = "Uncertain"  # or any other label you prefer for uncertain predictions
    return predicted_class_string
class PreprocessLayer(nn.Module):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def forward(self, data0, resize=True):
        N_TARGET_FRAMES = 124
        N_COLUMNS = 1629
        # Fill NaN Values With 0
        data0 = torch.tensor(data0)
        data = torch.where(torch.isnan(data0), torch.tensor(0.0), data0)

        # Add another dimension

        # # Empty Hand Frame Filtering
        # hands = data[:, :, :, :84].abs()
        # mask = hands.sum(dim=2) != 0
        # data = data[mask].unsqueeze(0)

        # Padding with Zeros
        N_FRAMES = data.shape[0] 
        if N_FRAMES < N_TARGET_FRAMES:
            zeros_tensor = torch.zeros(N_TARGET_FRAMES - N_FRAMES, N_COLUMNS, dtype=torch.float32)
            data = torch.cat((data, zeros_tensor), dim=0)
        data = data[None]
        tensor_downsampled = F.interpolate(data.unsqueeze(0), size=(N_TARGET_FRAMES, N_COLUMNS), mode='bilinear', align_corners=False)[0]
        data = tensor_downsampled.squeeze(axis=0)
        return data

preprocessLayer = PreprocessLayer()

class ConvertFileToParquet():
    def __init__(self, data, folder_path, save_path=None):
        self.folder_path = folder_path
        self.save_path = save_path
        self.data = data

    @staticmethod
    def column_name():
        coordinates = ['x', 'y', 'z']
        col_name = ['_face_', '_left_hand_', '_pose_', '_right_hand_']
        column_name_list = []
        for coordinate in coordinates:
            for name in col_name:
                if name == '_face_':
                    for i in range(0, 468):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_left_hand_':
                    for i in range(0, 21):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_pose_':
                    for i in range(0, 33):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_right_hand_':
                    for y in range(0, 21):
                        column_name_list.append(coordinate+name+str(y))
        return column_name_list

    def convert_to_dataframe(self):
        data_list = []
        frame = []
        for i in range(self.data.shape[0]):
            frame.append(i)
            data_list.append(self.data[i])
        data_df = pd.DataFrame(data=data_list, columns=ConvertFileToParquet.column_name())
        data_df.insert(0, 'frame', frame)
        return data_df

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
        
        self.stop_camera_button = QPushButton("Stop Webcam", self)
        self.stop_camera_button.clicked.connect(self.stop_webcam)
        self.stop_camera_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.stop_camera_button)

        self.export_csv_button = QPushButton("Export CSV", self)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.export_csv_button)
        
        self.layout = QGridLayout()
        self.layout.addLayout(button_layout1, 0, 0)

        # Add a line between the buttons and the video label
        line = QFrame(self)
        line.setFrameShape(QFrame.VLine)
        self.layout.addWidget(line, 0, 1)

        self.layout.addWidget(self.video_label, 0, 2)

        widget = QWidget(self)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.camera = None
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_face = mp.solutions.face_mesh.FaceMesh()
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_holistic = mp.solutions.holistic.Holistic()


        self.hands_results = None
        self.face_results = None
        self.pose_results = None
        
        self.landmark_dataframe = pd.DataFrame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.landmark_dataframe = pd.DataFrame(columns=["sequence_id", "frame"])

        self.res = []
        self.is_video_finished = False
        self.threshold = 40
        self.num_frame_space = 100
        self.list_frame = []
        self.predicted = None

    def run_webcam(self):
        if self.camera is not None:
            self.camera.release()  # Release the camera
        self.camera = cv2.VideoCapture(0)  # Open the default camera
        self.is_video_finished = False
        self.res = []
        self.timer.start(30)  # Update frame every 30 milliseconds
        
    def import_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video File")
        if video_path:
            if self.camera is not None:
                self.camera.release()
            self.camera = cv2.VideoCapture(video_path)
            self.is_video_finished = False
            self.res = []
            self.timer.start(30)  # Update frame every 30 milliseconds
            

    def stop_webcam(self):
        if self.camera is not None:
            self.camera.release()  # Release the camera
            self.camera = None  # Set the camera to None
            self.video_label.clear()  # Clear the video label
            self.ret = False

    def update_frame(self):
        if self.is_video_finished:
            self.stop_webcam()
            self.ret = False
            return self.landmark_dataframe
        else:   
            keypoints = None
            if self.camera is not None:
                self.ret, frame = self.camera.read()
            else:
                self.ret = False
                
        if self.ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_holistic.process(frame)
            keypoints = self.extract_keypoints(results)
            self.res.append(keypoints)
            landmarks_series = pd.Series(keypoints)
            self.landmark_dataframe = self.landmark_dataframe._append(landmarks_series, ignore_index=True)
            
            nres = len(self.res)
            sequence_arr = np.array(self.res)
            n_frame = sequence_arr.shape[0]
            self.landmark_dataframe = self.landmark_dataframe._append({"sequence_id": "sửa sau","frame": n_frame}, ignore_index=True)
            ranges = [(468, 489), (522, 543), (1011, 1032), (1065, 1086)]
            slices_arr = np.concatenate([sequence_arr[n_frame-self.num_frame_space:n_frame, start:end] for start, end in ranges], axis=1)
            if nres > self.num_frame_space and np.all(np.isnan(slices_arr)):
                self.res = []                                                                          
                print('reset')

            if nres==self.threshold:
                self.threshold += 1
                subarray=sequence_arr[nres-124:nres,:]
                subarray = preprocessLayer(subarray)
                subarray = subarray[:,:1086]
                self.predicted = predict(subarray)
                print(self.predicted)
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
            self.predicted_label = QLabel(str(self.predicted))
            self.layout.addWidget(self.predicted_label, 1, 2)  # Add the label to position (1,2)

            self.setLayout(self.layout)
            
        elif self.ret == False:
            self.res = np.array(self.res)
            self.res_1 = preprocessLayer(self.res)
            self.res_1 = self.res_1[:,:1086]
            
            self.predicted = predict(self.res_1)
            print(f'final predict: {self.predicted}')
            self.is_video_finished = True
            
            self.predicted_label = QLabel(str(self.predicted))
            self.layout.addWidget(self.predicted_label, 1, 2)  # Add the label to position (1,2)

            self.setLayout(self.layout)
            
            convert = ConvertFileToParquet(self.res, None ,None)
            self.landmark_dataframe = convert.convert_to_dataframe()
            
            return self.landmark_dataframe

    def extract_keypoints(self, results):
        if results.face_landmarks:
            face_x = np.array([res.x for res in results.face_landmarks.landmark]).flatten()   
            face_y = np.array([res.y for res in results.face_landmarks.landmark]).flatten()
            face_z = np.array([res.z for res in results.face_landmarks.landmark]).flatten()
        else: 
            face_x = np.full(468, np.nan)
            face_y = np.full(468, np.nan)
            face_z = np.full(468, np.nan)

        if results.left_hand_landmarks:
            lh_x = np.array([res.x for res in results.left_hand_landmarks.landmark]).flatten()   
            lh_y = np.array([res.y for res in results.left_hand_landmarks.landmark]).flatten()
            lh_z = np.array([res.z for res in results.left_hand_landmarks.landmark]).flatten()
        else: 
            lh_x = np.full(21, np.nan)
            lh_y = np.full(21, np.nan)
            lh_z = np.full(21, np.nan)

        if results.pose_landmarks:
            pose_x = np.array([res.x for res in results.pose_landmarks.landmark]).flatten()   
            pose_y = np.array([res.y for res in results.pose_landmarks.landmark]).flatten()
            pose_z = np.array([res.z for res in results.pose_landmarks.landmark]).flatten()
        else: 
            pose_x = np.full(33, np.nan)
            pose_y = np.full(33, np.nan)
            pose_z = np.full(33, np.nan)
            
        if results.right_hand_landmarks:
            rh_x = np.array([res.x for res in results.right_hand_landmarks.landmark]).flatten()   
            rh_y = np.array([res.y for res in results.right_hand_landmarks.landmark]).flatten()
            rh_z = np.array([res.z for res in results.right_hand_landmarks.landmark]).flatten()
        else: 
            rh_x = np.full(21, np.nan)
            rh_y = np.full(21, np.nan)
            rh_z = np.full(21, np.nan)     
            
        x_cor = np.concatenate([face_x, lh_x, pose_x, rh_x])
        y_cor = np.concatenate([face_y, lh_y, pose_y, rh_y])
        z_cor = np.concatenate([face_z, lh_z, pose_z, rh_z])
        return   np.concatenate([x_cor, y_cor, z_cor])

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

