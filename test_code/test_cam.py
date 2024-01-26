import cv2
import mediapipe as mp
import numpy as np
from test_code.style_mediapipe import  draw_styled_landmarks,extract_keypoints
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_face_mesh = mp.solutions.face_mesh #Hai_them
face_mesh = mp_face_mesh.FaceMesh() #Hai_them
# Khởi tạo Holistic
cap = cv2.VideoCapture(0)
sequence =  []
n_frame = 0
prev_time = 0
# Khởi tạo biến để lưu trữ landmarks
left_hand_data = np.zeros((200, 42))
right_hand_data = np.zeros((200, 42))
lips_data = np.zeros((200, 80))
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        # Chuyển đổi frame sang định dạng màu BGR
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_holistic = holistic.process(frame_rgb)
        results_face_mesh = face_mesh.process(frame_rgb)
        draw_styled_landmarks(frame, results_holistic, results_face_mesh)


        # Dùng Holistic để xử lý frame
        results = holistic.process(frame_rgb)
        keypoints = extract_keypoints(results)

        # sequence.append(keypoints)
        # sequence_arr = np.array(sequence)
        # print(sequence_arr.shape)
        if n_frame >= 200:
            sequence=[]
            n_frame = 0
        # Vẽ các landmark trên cơ thể và khuôn mặt
        cv2.imshow('Holistic', frame)
        n_frame +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
