import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp
from test_code.style_mediapipe import  draw_styled_landmarks,extract_keypoints

data_1 = np.load('./test/start_up.npy')
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_face_mesh = mp.solutions.face_mesh #Hai_them
face_mesh = mp_face_mesh.FaceMesh() #Hai_them
def draw_plot_mapping(data):
    left_hand_data = np.zeros((124,42))
    right_hand_data = np.zeros((124,42))
    lips_data = np.zeros((124,80))
    # Tạo figure và axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    plt.subplots_adjust(hspace=0.3)
    # Hàm khởi tạo, không cần thực hiện bất kỳ thay đổi gì ở đây
    def init():
        im1 = ax1.imshow(np.array(left_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Left Hand')
        ax1.set_yticks([])  
        
        im2 = ax2.imshow(np.array(right_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Right Hand')
        ax2.set_yticks([]) 
        
        im3 = ax3.imshow(np.array(lips_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_title('Lips')
        ax3.set_yticks([]) 


        return [im1, im2, im3]
    # Hàm cập nhật animation
    def update(frame):
        # Cập nhật chỉ dòng frame
        left_hand_data[frame,:] = data[frame,:42]
        right_hand_data[frame,:] = data[frame,42:84]
        lips_data[frame,:] = data[frame,84:]
        
        print(left_hand_data.shape)
        # lips_data = test[:frame ,42:122]q
        # right_hand_data = test[:frame,122:]
        # Cập nhật hình ảnh trên axes
        img1 = ax1.imshow(left_hand_data.T, cmap='viridis')
        img2 = ax2.imshow(right_hand_data.T, cmap='viridis')
        img3 = ax3.imshow(lips_data.T, cmap='viridis')

        return [img1, img2, img3]

    # Tạo animation
    animation = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True)

    plt.show()
n_frames = 0
def draw_plot_mapping_cam():
    global n_frames
    left_hand_data = np.zeros((124,42))
    right_hand_data = np.zeros((124,42))
    lips_data = np.zeros((124,80))
    cap = cv2.VideoCapture(0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    plt.subplots_adjust(hspace=0.3)
    # Hàm khởi tạo, không cần thực hiện bất kỳ thay đổi gì ở đây
    def init():
        im1 = ax1.imshow(np.array(left_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Left Hand')
        ax1.set_yticks([]) 
        
        im2 = ax2.imshow(np.array(right_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Right Hand')
        ax2.set_yticks([]) 
        
    
        im3 = ax3.imshow(np.array(lips_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_title('Lips')
        ax3.set_yticks([]) 
        
        return [im1, im2, im3]
    
    # Hàm cập nhật animation
    def update(frame):
        # Cập nhật chỉ dòng frame
        global n_frames
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        keypoints = extract_keypoints(results)
        left_hand_data[n_frames, :] = keypoints[:42]
        right_hand_data[n_frames, :] = keypoints[42:84]
        lips_data[n_frames, :] = keypoints[84:]
        # Cập nhật hình ảnh trên axes
        img1 = ax1.imshow(left_hand_data.T, cmap='viridis')
        img2 = ax2.imshow(right_hand_data.T, cmap='viridis')
        img3 = ax3.imshow(lips_data.T, cmap='viridis')
        n_frames +=1
        return [img1, img2, img3]

    # Tạo animation
    animation = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True)

    plt.show(block=False)
    # Chạy vòng lặp để hiển thị video từ camera
    while cap.isOpened():
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        frame = cv2.resize(frame, (frame.shape[1]*2+100, frame.shape[0]*2+100))
        results_face_mesh = face_mesh.process(frame_rgb)
        draw_styled_landmarks(frame, results, results_face_mesh)
        cv2.imshow('Video', frame)
        # Kiểm tra phím nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
draw_plot_mapping_cam()