import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 0))

LIPS_LANDMARK_IDXS = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results_holistic, results_face_mesh):
    # Draw face connections
    # lip_landmarks = np.array([results.face_landmarks.landmark[i] for i in LIPS_LANDMARK_IDXS])
    # mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(0, 0, 100), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=  (230, 216, 173), thickness=1, circle_radius=1)
    #                          ) 
    # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color= (0, 204, 255), thickness=1, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=  (0, 77, 0), thickness=1, circle_radius=2)
    #                          ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=  (0, 204, 255), thickness=1, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(220, 245, 245), thickness=1, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color= (128, 128, 128), thickness=1, circle_radius=4), 
                             mp_drawing.DrawingSpec(color= (208, 253, 255), thickness=1, circle_radius=2)
                             )
    #Draw lip
    if results_face_mesh.multi_face_landmarks:
          for face_landmarks in results_face_mesh.multi_face_landmarks:
               for id in LIPS_LANDMARK_IDXS:
                    landmark = face_landmarks.landmark[id]
                    point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, 
                                                                                        landmark.y, 
                                                                                        image.shape[1], 
                                                                                        image.shape[0])
                    cv2.circle(image, point, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)
 



def extract_keypoints(results):
        # x_face       = np.array([res.x for res in results.face_landmarks.landmark],dtype=np.float32) if results.face_landmarks else np.zeros(468)
        # y_face       = np.array([res.y for res in results.face_landmarks.landmark],dtype=np.float32) if results.face_landmarks else np.zeros(468)
        x_lips       = np.array([ results.face_landmarks.landmark[i].x for i in LIPS_LANDMARK_IDXS],dtype=np.float32) if results.face_landmarks else np.zeros(40)
        y_lips       = np.array([ results.face_landmarks.landmark[i].x for i in LIPS_LANDMARK_IDXS],dtype=np.float32) if results.face_landmarks else np.zeros(40)
        x_left_hand  = np.array([res.x for res in results.left_hand_landmarks.landmark],dtype=np.float32) if results.left_hand_landmarks else np.zeros(21)
        y_left_hand  = np.array([res.y for res in results.left_hand_landmarks.landmark],dtype=np.float32) if results.left_hand_landmarks else np.zeros(21)
        # x_pose       = np.array([res.x for res in results.pose_landmarks.landmark],dtype=np.float32) if results.pose_landmarks else np.zeros(33)
        # y_pose       = np.array([res.y for res in results.pose_landmarks.landmark],dtype=np.float32) if results.pose_landmarks else np.zeros(33)
        x_right_hand = np.array([res.x for res in results.right_hand_landmarks.landmark],dtype=np.float32) if results.right_hand_landmarks else np.zeros(21)
        y_right_hand = np.array([res.y for res in results.right_hand_landmarks.landmark],dtype=np.float32) if results.right_hand_landmarks else np.zeros(21)
        return np.concatenate((x_left_hand,y_left_hand,x_right_hand,y_right_hand,x_lips,y_lips),dtype=np.float32)