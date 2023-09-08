import pickle 
from sklearn.metrics import accuracy_score # Accuracy metrics 
import csv
import os
import numpy as np
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import time
import pandas as pd
from sklearn.model_selection import train_test_split

fram_counter_happy=0
fram_counter_fear=0
fram_counter_surprize=0
current_state = ""
num_times={
    "Happy": 0,
    "Fear": 0,
    "Surprize": 0
}

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

#loading models
with open('model/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture("main.mkv")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=False) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret==True):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            image_height, image_width, _ = image.shape
            results = holistic.process(image)


            
            
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 2. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 3. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            try:
                pose= results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                X = pd.DataFrame([pose_row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(type(body_language_class))

                #counting number of frames for Happy, Fear and Surprize states
                if((body_language_class=="Happy") and round(body_language_prob[np.argmax(body_language_prob)])>=0.7):
                    fram_counter_happy+=1
                elif((body_language_class=="Fear") and round(body_language_prob[np.argmax(body_language_prob)])>=0.6):
                    fram_counter_fear+=1
                elif((body_language_class=="Surprize") and round(body_language_prob[np.argmax(body_language_prob)])>=0.6):
                    fram_counter_surprize+=1

                #counting number of times this emmotion appeared
                if(current_state!=body_language_class and round(body_language_prob[np.argmax(body_language_prob)])>=0.6):
                    current_state=body_language_class
                    num_times[body_language_class]+=1
                print(body_language_class)
                    
                    # Get status box
                cv2.rectangle(image, (0,0), (140, 50), (245, 117, 16), -1)
                    
                    # Display Class
                cv2.putText(image, 'CLASS'
                                , (70,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                                , (70,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Display Probability
                cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                out.write(image) 
            except Exception as error:
                #displaying if error happens in try section
                print(error)
                pass 
            
                
            
    
            cv2.imshow("windows", image)  
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
#printing out the results
print("Happy time = " + str(fram_counter_happy/30)+"sec")
print("Fear time = " + str(fram_counter_fear/30)+"sec")
print("Surprize time = " + str(fram_counter_surprize/30)+"sec")
print("Happy number of times = " + str(num_times["Happy"]) )
print("Fear number of times = " + str(num_times["Fear"]) )
print("Surprize number of times = " + str(num_times["Surprize"]) )
cap.release()
out.release()
cv2.destroyAllWindows()