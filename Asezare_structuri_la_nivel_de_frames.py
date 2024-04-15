# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 01:48:48 2024

@author: Diana
"""

import cv2
import mediapipe as mp
import numpy as np
import json

# Initializare MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Functie pentru calcularea unghiului între trei puncte
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Functie pentru calcularea unghiurilor din fiecare parte
def calculeaza_unghiuri(landmarks, side):
    if side == "RIGHT":
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        foot_index=[landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        #Definire coordonate pentru anumite unghiuri din partea stanga, pentru ca sunt folosite la calcularea unor unghiuri din partea dreapta
        shoulder_left = [0, 0]
        hip_left = [0, 0]

    else:
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        foot_index=[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        #Definire coordonate pentru anumite unghiuri din partea dreapta, pentru ca sunt folosite la calcularea unor unghiuri din partea stanga
        shoulder_right = [0, 0]
        hip_right = [0, 0]

    #Unghiuri de interes
    angle_elbow = calculate_angle(wrist, elbow, shoulder)
    angle_shoulder = calculate_angle(elbow, shoulder, hip)
    angle_shoulder_right = calculate_angle(elbow, shoulder_right, shoulder_left)
    angle_shoulder_left = calculate_angle(elbow, shoulder_left, shoulder_right)
    angle_armpit = calculate_angle(elbow, shoulder, hip)
    angle_shoulder_int_right = calculate_angle(shoulder_left, shoulder_right, hip)
    angle_shoulder_int_left = calculate_angle(shoulder_right, shoulder_left, hip)
    angle_hip = calculate_angle(shoulder, hip, knee)
    angle_knee = calculate_angle(hip, knee, ankle)
    angle_ankle = calculate_angle(knee, ankle, foot_index)
    angle_hip_int = calculate_angle(shoulder, hip, knee)
    angle_hip_sub_right = calculate_angle(hip_left, hip_right, knee)
    angle_hip_sub_left = calculate_angle(hip_right, hip_left, knee)

    return angle_elbow, angle_shoulder, angle_shoulder_right, angle_shoulder_left, angle_armpit, angle_shoulder_int_right, angle_shoulder_int_left, angle_hip, angle_knee, angle_ankle, angle_hip_int, angle_hip_sub_right, angle_hip_sub_left

# Array pentru stocarea datelor pentru fiecare frame
frames = []

# Procesare videoclip
cap = cv2.VideoCapture('Hip Adductor Stretch _ Patient Exercises.mp4')

# Creare un obiect de tip Pose cu anumite praguri de detectare și urmărire
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Buclă care rulează cât timp fluxul video este deschis
    while cap.isOpened():
        # Citire cadru din fluxul video
        ret, frame = cap.read()
        # Verificare dacă citirea cadrelor a fost reușită
        if not ret:
            # Întrerupere bucla în cazul în care citirea nu a fost reușită
            break

        # Convertire imagine din formatul BGR in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectare poziții cheie ale corpului
        results_pose = pose.process(frame_rgb)
        if results_pose.pose_landmarks:
            # Extragere coordonate cheie ale corpului
            landmarks = results_pose.pose_landmarks.landmark

            # Calculare unghiuri de interes, partea dreapta
            angle_elbow_right, angle_shoulder_right, angle_shoulder_right_int, angle_shoulder_left, angle_armpit_right, angle_shoulder_int_right, angle_shoulder_int_left, angle_hip_right, angle_knee_right, angle_ankle_right, angle_hip_int_right, angle_hip_sub_right, angle_hip_sub_left = calculeaza_unghiuri(landmarks, "RIGHT")

            # Calculare unghiuri de interes, partea stanga
            angle_elbow_left, angle_shoulder_left, angle_shoulder_left_int, angle_shoulder_left, angle_armpit_left, angle_shoulder_int_left, angle_shoulder_int_right, angle_hip_left, angle_knee_left, angle_ankle_left, angle_hip_int_left, angle_hip_sub_left, angle_hip_sub_right = calculeaza_unghiuri(landmarks, "LEFT")
            
            # Afisare unghiuri pe imagine
            for i, landmark in enumerate(landmarks):
                #partea stanga
                if i == mp_pose.PoseLandmark.LEFT_ELBOW.value:
                    cv2.putText(frame, f"{int(angle_elbow_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_armpit_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_shoulder_int_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_shoulder_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_HIP.value:
                    cv2.putText(frame, f"{int(angle_hip_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_HIP.value:
                    cv2.putText(frame, f"{int(angle_hip_int_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_KNEE.value:
                    cv2.putText(frame, f"{int(angle_knee_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.LEFT_ANKLE.value:
                    cv2.putText(frame, f"{int(angle_ankle_left)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #partea dreapta
                elif i == mp_pose.PoseLandmark.RIGHT_ELBOW.value:
                    cv2.putText(frame, f"{int(angle_elbow_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_armpit_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_shoulder_int_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    cv2.putText(frame, f"{int(angle_shoulder_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_HIP.value:
                    cv2.putText(frame, f"{int(angle_hip_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_HIP.value:
                    cv2.putText(frame, f"{int(angle_hip_int_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_KNEE.value:
                    cv2.putText(frame, f"{int(angle_knee_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif i == mp_pose.PoseLandmark.RIGHT_ANKLE.value:
                    cv2.putText(frame, f"{int(angle_ankle_right)}", (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
             # Desenare landmark-uri și conexiunile pe cadru
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Adăugare unghiuri la array-ul corespunzător pentru fiecare frame
            frame_data = {
                "angles": {
                    "0": angle_elbow_right,
                    "1": angle_elbow_left,
                    "2": angle_armpit_right,
                    "3": angle_armpit_left,
                    "4": angle_shoulder_int_right,
                    "5": angle_shoulder_int_left,
                    "6": angle_shoulder_right,
                    "7": angle_shoulder_left,
                    "8": angle_hip_right,
                    "9": angle_hip_left,
                    "10": angle_hip_sub_right,
                    "11": angle_hip_sub_left,
                    "12": angle_hip_int_right,
                    "13": angle_hip_int_left,
                    "14": angle_knee_right,
                    "15": angle_knee_left,
                    "16": angle_ankle_right,
                    "17": angle_ankle_left
                }
            }
            frames.append(frame_data)

        # Afisare cadru
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Eliberare resurse
cap.release()
# Închidere toate ferestrele deschise de OpenCV
cv2.destroyAllWindows()

# Numele fișierului JSON pentru stocare
json_filename = 'Hip_Adductor_Stretch.json'

# Deschidere fișier JSON în modul de scriere
with open(json_filename, 'w') as file:
    # Scriere date în fișierul JSON folosind metoda dump() din modulul json
    json.dump(frames, file)

print(f"Datele au fost salvate in fisierul JSON: {json_filename}")