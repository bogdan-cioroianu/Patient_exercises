import cv2
import mediapipe as mp
import numpy as np

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

# Verificare dacă unghiurile calculate sunt apropiate de unghiurile țintă cu o toleranță între unghiul calculat și unghiul țintă egala cu 10.
def check_angles(angles, target_angles, tolerance=10):
   
    # Inițializare listă pentru a stoca rezultatele
    result = []
    # Parcurgere fiecare pereche de unghiuri calculate și unghiuri țintă
    for angle, target_angle in zip(angles, target_angles):
        # Verificare dacă diferența absolută dintre unghiul calculat și unghiul țintă este mai mică sau egală cu toleranța
        if abs(angle - target_angle) <= tolerance:
            # Adaugare True în lista de rezultate dacă unghiul este în interiorul toleranței
            result.append(True)
        else:
            # Adaugare False în lista de rezultate dacă unghiul nu este în interiorul toleranței
            result.append(False)
    # Returnare lista finală de valori booleene
    return result

# Frame data target pentru fiecare stage

####### Bogdan #####
# Ar vfi bine sa pui template-urile in fisiere externe pe care sa le poti pune ca input. Astfel poti aplica codul pe mai multe fisiere fara sa il modifici.
################
frame_data_target_template = [
    {"angles": {"2": [25], "8": [160], "9": [160], "14": [175], "15": [175]}},
    {"angles": {"2": [25], "8": [155], "9": [155], "14": [150], "15": [175]}},
    {"angles": {"2": [25], "8": [160], "9": [160], "14": [175], "15": [175]}},
    {"angles": {"2": [25], "8": [145], "9": [175], "14": [175], "15": [175]}},
    {"angles": {"2": [25], "8": [160], "9": [160], "14": [175], "15": [175]}}
]
# Procesare videoclip
cap = cv2.VideoCapture('Hip Adductor Stretch _ Patient Exercises.mp4')

# Creare obiect de tip Pose cu praguri de detectare și urmărire specificate
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Inițializarea stadiului curent cu valoarea 0
current_stage = 0
# Inițializare listă de etape completate cu valoarea False, având lungimea egală cu șablonul de date țintă al cadrelor
stage_completed = [False] * len(frame_data_target_template)

# Executare în mod continuu până când fluxul video este deschis
while cap.isOpened():
    # Citirea unui cadru din fluxul video
         ret, frame = cap.read()
    # Verificare dacă citirea cadrelor a reușit
         if not ret:
        # Ieșire din buclă dacă citirea nu a reușit
            break
     
         # Convertire imagine din formatul BGR in RGB
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

         # Procesare imagine pentru a detecta pozițiile și mișcările corpului
         results_pose = pose.process(frame_rgb)

         # Verificare dacă sunt detectate landmark-urile poziției în rezultatele procesării
         if results_pose.pose_landmarks:
            # Extragere landmark-uri ale poziției detectate
            landmarks = results_pose.pose_landmarks.landmark

            # Calculare unghiuri partea dreapta
             ####### Bogdan #####
            # Tiput asta de stocare a unghiurilor nu este foarte eficient
            # Iti recomand sa faci un array si pe pozitia X din arraty stochezi valoarea unghiului X, conform modelului pe care l-am discutat. Poti modificat functia calculeaza_unghiuri astfel incat sa iti returneze acest array. In functie trebuie sa faci o assignare statitica pe fiecare pozitie.
            ################
            angle_elbow_right, angle_shoulder_right, angle_shoulder_right_int, angle_shoulder_left, angle_armpit_right, angle_shoulder_int_right, angle_shoulder_int_left, angle_hip_right, angle_knee_right, angle_ankle_right, angle_hip_int_right, angle_hip_sub_right, angle_hip_sub_left = calculeaza_unghiuri(landmarks, "RIGHT")

            # Calculare unghiuri partea stângă
            angle_elbow_left, angle_shoulder_left, angle_shoulder_left_int, angle_shoulder_left, angle_armpit_left, angle_shoulder_int_left, angle_shoulder_int_right, angle_hip_left, angle_knee_left, angle_ankle_left, angle_hip_int_left, angle_hip_sub_left, angle_hip_sub_right = calculeaza_unghiuri(landmarks, "LEFT")
            
            # Afisare unghiuri pe imagine
            for i, landmark in enumerate(landmarks):

            ####### Bogdan #####
            # 100 de elseif -uri, nu e deloc elegant :). Te rog sa te gandesti cum ai putea sa scapi de ele. 
            # Hint: poti sa folosesti array-ul de unghiuri de care am vb mai sus si un o alta mapare statica intre pozitia unghiului in array si valoare din mp_pose.PoseLandmark.nnnnn 
            ################
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
            
            # Verificare dacă stage-ul curent este mai mic decât lungimea șablonului de date țintă al cadrelor
            if current_stage < len(frame_data_target_template):

            ####### Bogdan #####
            # Aici e o mapare statica care iti blocheaza codul doar pe acest template. Trebui sa te gandesti cum ai putea face un code generic care sa iti poate extrage unghiurile pe care trebuie sa le compari din template. Tu ai index-ul fiecarui unghi care trebui comparat asa ca poti parcurge   
            # template-ul si compara fiecare unghit targe cu valoarea lui din acel moment. Asta in loc sa hardcodezi care unghiuri sunt in scop.
            ################
                
              # Extragere unghiuri din partea dreapta/stanga și unghiurile țintă corespunzătoare din template de date țintă al cadrelor pentru stage-ul curent
              right_angles = [angle_armpit_right, angle_hip_right, angle_knee_right]
              right_target_angles = frame_data_target_template[current_stage]["angles"]["2"], frame_data_target_template[current_stage]["angles"]["8"], frame_data_target_template[current_stage]["angles"]["14"]
              left_angles = [angle_hip_left, angle_knee_left]
              left_target_angles = frame_data_target_template[current_stage]["angles"]["9"], frame_data_target_template[current_stage]["angles"]["15"]

              # Verificare dacă unghiurile calculate se apropie de unghiurile țintă pentru ambele părți
              result = check_angles(left_angles + right_angles, left_target_angles + right_target_angles)
            else:
              #Inițializare o listă goală dacă stadiul curent depășește lungimea template-ului de date țintă al cadrelor
              result = []

            # Verifică dacă toate elementele din lista de rezultate sunt True și dacă stadiul curent este mai mic decât lungimea listei stage_completed
            if all(result) and current_stage < len(stage_completed):
              stage_completed[current_stage] = True  # Marcam stage-ul curent ca fiind completat în lista stage_completed
              current_stage += 1  # Trecem la următorul stage
              
         #Parcurgere lista stage_completed
         for i, stage in enumerate(stage_completed):
          if stage:  # Verificam daca stage-ul a fost completat
            cv2.putText(frame, f"Stage {i+1} checked", (50, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Afișare într-o fereastră cu titlul 'Frame'
            cv2.imshow('Frame', frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Eliberare resurse
cap.release()
# Închidere toate ferestrele deschise de OpenCV
cv2.destroyAllWindows()
