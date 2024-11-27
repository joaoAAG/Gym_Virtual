import cv2
import os
import time
import mediapipe as mp
import customtkinter as ctk
from tkinter import messagebox, Label
from PIL import Image, ImageTk
import screeninfo
from ultralytics import YOLO

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variables to count exercises
hand_elevation_count = 0
hand_above_head = False
jumping_jack_count = 0
legs_apart = False
push_up_count = 0
push_up_down = False
in_position = False  # Variable to track if in position for jumping jacks
tree_pose_count = 0  # Count for Tree Pose

selected_exercise = None
highlighted_button_index = 0
buttons = []
countdown_label = None

current_set = 1
total_sets = 1
repetitions = 1

pose_detection_times = {"Tree Pose": 0}  # Timer for holding yoga poses

# Load pre-trained models for gender and age prediction
model_dir = "models"
face_proto = os.path.join(model_dir, "deploy.prototxt")
face_model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
age_proto = os.path.join(model_dir, "age_deploy.prototxt")
age_model = os.path.join(model_dir, "age_net.caffemodel")
gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
gender_model = os.path.join(model_dir, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-22)', '(23-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Load YOLO model
yolo_model = YOLO(r'runs/face-recognition6/weights/best.pt')

gender_age_label = None
detected_gender_age = ""

# Variables for face recognition timing
last_face_detected_time = 0
face_recognition_active = True
face_detected = False

# Add debounce variables
last_action_time = time.time()
debounce_delay = 0.5  # 0.5 seconds delay between actions

# Function to detect gender and age
def detect_gender_age(face):
    global detected_gender_age
    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender
    gender_net.setInput(face_blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Predict Age
    age_net.setInput(face_blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    detected_gender_age = f"{gender}, {age}"
    return detected_gender_age

def detect_hand_gestures(hand_landmarks):
    thumb_is_open = hand_landmarks[4].y < hand_landmarks[3].y < hand_landmarks[2].y < hand_landmarks[1].y
    index_is_open = hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y
    middle_is_open = hand_landmarks[12].y < hand_landmarks[10].y < hand_landmarks[9].y
    ring_is_open = hand_landmarks[16].y < hand_landmarks[14].y < hand_landmarks[13].y
    pinky_is_open = hand_landmarks[20].y < hand_landmarks[18].y < hand_landmarks[17].y

    if not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "fist"
    elif index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "open"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "thumb_up"
    return "unknown"

def detect_hand_elevations(image, landmarks, difficulty):
    global hand_elevation_count, hand_above_head
    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    head_top = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y - 0.1]
    chest_level = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    ground_level = 0.9  # Near the bottom of the screen

    # Set thresholds based on difficulty
    if difficulty == 1:
        threshold = head_top[1]  # Cross the eyeline
    elif difficulty == 2:
        threshold = chest_level  # Reach the chest level
    else:
        threshold = ground_level  # Touch the ground

    if left_hand[1] < threshold or right_hand[1] < threshold:
        if not hand_above_head:
            hand_elevation_count += 1
            hand_above_head = True
    else:
        hand_above_head = False

    cv2.putText(image, f'Hand Elevations: {hand_elevation_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

def detect_jumping_jacks(image, landmarks):
    global jumping_jack_count, in_position

    # Extract landmarks
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    # Calculate hip width
    hips_width = abs(left_hip[0] - right_hip[0])

    # Adjust thresholds based on difficulty or dynamically
    hips_width_threshold = hips_width * 1.5  # Adjusted threshold for detecting wide stance
    head_top_threshold = 0.2  # Adjusted threshold for hands above head

    # Check if feet are apart (greater than hip-width)
    feet_apart = abs(left_ankle[0] - right_ankle[0]) > hips_width_threshold

    # Check if hands are above head
    hands_above_head = left_wrist[1] < head_top_threshold and right_wrist[1] < head_top_threshold

    # Check if the user is in jumping jack position
    if feet_apart and hands_above_head:
        if not in_position:
            jumping_jack_count += 1
            in_position = True
    else:
        in_position = False

    # Display jumping jack count on the image
    cv2.putText(image, f'Jumping Jacks: {jumping_jack_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return jumping_jack_count

def detect_push_ups(image, landmarks, difficulty):
    global push_up_count, push_up_down
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # Set thresholds based on difficulty
    if difficulty == 1:
        threshold = 0.2  # Less depth required
    elif difficulty == 2:
        threshold = 0.4  # Medium depth
    else:
        threshold = 0.5  # Full depth

    if shoulder_y > threshold:
        if not push_up_down:
            push_up_down = True
    else:
        if push_up_down:
            push_up_count += 1
            push_up_down = False

    cv2.putText(image, f'Push-Ups: {push_up_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

def detect_yoga_poses(image, landmarks, difficulty):
    global tree_pose_count

    # Pegar os pontos-chave dos marcos
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    detected_pose = "Unknown"

    # Verificar Tree Pose (uma perna levantada, pé na coxa interna, braços acima da cabeça)
    if (abs(left_ankle.y - left_knee.y) < 0.1 or abs(right_ankle.y - right_knee.y) < 0.1) and \
            abs(left_wrist.y - left_hip.y) < 0.5 and abs(right_wrist.y - right_hip.y) < 0.5:
        detected_pose = "Tree Pose"

    current_time = time.time()
    hold_time = {1: 5, 2: 10, 3: 15}[difficulty]

    for pose in pose_detection_times:
        if detected_pose == pose:
            if pose_detection_times[pose] == 0:
                pose_detection_times[pose] = current_time
            elapsed_time = current_time - pose_detection_times[pose]
            remaining_time = hold_time - elapsed_time
            if remaining_time > 0:
                cv2.putText(image, f'{int(remaining_time)} seconds left', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2, cv2.LINE_AA)
            if elapsed_time > hold_time:
                if pose == "Tree Pose":
                    tree_pose_count += 1

                pose_detection_times[pose] = 0
                cv2.putText(image, f'{pose} done!', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            pose_detection_times[pose] = 0

    # Display detected pose and current counts
    cv2.putText(image, f'Yoga Pose: {detected_pose}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(image, f'Tree Pose Count: {tree_pose_count}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2, cv2.LINE_AA)

    return detected_pose

def start_exercise():
    global cap, mp_pose, pose, mp_drawing, mp_hands, hands, selected_exercise
    global hand_elevation_count, jumping_jack_count, push_up_count, tree_pose_count, current_set

    if selected_exercise is None:
        messagebox.showwarning("Exercise Selection", "Please select an exercise!")
        return

    cv2.namedWindow("Exercise", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Exercise", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while current_set <= total_sets:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            difficulty = int(difficulty_slider.get())  # Get the difficulty level

            if selected_exercise == "Hand Elevations":
                detect_hand_elevations(image, landmarks, difficulty)
                if hand_elevation_count >= repetitions:
                    hand_elevation_count = 0
                    current_set += 1
            elif selected_exercise == "Jumping Jacks":
                detect_jumping_jacks(image, landmarks)
                if jumping_jack_count >= repetitions:
                    jumping_jack_count = 0
                    current_set += 1
            elif selected_exercise == "Push-Ups":
                detect_push_ups(image, landmarks, difficulty)
                if push_up_count >= repetitions:
                    push_up_count = 0
                    current_set += 1
            elif selected_exercise == "Yoga":
                detect_yoga_poses(image, landmarks, difficulty)
                if tree_pose_count >= repetitions:
                    tree_pose_count = 0
                    current_set += 1

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        except Exception as e:
            print(f"An error occurred: {e}")

        cv2.putText(image, f'Set: {current_set}/{total_sets}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        cv2.imshow("Exercise", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def highlight_button(index):
    for i, btn in enumerate(buttons):
        if i == index:
            btn.configure(fg_color="green")
        else:
            btn.configure(fg_color="gray")

def set_exercise(exercise):
    global selected_exercise, current_set, hand_elevation_count, jumping_jack_count, push_up_count, tree_pose_count
    selected_exercise = exercise
    current_set = 1
    hand_elevation_count = 0
    jumping_jack_count = 0
    push_up_count = 0
    tree_pose_count = 0
    show_countdown(6)  # Start the countdown with 6 seconds

def show_countdown(seconds):
    def countdown():
        nonlocal seconds
        if seconds > 0:
            countdown_label.configure(text=f"{seconds} seconds remaining")
            seconds -= 1
            root.after(1000, countdown)
        else:
            countdown_label.pack_forget()  # Remove countdown label
            start_exercise()  # Start the exercise

    countdown_label.pack(pady=20)
    countdown()  # Start the countdown

def set_sliders_based_on_age(age):
    global sets_slider, reps_slider, difficulty_slider
    if age in ['(0-2)', '(4-6)', '(8-12)']:
        sets_slider.set(1)
        reps_slider.set(10)
        difficulty_slider.set(1)
    elif age in ['(15-22)', '(23-32)']:
        sets_slider.set(3)
        reps_slider.set(12)
        difficulty_slider.set(3)
    elif age in ['(38-43)', '(48-53)', '(60-100)']:
        sets_slider.set(2)
        reps_slider.set(6)
        difficulty_slider.set(2)

def reload_face_recognition():
    global face_recognition_active, face_detected, last_face_detected_time
    face_recognition_active = True
    face_detected = False
    last_face_detected_time = 0
    gender_age_label.configure(text="")

def create_ui():
    global buttons, highlighted_button_index, root, countdown_label, total_sets, repetitions, sets_slider, reps_slider, difficulty_slider, camera_label, gender_age_label, detected_once, last_action_time
    root = ctk.CTk()
    root.title("Exercise Selection")

    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    # Set the position and size of the Tkinter window
    root.attributes('-fullscreen', True)
    root.geometry(f"{screen_width}x{screen_height}")

    # Load icons after root is created
    dumbbell_icon = ctk.CTkImage(Image.open("dumbbell_icon.png").resize((50, 50), Image.LANCZOS))
    sports_icon = ctk.CTkImage(Image.open("sports_icon.png").resize((50, 50), Image.LANCZOS))
    pushup_icon = ctk.CTkImage(Image.open("pushup_icon.png").resize((50, 50), Image.LANCZOS))
    reload_icon = ctk.CTkImage(Image.open("reload_icon.png").resize((50, 50), Image.LANCZOS))
    yoga_icon = ctk.CTkImage(Image.open("yoga.png").resize((50, 50), Image.LANCZOS))

    # Create the frame on the left for exercise buttons
    left_frame = ctk.CTkFrame(root, width=400, height=300, corner_radius=15, fg_color="#2e2e26")
    left_frame.pack(side="left", pady=20, padx=20, fill="both", expand=True)

    label = ctk.CTkLabel(left_frame, text="Select an Exercise", font=("Helvetica", 50))
    label.pack(pady=25)

    instruction_label = ctk.CTkLabel(left_frame,
                                     text="Wave your left hand up or down to highlight.\nMake a fist to select.",
                                     font=("Helvetica", 20))
    instruction_label.pack(pady=30)

    btn_hand_elevations = ctk.CTkButton(left_frame, text="Hand Elevations", corner_radius=32, width=250, height=45,
                                        fg_color="gray",
                                        image=dumbbell_icon, compound="left",
                                        command=lambda: set_exercise("Hand Elevations"))
    btn_hand_elevations.pack(pady=10)

    btn_jumping_jacks = ctk.CTkButton(left_frame, text="Jumping Jacks", corner_radius=32, width=250, height=45,
                                      fg_color="gray",
                                      image=sports_icon, compound="left",
                                      command=lambda: set_exercise("Jumping Jacks"))
    btn_jumping_jacks.pack(pady=10)

    btn_push_ups = ctk.CTkButton(left_frame, text="Push-Ups", corner_radius=32, width=250, height=45, fg_color="gray",
                                 image=pushup_icon, compound="left",
                                 command=lambda: set_exercise("Push-Ups"))
    btn_push_ups.pack(pady=10)

    btn_yoga = ctk.CTkButton(left_frame, text="Yoga", corner_radius=32, width=250, height=45, fg_color="gray",
                             image=yoga_icon, compound="left",
                             command=lambda: set_exercise("Yoga"))
    btn_yoga.pack(pady=10)

    btn_reload = ctk.CTkButton(left_frame, text="Reload Face Recognition", corner_radius=32, width=250, height=45, fg_color="gray",
                                 image=reload_icon, compound="left",
                                 command=reload_face_recognition)
    btn_reload.pack(pady=10)

    countdown_label = ctk.CTkLabel(left_frame, text="", font=("Helvetica", 24))
    buttons = [btn_hand_elevations, btn_jumping_jacks, btn_push_ups, btn_yoga, btn_reload]
    highlight_button(highlighted_button_index)

    # Create the frame on the right for sliders
    right_frame = ctk.CTkFrame(root, width=400, height=300, corner_radius=15, fg_color="#2e2e26")
    right_frame.pack(side="right", pady=20, padx=20, fill="both", expand=True)

    sets_label = ctk.CTkLabel(right_frame, text="Number of Sets", font=("Helvetica", 20))
    sets_label.pack(pady=10)
    sets_slider = ctk.CTkSlider(right_frame, from_=1, to=3, number_of_steps=2, width=300)
    sets_slider.pack(pady=10)
    sets_slider.set(1)

    sets_help_label = ctk.CTkLabel(right_frame,
                                   text="Adjust the number of sets (1 to 3) using the slider.\nUse a fist gesture to change the value.",
                                   font=("Helvetica", 14))
    sets_help_label.pack(pady=5)

    reps_label = ctk.CTkLabel(right_frame, text="Number of Repetitions", font=("Helvetica", 20))
    reps_label.pack(pady=10)
    reps_slider = ctk.CTkSlider(right_frame, from_=1, to=12, number_of_steps=11, width=300)
    reps_slider.pack(pady=10)
    reps_slider.set(1)

    reps_help_label = ctk.CTkLabel(right_frame,
                                   text="Adjust the number of repetitions (1 to 12) using the slider.\nUse a fist gesture to change the value.",
                                   font=("Helvetica", 14))
    reps_help_label.pack(pady=5)

    difficulty_label = ctk.CTkLabel(right_frame, text="Difficulty", font=("Helvetica", 20))
    difficulty_label.pack(pady=10)
    difficulty_slider = ctk.CTkSlider(right_frame, from_=1, to=3, number_of_steps=2, width=300)
    difficulty_slider.pack(pady=10)
    difficulty_slider.set(1)

    difficulty_help_label = ctk.CTkLabel(right_frame,
                                         text="Adjust the difficulty level (1 to 3) using the slider.\nUse a thumb up gesture to change the value.",
                                         font=("Helvetica", 14))
    difficulty_help_label.pack(pady=5)

    # Create a label for displaying the camera feed
    camera_label = Label(root)
    camera_label.pack(pady=20)

    # Create a label for displaying gender and age information
    gender_age_label = ctk.CTkLabel(root, text="", font=("Helvetica", 24))
    gender_age_label.pack(pady=20)

    def update_ui():
        global highlighted_button_index, total_sets, repetitions, detected_gender_age, last_face_detected_time, face_recognition_active, face_detected, last_action_time
        total_sets = int(sets_slider.get())
        repetitions = int(reps_slider.get())

        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        if face_recognition_active:
            results = yolo_model(frame)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                names = result.names
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    name = names[int(result.boxes.cls[i])]
                    face = frame[y1:y2, x1:x2]

                    if face.shape[0] > 0 and face.shape[1] > 0:
                        detected_gender_age = detect_gender_age(face)

                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        gender_age_label.configure(text=f'{name}: {detected_gender_age}')

                        # Start the timer when a face is detected
                        if not face_detected:
                            face_detected = True
                            last_face_detected_time = time.time()
                        else:
                            # Check if the face has been detected for 2 seconds
                            if time.time() - last_face_detected_time >= 2:
                                face_recognition_active = False

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hand_results = hands.process(image)
        image.flags.writeable = True

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                current_time = time.time()
                if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:  # Left hand
                    gesture = detect_hand_gestures(hand_landmarks.landmark)
                    if current_time - last_action_time > debounce_delay:
                        if gesture == "fist":
                            buttons[highlighted_button_index].invoke()
                            last_action_time = current_time
                        elif gesture == "open":
                            if hand_landmarks.landmark[9].y < 0.4:  # Adjusted threshold for more reliable detection
                                highlighted_button_index = max(0, highlighted_button_index - 1)
                                last_action_time = current_time
                            elif hand_landmarks.landmark[9].y > 0.6:  # Adjusted threshold for more reliable detection
                                highlighted_button_index = min(len(buttons) - 1, highlighted_button_index + 1)
                                last_action_time = current_time
                            highlight_button(highlighted_button_index)
                else:  # Right hand
                    # Adjust sliders with right hand gestures
                    gesture = detect_hand_gestures(hand_landmarks.landmark)
                    if current_time - last_action_time > debounce_delay:
                        if gesture == "fist":
                            if hand_landmarks.landmark[9].y < 0.4:
                                sets_slider.set(min(sets_slider.get() + 1, 3))
                                last_action_time = current_time
                            elif hand_landmarks.landmark[9].y > 0.6:
                                sets_slider.set(max(sets_slider.get() - 1, 1))
                                last_action_time = current_time
                        elif gesture == "open":
                            if hand_landmarks.landmark[9].y < 0.4:
                                reps_slider.set(min(reps_slider.get() + 1, 12))
                                last_action_time = current_time
                            elif hand_landmarks.landmark[9].y > 0.6:
                                reps_slider.set(max(reps_slider.get() - 1, 1))
                                last_action_time = current_time
                        elif gesture == "thumb_up":
                            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                                difficulty_slider.set(min(difficulty_slider.get() + 1, 3))
                            else:
                                difficulty_slider.set(max(difficulty_slider.get() - 1, 1))
                            last_action_time = current_time

        # Convert image to PhotoImage for Tkinter
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        root.after(100, update_ui)

    root.after(100, update_ui)
    root.mainloop()

if __name__ == "__main__":
    create_ui()
