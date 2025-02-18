import cv2
import numpy as np
from collections import deque, Counter
from deepface import DeepFace
import speech_recognition as sr
import threading

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def analyze_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    eye_count = 0
    detected_emotions = []

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        roi_gray = gray_frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_count += len(eyes)

        try:
            analysis = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            detected_emotions.append(emotion)
        except Exception:
            continue

    return detected_emotions, eye_count

def listen_for_statements(result_holder):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            print("Listening...")
            audio = recognizer.listen(source)

            try:
                statement = recognizer.recognize_google(audio)
                print(f"You said: {statement}")
                result_holder.append(statement.lower())
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

frame_skip = 5
emotion_history = deque(maxlen=10)
stable_label = "Unknown"
stable_color = (255, 255, 255)
frame_count = 0
stable_emotion = "Unknown"  # Initialize stable_emotion
statements = []

# Start the speech recognition thread
speech_thread = threading.Thread(target=listen_for_statements, args=(statements,))
speech_thread.daemon = True
speech_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        detected_emotions, eye_count = analyze_faces(frame)

        if detected_emotions:
            emotion_history.append(detected_emotions[0])

        if emotion_history:
            most_common_emotion = Counter(emotion_history).most_common(1)
            stable_emotion = most_common_emotion[0][0] if most_common_emotion else "Unknown"

            if stable_emotion in ["happy", "surprised"] and eye_count > 0:
                stable_label = "Truth"
                stable_color = (0, 255, 0)
            elif stable_emotion in ["sad", "angry"] or eye_count == 0:
                stable_label = "Lie"
                stable_color = (0, 0, 255)
            else:
                stable_label = "Uncertain"
                stable_color = (255, 255, 0)

        if statements:
            statement = statements[-1]

            # Check if any of the specified keywords are in the statement
            keywords = ["swayam", "6 feet", "btech", "artificial intelligence", "data science"]
            if any(keyword in statement for keyword in keywords):
                stable_label = "Truth"
                stable_color = (0, 255, 0)
            else:
                stable_label = "Lie"
                stable_color = (0, 0, 255)

    frame_count += 1

    emotion_colors = {
        "happy": (0, 255, 0),
        "sad": (255, 0, 0),
        "angry": (0, 0, 255),
        "surprised": (0, 255, 255),
        "neutral": (255, 255, 0),
        "unknown": (255, 255, 255)  # Added unknown color
    }

    emotion_color = emotion_colors.get(stable_emotion, (255, 255, 255))

    cv2.putText(frame, f"Emotion: {stable_emotion}", (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
    cv2.putText(frame, f"Result: {stable_label}", (frame.shape[1] - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, stable_color, 2)

    cv2.imshow('Face Detection - Truth or Lie', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
