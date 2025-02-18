# Facial-Analysis-for-Emotion-Recognition-and-Lie-Detection

# Overview
This project integrates sentiment analysis and lie detection by analyzing facial expressions and speech patterns. Using real-time video and audio inputs, the system evaluates emotional states and credibility.

# Features

- Facial Expression Analysis: Detects micro-expressions using OpenCV and DeepFace.
- Speech Recognition: Converts speech to text using the SpeechRecognition library.
- Lie Detection: Evaluates truthfulness based on facial cues and speech sentiment analysis.
- Real-Time Processing: Provides instant feedback on emotions and detected deception.

# Technologies Used
- Python
- OpenCV - For facial detection and tracking
- DeepFace - For emotion recognition
- SpeechRecognition - For speech-to-text conversion
- Numpy - For data manipulation
- Threading - For parallel speech recognition

# Installation

- Clone the repository:

git clone https://github.com/your-repo/facial-analysis-lie-detection.git
cd facial-analysis-lie-detection

- Install dependencies:

pip install opencv-python numpy deepface SpeechRecognition

# Usage

Run the main script:

python face_analysis.py

The system will capture real-time video and analyze facial expressions.

It will also listen for spoken statements and analyze sentiment.

Results will be displayed on-screen with detected emotions and lie/truth assessment.

# Applications

Security & Surveillance: Lie detection for interrogations.

Customer Insights: Analyzing emotions in customer service.

Psychological Assessments: Understanding emotional well-being.

# Future Improvements

Expanding the dataset for better accuracy.

Implementing additional biometric indicators.

Enhancing speech sentiment classification.
