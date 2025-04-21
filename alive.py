import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from scipy import signal

class AliveDetectionSystem:
    def __init__(self):
        """
        Initialize the alive detection system using only RGB camera input
        """
        # Mediapipe pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Machine learning model for classification
        self.ml_model = self.create_alive_classification_model()
        
        # Signal history for tracking
        self.signal_history = {
            'breathing': [],
            'pulse': [],
            'movement': []
        }
    
    def create_alive_classification_model(self):
        """
        Create a simple neural network for alive status classification
        """
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )
        return model
    
    def detect_face_and_body(self, frame):
        """
        Detect face and body in the frame
        
        :param frame: Input RGB frame
        :return: Dictionary with face and body detection results
        """
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Detect body using Mediapipe
        pose_results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        return {
            'faces': faces,
            'body_landmarks': pose_results.pose_landmarks
        }
    
    def analyze_breathing(self, frame, body_landmarks):
        """
        Analyze breathing using body landmark movements
        
        :param frame: Input RGB frame
        :param body_landmarks: Mediapipe body landmarks
        :return: Breathing analysis results
        """
        if not body_landmarks:
            return None
        
        # Extract chest and shoulder landmarks
        chest_landmarks = [
            body_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            body_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ]
        
        # Calculate chest movement
        chest_movement = [
            abs(chest_landmarks[0].y - chest_landmarks[1].y)
        ]
        
        # Spectral analysis
        breathing_frequency = self.perform_spectral_analysis(chest_movement)
        
        return {
            'movement': np.mean(chest_movement),
            'frequency': breathing_frequency,
            'is_breathing': 0.1 < breathing_frequency < 0.5
        }
    
    def detect_pulse(self, frame, faces):
        """
        Detect pulse using facial region color variations
        
        :param frame: Input RGB frame
        :param faces: Detected faces
        :return: Pulse detection results
        """
        if len(faces) == 0:
            return None
        
        # Take first detected face
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Extract color channels
        b, g, r = cv2.split(face_roi)
        
        # Pulse signal estimation (Green channel most sensitive to blood flow)
        pulse_signal = np.mean(g)
        signal_variance = np.var(g)
        
        return {
            'pulse_signal': pulse_signal,
            'signal_variance': signal_variance
        }
    
    def perform_spectral_analysis(self, signal_data):
        """
        Perform spectral analysis to detect frequency patterns
        
        :param signal_data: Input signal data
        :return: Dominant frequency
        """
        if len(signal_data) < 2:
            return 0
        
        # Apply windowing
        windowed_signal = signal.windows.hann(len(signal_data)) * signal_data
        
        # Compute FFT
        transformed_signal = np.fft.fft(windowed_signal)
        frequencies = np.fft.fftfreq(len(signal_data))
        
        # Find dominant frequencies
        magnitude_spectrum = np.abs(transformed_signal)
        dominant_indices = np.argsort(magnitude_spectrum)[-3:]
        dominant_frequencies = frequencies[dominant_indices]
        
        return float(np.mean(np.abs(dominant_frequencies)))
    
    def extract_features(self, breathing, pulse):
        """
        Extract features for machine learning classification
        
        :param breathing: Breathing analysis results
        :param pulse: Pulse detection results
        :return: Extracted feature vector
        """
        features = []
        
        # Breathing features
        if breathing:
            features.extend([
                breathing.get('movement', 0),
                breathing.get('frequency', 0)
            ])
        
        # Pulse features
        if pulse:
            features.extend([
                pulse.get('pulse_signal', 0),
                pulse.get('signal_variance', 0)
            ])
        
        # Pad features to fixed length
        features = features[:100] + [0] * max(0, 100 - len(features))
        
        return np.array(features)
    
    def determine_alive_status(self, frame):
        """
        Comprehensive alive detection method
        
        :param frame: Input RGB frame
        :return: Alive detection results
        """
        # Detect face and body
        detection = self.detect_face_and_body(frame)
        
        # Analyze breathing
        breathing_analysis = self.analyze_breathing(frame, detection['body_landmarks']) \
            if detection['body_landmarks'] else None
        
        # Detect pulse
        pulse_detection = self.detect_pulse(frame, detection['faces']) \
            if len(detection['faces']) > 0 else None
        
        # Extract features
        features = self.extract_features(breathing_analysis, pulse_detection)
        
        # Convert to tensor for ML model
        features_tensor = torch.tensor(features).float().unsqueeze(0)
        
        # Predict alive probability
        with torch.no_grad():
            alive_probability = self.ml_model(features_tensor)
        
        return {
            'is_alive': bool(alive_probability[0][1] > 0.5),
            'alive_probability': float(alive_probability[0][1]),
            'breathing': breathing_analysis,
            'pulse': pulse_detection
        }

def main():
    # Initialize the detection system
    alive_detector = AliveDetectionSystem()
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform alive detection
        result = alive_detector.determine_alive_status(frame)
        
        # Display results
        display_text = f"Alive: {result['is_alive']} (P: {result['alive_probability']:.2f})"
        cv2.putText(frame, 
                    display_text, 
                    (10,50), 
                    cv2.FONT_HERSHEY_TRIPLEX, 
                    2, 
                    (12, 255, 0) if result['is_alive'] else (0, 0,255), 
                    2)
        
        # Show frame
        cv2.imshow('Alive Detection', frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()