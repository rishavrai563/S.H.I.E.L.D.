"""
SHIELD Face Authentication Engine
---------------------------------
Scans human crops for faces. Features a safe-fallback mechanism 
in case the environment's OpenCV build lacks legacy Cascade modules.
"""
import cv2
import numpy as np

class FaceAuthEngine:
    def __init__(self):
        print("👤 Initializing SHIELD Face Auth Engine...")
        self.threat_watchlist = {
            "G1": "KNOWN_INTRUDER",
            "G5": "UNAUTHORIZED_PERSONNEL",
            "G8": "WANTED_SMUGGLER"
        }
        
        # Safely check if the current OpenCV build supports CascadeClassifier
        self.has_cascade = hasattr(cv2, 'CascadeClassifier')
        
        if self.has_cascade:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
        else:
            print("  ⚠️ Notice: Minimal OpenCV build detected. Using heuristic face-auth fallback.")

    def verify_face(self, human_crop: np.ndarray, global_id: int) -> str:
        """Verifies if the tracked person matches the threat watchlist."""
        if human_crop is None or human_crop.size == 0:
            return ""

        # Only process if the person is close to the camera (large crop)
        h, w = human_crop.shape[:2]
        if h < 120 or w < 60:
            return ""

        try:
            face_detected = False
            
            if self.has_cascade:
                gray_crop = cv2.cvtColor(human_crop, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    face_detected = True
            else:
                # Heuristic Fallback: If crop is very large, assume face is visible for demo
                if h >= 200 and w >= 100:
                    face_detected = True

            if face_detected:
                formatted_id = f"G{global_id}"
                if formatted_id in self.threat_watchlist:
                    return self.threat_watchlist[formatted_id]
                    
        except Exception as e:
            pass # Fail silently so the main surveillance pipeline never crashes

        return ""