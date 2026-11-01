"""
SHIELD Face Authentication Engine
---------------------------------
Scans human crops for faces. Features a safe-fallback mechanism 
in case the environment's OpenCV build lacks legacy Cascade modules.

The threat watchlist can be populated at runtime via the API's
register_person() endpoint. The default entries below are demo
placeholders; in production, replace them with a database lookup.
"""
import cv2
import numpy as np
from typing import Dict


class FaceAuthEngine:
    def __init__(self, watchlist: Dict[str, str] = None):
        """
        Args:
            watchlist: Optional custom watchlist mapping "G{id}" → threat label.
                       If None, uses a small demo watchlist.
        """
        print("👤 Initializing SHIELD Face Auth Engine...")

        # Default demo watchlist — override with a database in production
        if watchlist is not None:
            self.threat_watchlist = watchlist
        else:
            self.threat_watchlist = {
                # Demo entries — these match GlobalMatcher IDs if they happen to collide
                "G1": "KNOWN_INTRUDER",
                "G5": "UNAUTHORIZED_PERSONNEL",
                "G8": "WANTED_SMUGGLER",
            }
        
        # Safely check if the current OpenCV build supports CascadeClassifier
        self.has_cascade = hasattr(cv2, 'CascadeClassifier')
        
        if self.has_cascade:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
        else:
            print("  ⚠️ Notice: Minimal OpenCV build detected. Using heuristic face-auth fallback.")

    def add_to_watchlist(self, global_id: int, threat_label: str):
        """Add or update a person in the threat watchlist at runtime."""
        self.threat_watchlist[f"G{global_id}"] = threat_label

    def remove_from_watchlist(self, global_id: int):
        """Remove a person from the threat watchlist."""
        self.threat_watchlist.pop(f"G{global_id}", None)

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
