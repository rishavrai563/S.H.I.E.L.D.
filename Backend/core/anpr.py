"""
Automatic Number Plate Recognition (ANPR) Engine
Powered by EasyOCR for robust CPU/GPU text extraction.
"""
import cv2
import numpy as np
import easyocr
import logging

# Suppress verbose EasyOCR logs
logging.getLogger('easyocr').setLevel(logging.ERROR)

class ANPREngine:
    def __init__(self, use_gpu: bool = False):
        print("🚗 Initializing Production ANPR Engine (EasyOCR)...")
        # Initialize EasyOCR for English. Automatically uses CPU if CUDA is unavailable.
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def extract_plate(self, vehicle_crop: np.ndarray) -> tuple[str, float]:
        """Extracts text from a vehicle crop. Returns (plate_text, confidence)."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "", 0.0

        try:
            # EasyOCR expects RGB images
            rgb_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
            results = self.reader.readtext(rgb_crop)
            
            if not results:
                return "", 0.0
            
            best_text = ""
            best_conf = 0.0
            
            for (bbox, text, conf) in results:
                # Sanitize: Keep only alphanumeric characters
                clean_text = "".join(c for c in text if c.isalnum()).upper()
                
                # Indian plates are typically 8-10 characters (e.g., MH01AB1234)
                if 5 <= len(clean_text) <= 12 and conf > best_conf:
                    best_text = clean_text
                    best_conf = float(conf)
                    
            return best_text, best_conf

        except Exception as e:
            print(f"⚠️ ANPR Error: {e}")
            return "", 0.0