import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import cv2
from PIL import Image
import pickle
from deepface import DeepFace

class SimpleFaceRecognition():
    def __init__(self, detector = MTCNN(), threshold = 0.8):
        self.detector = detector
        self.threshold = threshold
        self.db = {}
        self.model_name = "Facenet"
        print(f"Initialized DeepFace with {self.model_name}")
        self.load_database()

    def extract_face(self, image, size = (160,160), margin = 10):
        image_np = np.asarray(image)
        results = self.detector.detect_faces(image_np)
        if not results:
            print("Did not detected any faces.")
            return 
        best_face = max(results, key = lambda x: x['confidence'])
        x1, y1, w, h = best_face['box']
        x1, y1 = max(x1 - margin, 0), max(y1 - margin, 0)
        x2, y2 = x1 + w + 2*margin, y1 + h + 2*margin

        face = image_np[y1:y2, x1:x2] 
        key_points = best_face['keypoints']
        left_eye = key_points['left_eye']
        right_eye = key_points['right_eye']

        #Calculate rotation angle to align
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = tuple(np.array(face.shape[1::-1]) / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(face, rot_matrix, face.shape[1::-1], flags=cv2.INTER_LINEAR)
        face_image = Image.fromarray(aligned_face).resize(size)
        return np.asarray(face_image)

    def get_embedd(self, face_image):
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = face_image
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get embedding using DeepFace
            embedding_result = DeepFace.represent(img_path=face_rgb, model_name=self.model_name, enforce_detection=False)
            
            # Handle different return formats
            if isinstance(embedding_result, list):
                # DeepFace may return a list of dicts
                if len(embedding_result) > 0 and isinstance(embedding_result[0], dict) and 'embedding' in embedding_result[0]:
                    embedding = embedding_result[0]['embedding']
                else:
                    embedding = embedding_result[0]
            elif isinstance(embedding_result, dict):
                embedding = embedding_result['embedding']
            else:
                embedding = embedding_result
            
            embedding = np.array(embedding)
            return embedding
            
        except Exception as e:
            print(f"DeepFace error: {e}")
            return None

    def register(self, name, frames = 30):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not open webcam"
        
        embeddings = []
        frame_count = 0
        print(f"CAPTURING {frames} SAMPLES FOR '{name}.....'")
        while len(embeddings) < frames:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.extract_face(Image.fromarray(frame_rgb))

            if face is not None:
                emb = self.get_embedd(face)
                if emb is not None:
                    embeddings.append(emb)
                    frame_count += 1
                    print(f"CAPTURED {frame_count}/{frames}")

            cv2.imshow("Registering....", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        if embeddings:
            avg_embedding = np.mean(embeddings, axis = 0)
            self.db[name] = avg_embedding
            self.save_database()
            return f"{name} registered!!!"
        else:
            return "No face captured."
        
    def recognize(self, face_image):
        if not self.db:
            return "No registered face in database"
        face = self.extract_face(face_image)
        if face is None:
            return "No face detected"
        embedding = self.get_embedd(face)
        if embedding is None:
            return "Failed to generate embedding"
            
        best_match = None
        best_score = 0
        for name, stored_embedding in self.db.items():
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
            if similarity > best_score and similarity > self.threshold:
                best_score = similarity
                best_match = name
        
        if best_match:
            return f"Recognized: {best_match} (confidence: {best_score:.2f})"
        else:
            return "Unknown face, please register first."

    def recognize_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Camera is not opened."
        
        print("Starting real-time recognition... Press 'q' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)

            if results:
                best_face = max(results, key=lambda x: x['confidence'])
                x, y, w, h = best_face['box']
                x, y = max(x, 0), max(y, 0)

                aligned = self.extract_face(Image.fromarray(frame_rgb))
                if aligned is not None:
                    emb = self.get_embedd(aligned)
                    if emb is not None:
                        best_match = None
                        best_score = 0
                        for name, db_emb in self.db.items():
                            score = cosine_similarity([emb], [db_emb])[0][0]
                            if score > best_score:
                                best_score = score
                                best_match = name

                        if best_score > self.threshold:
                            label = f"{best_match} ({best_score:.2f})"
                            box_color = (0, 255, 0)
                        else:
                            label = f"Unknown ({best_score:.2f})"
                            box_color = (0, 0, 255)
                    else:
                        label = "Embedding failed"
                        box_color = (0, 0, 255)
                else:
                    label = "Face alignment failed"
                    box_color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            else:
                cv2.putText(frame, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Real-time Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_database(self, filename='face_db.pkl'):
        """Save the face database to a file in the db/ folder"""
        os.makedirs('db', exist_ok=True)
        db_path = os.path.join('db', filename)
        with open(db_path, 'wb') as f:
            pickle.dump(self.db, f)

    def load_database(self, filename='face_db.pkl'):
        """Load the face database from a file in the db/ folder"""
        db_path = os.path.join('db', filename)
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.db = pickle.load(f) 