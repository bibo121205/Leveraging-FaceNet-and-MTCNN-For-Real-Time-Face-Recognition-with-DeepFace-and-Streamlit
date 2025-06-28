import streamlit as st
import numpy as np
from PIL import Image
import cv2
from main import SimpleFaceRecognition

recognizer = SimpleFaceRecognition()
st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("Real-Time Face Recognition")
st.markdown("Register or recognize a face using DeepFace + Streamlit.")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section", ["Register", "Recognize", "About"])

def show_result(name, confidence):
    st.success(f"**{name}** recognized with confidence **{confidence:.2f}**")

def show_unknown(confidence):
    st.warning(f"Unknown face. Cosine similarity: {confidence:.2f}")

def show_image(image, caption):
    st.image(image, caption=caption, use_column_width=True)

if section == "Register":
    st.header("Register a New Face")

    # Always show name input and file upload
    name = st.text_input("Enter your name:")
    image = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    # Camera toggle
    if "camera_enabled" not in st.session_state:
        st.session_state.camera_enabled = False

    if st.button("Enable Camera"):
        st.session_state.camera_enabled = True

    if st.session_state.camera_enabled:
        cam_image = st.camera_input("Take a photo")
        if st.button("Disable Camera"):
            st.session_state.camera_enabled = False
        if cam_image:
            image = cam_image  # camera overrides upload

    # Main registration logic
    if name and image:
        with st.spinner("Processing..."):
            pil_image = Image.open(image)
            face = recognizer.extract_face(pil_image)
            if face is not None:
                embedding = recognizer.get_embedd(face)
                if embedding is not None:
                    recognizer.db[name] = embedding
                    recognizer.save_database()
                    show_image(pil_image, f"Registered {name}")
                    st.success(f"{name} has been registered.")
                else:
                    st.error("Failed to generate embedding.")
            else:
                st.error("No face detected.")
    elif not name:
        st.info("Please enter a name to register.")

elif section == "Recognize":
    st.header("Recognize a Face")
    realtime = st.checkbox("Use real-time webcam (OpenCV)")

    if realtime:
        if st.button("Start Webcam"):
            st.info("Press 'Q' in the OpenCV window to stop.")
            recognizer.recognize_live()
    else:
        image = st.camera_input("Take a photo") or st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])
        if image:
            with st.spinner("Analyzing..."):
                pil_image = Image.open(image)
                result = recognizer.extract_face(pil_image)
                if result is not None:
                    emb = recognizer.get_embedd(result)
                    if emb is not None:
                        best_match = None
                        best_score = 0
                        for name, db_emb in recognizer.db.items():
                            score = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
                            if score > best_score:
                                best_score = score
                                best_match = name
                        if best_score > recognizer.threshold:
                            show_result(best_match, best_score)
                        else:
                            show_unknown(best_score)
                        show_image(pil_image, "Input Image")
                    else:
                        st.error("Embedding failed.")
                else:
                    st.error("No face detected.")

elif section == "About":
    st.header("About this App")
    st.markdown("""
    This face recognition system is built with:
    - DeepFace for facial embeddings
    - MTCNN for face detection and alignment
    - Streamlit for the user interface
    
    Created as a personal project, is still a work in progress. 
    """)