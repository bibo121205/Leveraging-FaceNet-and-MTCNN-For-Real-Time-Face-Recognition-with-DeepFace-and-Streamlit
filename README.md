# Face Recognition System

A simple yet effective face recognition system built with Python, leveraging DeepFace with FaceNet 128 embeddings and manual MTCNN face detection and alignment:
- Register multiple individuals using webcam or uploaded images
- Recognize faces in real-time with confidence scores
- Store face embeddings persistently in a database
- Provide both command-line and web interface options

## Technologies Used

- **DeepFace + FaceNet 128**: Generates 128-dimensional facial embeddings for robust face representation
- **MTCNN**: Manual implementation for face detection, landmark detection, and face alignment
- **OpenCV**: Image processing and webcam handling
- **Streamlit**: Web interface for easy interaction
- **scikit-learn**: Cosine similarity calculations for face matching

## Key Features

- **Face Detection & Alignment**: MTCNN detects faces and extracts facial landmarks (eyes) for proper alignment
- **Embedding Generation**: DeepFace with FaceNet model produces 128-dimensional embeddings
- **Face Registration**: Register multiple individuals with webcam or image upload
- **Real-time Recognition**: Live webcam recognition with bounding boxes and confidence scores
- **Persistent Storage**: Automatic save/load of face database using pickle

## Usage

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/bibo121205/Leveraging-FaceNet-and-MTCNN-For-Real-Time-Face-Recognition-with-DeepFace-and-Streamlit.git
cd Leveraging-FaceNet-and-MTCNN-For-Real-Time-Face-Recognition-with-DeepFace-and-Streamlit
```
### 2. Install Dependencies
Install all the required Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit Application
Launch the application by running the main.py file with Streamlit:

```bash
streamlit run main.py
```
or
```bash
python -m streamlit run main.py
```

Your web browser should open with the application running. You can now use the interface to perform real-time face recognition.

## File Structure

```
facial-recognition/
├── main.py              # Core face recognition class
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
└── db/
    └── face_db.pkl      # Face database storage
```

## Why This Approach?

1. **DeepFace + FaceNet**: Provides near state-of-the-art 128-dimensional embeddings that capture facial features effectively
2. **Manual MTCNN**: Gives fine-grained control over face detection, landmark extraction, and alignment process
3. **Eye Alignment**: Ensures consistent face orientation for better embedding quality
4. **Modular Design**: Easy to extend and modify individual components

This system demonstrates how to combine powerful pre-trained models (DeepFace) with manual computer vision techniques (MTCNN) to create a robust face recognition solution. 
