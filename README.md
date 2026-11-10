# ✍️ Air Writing Using Computer Vision and Deep Learning

Air Writing is a Python-based computer vision project that allows users to write in the air using their index finger without touching a screen, keyboard, or physical writing surface.

The project uses **OpenCV and MediaPipe** for real-time hand tracking and **TensorFlow** for recognizing the characters drawn in the air.

## 🚀 Features

- **Real-Time Hand Tracking:** Detects and tracks hand landmarks using MediaPipe.
- **Air Writing:** Allows users to draw by moving their index finger in front of the camera.
- **Character Recognition:** Uses a trained TensorFlow model to predict drawn characters.
- **Virtual Drawing Canvas:** Displays the user's finger movements as digital strokes.
- **Undo Function:** Removes the most recent drawing action.
- **Clear Canvas:** Clears the entire drawing area.
- **Camera Switching:** Supports switching between the laptop camera and an external webcam.
- **Live Camera Feed:** Displays the hand tracking and drawing interface in real time.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Computer Vision
- Deep Learning

## 📂 Project Structure

```text
Air_Writting/
│
├── air.py
├── saved_model/
│   ├── assets/
│   ├── variables/
│   └── saved_model.pb
│
├── labels.json
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sunnyrai2006/Air_Writting.git
```

### 2. Navigate to the Project Directory

```bash
cd Air_Writting
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python air.py
```

Make sure the `saved_model` directory, `labels.json`, and `hand_landmarker.task` file are present in the project directory.

## 🎮 Controls

| Key / Action | Function |
|---|---|
| Index Finger Up, Middle Finger Down | Start Air Writing |
| `P` | Predict the drawn character |
| `C` | Clear the canvas |
| `0` | Switch to Laptop Camera |
| `1` | Switch to External Webcam |
| `ESC` | Exit the application |

**Mouse Controls:**

- CLEAR button: Clears the drawing canvas.
- UNDO button: Removes the previous drawing state.

## 🧠 How It Works

1. The webcam captures real-time video frames.
2. MediaPipe Hand Landmarker detects hand landmarks.
3. The application tracks the index finger position.
4. Finger movements are converted into digital strokes on a virtual canvas.
5. When the user presses `P`, the drawing is preprocessed into a 28 × 28 image.
6. The TensorFlow SavedModel processes the image and predicts the corresponding character.
7. The predicted character is displayed on the screen.

## 📌 Requirements

- Python 3.x
- Webcam or external camera
- TensorFlow-compatible environment
- Required Python libraries listed in `requirements.txt`

## 🔮 Future Improvements

- Add word and sentence formation.
- Improve character recognition accuracy.
- Add support for more gestures.
- Develop a graphical user interface.
- Add support for continuous handwriting recognition.

## 👨‍💻 Author

**Sunny Rai**

BCA – Artificial Intelligence and Machine Learning

GitHub: https://github.com/sunnyrai2006

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!
