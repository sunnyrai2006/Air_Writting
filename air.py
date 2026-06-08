import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import time

from tensorflow.keras.models import load_model
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# LOAD AIR WRITING MODEL
air_model = tf.saved_model.load("saved_model")
air_predict_fn = air_model.signatures["serving_default"]

# LOAD PRETRAINED HAND SIGN MODEL
asl_model = load_model(
    "asl_mediapipe_mlp_model.h5"
)

# LOAD AIR WRITING LABELS
with open("labels.json") as f:
    air_labels = json.load(f)

predicted_text = ""

prediction_timestamp = 0
PREDICTION_TIMEOUT = 5  # seconds

# CURRENT MODE
mode = "AIR"

# MEDIAPIPE HAND TRACKER
base_options = BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

tracker = vision.HandLandmarker.create_from_options(options)

# CAMERA
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# COLORS
WHITE = (255,255,255)
GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
YELLOW = (0,255,255)
GRAY = (70,70,70)

BLACK  = (0,0,0)
PURPLE = (255,0,255)
ORANGE = (0,165,255)
CYAN   = (255,255,0)

selected_draw_color = WHITE

color_buttons = {
    "BLACK":  ((650,600),(690,640), BLACK),
    "WHITE":  ((700,600),(740,640), WHITE),
    "RED":    ((750,600),(790,640), RED),
    "GREEN":  ((800,600),(840,640), GREEN),
    "BLUE":   ((850,600),(890,640), BLUE),
    "YELLOW": ((900,600),(940,640), YELLOW),
    "PURPLE": ((950,600),(990,640), PURPLE),
    "ORANGE": ((1000,600),(1040,640), ORANGE),
    "CYAN":   ((1050,600),(1090,640), CYAN),
}

# DRAWING CANVAS
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

prev_x = None
prev_y = None

# BUTTONS
buttons = {
    "CLEAR": (40, 600, 180, 650),
    "AIR":   (200, 600, 340, 650),
    "HAND":  (360, 600, 500, 650)
}

undo_stack = []

# BUTTON CHECK
def inside_button(x, y, rect):

    x1, y1, x2, y2 = rect

    return x1 < x < x2 and y1 < y < y2

# DRAW UI
def draw_ui(frame):

    overlay = frame.copy()

    # Main Frame
    cv2.rectangle(
        overlay,
        (20,20),
        (1260,700),
        (35,35,35),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.35,
        frame,
        0.65,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (20,20),
        (1260,700),
        (120,120,120),
        2
    )

    # Dynamic Heading
    heading = (
        "AIR WRITING"
        if mode == "AIR"
        else "HAND SIGN PREDICTION"
    )

    text_size = cv2.getTextSize(
        heading,
        cv2.FONT_HERSHEY_DUPLEX,
        1.3,
        3
    )[0]

    x = (FRAME_WIDTH - text_size[0]) // 2

    cv2.putText(
        frame,
        heading,
        (x,60),
        cv2.FONT_HERSHEY_DUPLEX,
        1.3,
        (0,255,255),
        3
    )

    # Buttons Bottom Left
    for name, rect in buttons.items():

        x1,y1,x2,y2 = rect

        color = (60,60,60)

        if name == "AIR" and mode == "AIR":
            color = GREEN

        if name == "HAND" and mode == "HAND":
            color = GREEN

        if name == "CLEAR":
            color = RED

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            -1
        )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            WHITE,
            2
        )

        cv2.putText(
            frame,
            name,
            (x1+20,y1+35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2
        )

    # Color Palette (Only AIR Mode)
    if mode == "AIR":

        cv2.putText(
            frame,
            "COLORS",
            (650,580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2
        )

        for _, data in color_buttons.items():

            p1,p2,color = data

            cv2.rectangle(
                frame,
                p1,
                p2,
                color,
                -1
            )

            if color == selected_draw_color:

                cv2.rectangle(
                    frame,
                    (p1[0]-3,p1[1]-3),
                    (p2[0]+3,p2[1]+3),
                    WHITE,
                    3
                )

# PREPROCESS AIR WRITING
def preprocess_air(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    roi = thresh[y:y+h, x:x+w]

    size = max(w, h) + 40

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = roi

    final = cv2.resize(square, (28,28))

    final = final / 255.0

    final = final.reshape(1,28,28,1)

    return final.astype(np.float32)

# MOUSE CALLBACK
def mouse_callback(event, x, y, flags, param):

    global canvas
    global mode

    if event == cv2.EVENT_LBUTTONDOWN:
        global selected_draw_color

        for _, data in color_buttons.items():

            p1,p2,color = data

            if p1[0] < x < p2[0] and p1[1] < y < p2[1]:

                selected_draw_color = color
                return

        if inside_button(x, y, buttons["CLEAR"]):

            undo_stack.append(canvas.copy())

            canvas = np.zeros(
                (FRAME_HEIGHT, FRAME_WIDTH, 3),
                dtype=np.uint8
            )

        elif inside_button(x, y, buttons["AIR"]):

            mode = "AIR"

        elif inside_button(x, y, buttons["HAND"]):

            mode = "HAND"

# WINDOW
cv2.namedWindow("AI System")
cv2.setMouseCallback(
    "AI System",
    mouse_callback
)

# MAIN LOOP
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = tracker.detect(mp_image)

    draw_ui(frame)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # Index finger
        x = int(hand[8].x * w)
        y = int(hand[8].y * h)

        cv2.circle(
            frame,
            (x,y),
            10,
            GREEN,
            -1
        )

        # Finger states
        index_up = hand[8].y < hand[6].y
        middle_up = hand[12].y < hand[10].y

        # AIR WRITING MODE
        if mode == "AIR":

            if index_up and not middle_up:

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    selected_draw_color,
                    18
                )

                prev_x, prev_y = x, y

            else:

                prev_x = None
                prev_y = None

        # HAND SIGN MODE
        elif mode == "HAND":

            try:

                landmark_data = []

                x_list = []
                y_list = []

                for lm in hand:

                    px = int(lm.x * w)
                    py = int(lm.y * h)

                    x_list.append(px)
                    y_list.append(py)

                    landmark_data.append(lm.x)
                    landmark_data.append(lm.y)
                    landmark_data.append(lm.z)

                # Hand Box
                x1 = max(0, min(x_list) - 40)
                y1 = max(0, min(y_list) - 40)

                x2 = min(w, max(x_list) + 40)
                y2 = min(h, max(y_list) + 40)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    BLUE,
                    3
                )

                # Prediction
                landmark_data = np.array(
                    landmark_data
                ).reshape(1, -1)

                prediction = asl_model.predict(
                    landmark_data,
                    verbose=0
                )

                class_id = np.argmax(
                    prediction
                )

                confidence = float(
                    np.max(prediction)
                )

                labels = {
                    0:'A',
                    1:'B',
                    2:'C',
                    3:'D',
                    4:'E',
                    5:'F',
                    6:'G',
                    7:'H',
                    8:'I',
                    9:'J',
                    10:'K',
                    11:'L',
                    12:'M',
                    13:'N',
                    14:'O',
                    15:'P',
                    16:'Q',
                    17:'R',
                    18:'S',
                    19:'T',
                    20:'U',
                    21:'V',
                    22:'W',
                    23:'X',
                    24:'Y',
                    25:'Z'
                }

                predicted_text = labels.get(
                    class_id,
                    "?"
                )

                prediction_timestamp = time.time()

                cv2.putText(
                    frame,
                    f"{predicted_text} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    GREEN,
                    3
                )

            except Exception as e:

                print(
                    "HAND MODE ERROR:",
                    e
                )

    # MERGE CANVAS
    frame = cv2.add(
        frame,
        canvas
    )

    # AUTO CLEAR PREDICTION AFTER 5 SECONDS
    if (
        predicted_text != ""
        and time.time() - prediction_timestamp >= PREDICTION_TIMEOUT
    ):
        predicted_text = ""
  

    # SHOW PREDICTION
    cv2.putText(
        frame,
        f"Prediction: {predicted_text}",
        (250,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        GREEN,
        3
    )

    # SHOW WINDOW
    cv2.imshow(
        "AI System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # AIR WRITING PREDICTION
    if key == ord('p') and mode == "AIR":

        try:

            processed = preprocess_air(canvas)

            if processed is not None:

                tensor = tf.convert_to_tensor(
                    processed,
                    dtype=tf.float32
                )

                prediction = air_predict_fn(
                    tensor
                )

                pred = list(
                    prediction.values()
                )[0].numpy()

                class_id = np.argmax(pred)

                predicted_text = air_labels[
                    str(class_id)
                ]

                prediction_timestamp = time.time()

                print(
                    "Prediction:",
                    predicted_text
                )

        except Exception as e:

            print(
                "Prediction Error:",
                e
            )

    # CLEAR WITH KEY
    if key == ord('c'):

        canvas = np.zeros(
            (FRAME_HEIGHT, FRAME_WIDTH, 3),
            dtype=np.uint8
        )

    # EXIT
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()