import cv2
import numpy as np
from collections import deque
from datetime import datetime
import tensorflow as tf
import argparse

# The load_model function attempts to load a TensorFlow model
# from the given path and handles exceptions if the loading fails.
def load_model(model_path):
    try:
       # use the below code to call saved Keras models
       model = tf.keras.models.load_model(model_path)
       print("Model loaded successfully.")

       return model
    except Exception as e:
       print("Error loading the model:", e)
       return None

# The preprocess_frame function resizes the frame to the required size, 
# converts it to a float32 type, and expands its dimensions to make it suitable for the model input.
def preprocess_frame(frame, frame_size):
    resized_frame = cv2.resize(frame, frame_size)
    gray_image = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
    three_channel_image = cv2.merge([gray_image, gray_image, gray_image])
    input_frame = three_channel_image.astype(np.float32)
    input_frame = np.expand_dims(input_frame, axis=0)
    return input_frame

# The predict_frame function uses the model to predict 
# the anomaly in the preprocessed frame and returns the rounded prediction.
def predict_frame(model, input_frame):
    # prediction of the class of the frame using Keras applications (pretrained models)
    predictions = model.predict(input_frame)
    prediction = int(np.round(predictions[0]))

    return prediction

# The display_frame function overlays the prediction label on the frame
# displaying it using OpenCV.
def display_frame(frame, label, color):
    frame = cv2.putText(frame, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
    cv2.imshow('Live Stream', frame)

# The save_video function takes a list of frames and writes them into a video file
# using the specified FPS and codec.
def save_video(frames, output_path, fps):
    if len(frames) == 0:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# The main function handles the video capture, frame processing, prediction, display, and recording logic.
# It maintains a buffer of frames to save the pre-anomaly frames and records additional frames when an anomaly is detected.
# The video recording stops once the buffer size is doubled (i.e., after 5 seconds of anomaly detection).
# The loop breaks and resources are released when 'q' is pressed.
def main(args):
    model = load_model(args.model_path)
    
    if model is None:
        return
    
    cap = cv2.VideoCapture(args.video_url)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    frame_size = (224, 224) # Please change it to your own model input size.
    fps = cap.get(cv2.CAP_PROP_FPS)
    buffer_seconds = args.buffer_seconds
    buffer_size = int(buffer_seconds * fps)
    frame_buffer = deque(maxlen=buffer_size) # Doubly Ended Queue
    recording = False
    frames_to_save = []

    window_name = 'Live Stream'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1300, 700)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        input_frame = preprocess_frame(frame, frame_size)
        prediction = predict_frame(model, input_frame)
        
        if prediction == 0:
            label = 'Anomaly'
            color = (0, 0, 255)  # Red(BGR)
            if not recording:
                recording = True
                frames_to_save = list(frame_buffer)
        else:
            label = 'Normal'
            color = (0, 255, 0)  # Green
        
        display_frame(frame, label, color)

        if recording:
            frames_to_save.append(frame)
            if len(frames_to_save) >= 2 * buffer_size:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"anomaly_{timestamp}.avi"
                save_video(frames_to_save, output_path, fps)
                recording = False
                frames_to_save = []

        frame_buffer.append(frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        
        try:
            xPos, yPos, width, height = cv2.getWindowImageRect(window_name)
        except:
            break
        if xPos == -1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection from Live Stream using T-FT Based VAD Model (MobileNetV3Small as base model)")
    parser.add_argument("--model_path", default="D:\Doktora\Tez Kaynak Kodlar\Real-Time-Anomaly-Detection\Best", type=str, help="Path of the MobileNetV3Small model file (saved_model.pb)")
    parser.add_argument("--video_url", default="test_video_2.mp4", type=str, help="URL of the video stream")
    #parser.add_argument("--video_url", default="rtsp://172.19.201.97/stream1", type=str, help="URL of the live video stream")
    parser.add_argument("--buffer_seconds", default=5, type=int, help="Buffer size in seconds for pre-anomaly recording")
    args = parser.parse_args()
    main(args)