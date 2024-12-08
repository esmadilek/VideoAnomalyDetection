import cv2
from collections import deque
from datetime import datetime
import argparse
import torch
import torch.nn as nn

from transformers import MobileViTForImageClassification
import torchvision.transforms as transforms

# Define your model architecture
class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()

        # Initialize the pre-trained MobileViT model
        self.backbone = MobileViTForImageClassification.from_pretrained(
            "apple/mobilevit-small",  # Pre-trained MobileViT-Small
            num_labels=1000,  # Original number of labels
            ignore_mismatched_sizes=True
        )

        # Adjust the classifier for the required number of classes
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Forward pass through the model
        outputs = self.backbone(pixel_values=x)
        return outputs.logits


# The load_model function attempts to load the model from the given path and handles exceptions if the loading fails.
def load_model(model_path, num_classes=2):
    """
    Load the fine-tuned model from a state dictionary.

    Args:
        model_path (str): Path to the saved state dictionary.
        num_classes (int): Number of output classes.

    Returns:
        MyModel: The loaded model with weights applied.
    """
    try:
        # Initialize the model with the correct number of classes
        model = MyModel(num_classes=num_classes)

        # Load state dict and update only the backbone weights
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Print out the classifier weights and ensure they are loaded
        #print("Loaded model state dict keys:", state_dict.keys())
        #print("Classifier weight shape:", state_dict.get('backbone.classifier.weight', 'Not found'))
        #print("Classifier bias shape:", state_dict.get('backbone.classifier.bias', 'Not found'))

        # Load the backbone weights (skip classifier weights)
        model.backbone.load_state_dict(state_dict, strict=False)

        # Set to evaluation mode
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# The preprocess_frame function resizes the frame to the required size, converts it to grayscale and 
# applies transformations to make it suitable for the model input.
def preprocess_frame(frame, frame_size):
    # Resize the frame to the desired size
    resized_frame = cv2.resize(frame, frame_size)

    # Convert the RGB frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)

    # Duplicate the grayscale channel to create a 3-channel image
    three_channel_image = cv2.merge([gray_frame, gray_frame, gray_frame])

    # Normalize and transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Convert to tensor
    input_tensor = transform(three_channel_image)
    
    # Add batch dimension (to make it 4D: BxCxHxW)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

# The predict_frame function uses the model to predict 
# the anomaly in the preprocessed frame and returns the rounded prediction.
def predict_frame(model, input_frame):
    """
    Predict the class of the input frame using the hybrid model.

    Args:
        model (torch.nn.Module): The trained hybrid model.
        input_frame (torch.Tensor): Preprocessed input tensor of shape (1, 3, H, W).

    Returns:
        int: Predicted class label (e.g., 0 or 1).
    """
    with torch.no_grad():
        # Pass the input through the model
        logits = model(input_frame)  # Shape: (1, num_classes)
        #print("Model output logits:", logits)  # Check logits for both classes

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # Shape: (1, num_classes)        
        #print("Model probabilities:", probabilities)

        # Get the predicted class (index of the maximum probability)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print("Predicted class:", predicted_class)
    return predicted_class

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

    # Inspect the state dictionary
    #inspect_state_dict(args.model_path)
    
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
# Function to load model and print state dict keys

def inspect_state_dict(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # List all keys in the state dictionary
    print("State dictionary keys:")
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection from Live Stream using FT-ViT Model (MobileViT as base model)")
    parser.add_argument("--model_path", default="D:\Doktora\Tez Kaynak Kodlar\Real-Time-Anomaly-Detection\AI_Model\MobileViT_FT.pth", type=str, help="Path of the MobileViT_FT state dictionary")
    parser.add_argument("--video_url", default="test_video_2.mp4", type=str, help="URL of the video stream")
    #parser.add_argument("--video_url", default="rtsp://172.19.201.97/stream1", type=str, help="URL of the video stream")
    parser.add_argument("--buffer_seconds", default=5, type=int, help="Buffer size in seconds for pre-anomaly recording")
    args = parser.parse_args()
    main(args)