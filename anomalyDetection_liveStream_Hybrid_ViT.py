import cv2
from collections import deque
from datetime import datetime
import argparse
import torch
import torch.nn as nn

from transformers import ViTModel
import torchvision.transforms as transforms
from torchvision import  models

# Helper function to load CNN backbone
def get_cnn_backbone(cnn_variant):

    """
    Retrieves the specified CNN backbone for feature extraction and returns its feature extractor along with the feature dimensionality.
    Initializes a CNN backbone for feature extraction.

    Args:
        cnn_variant (str): The name of the CNN backbone to use.
    
    Returns:
        cnn_features (nn.Sequential): CNN feature extraction layers.
        feature_dim (int): Number of output features from the CNN (Dimensionality of the extracted features).
    """
    if cnn_variant == 'mobilenetv3_small':
        cnn = models.mobilenet_v3_small(pretrained=True)
        feature_dim = cnn.classifier[0].in_features
        cnn_features = nn.Sequential(*list(cnn.children())[:-1])  # Remove classifier
    else:
        raise ValueError(f"Unknown CNN variant: {cnn_variant}")
    
    return cnn_features, feature_dim

class HybridCNNViT(nn.Module):

    """
    Hybrid model combining CNN feature extraction and a Vision Transformer.
    
    Args:
        cnn_variant (str): Name of the CNN model to use for feature extraction.
        num_classes (int): Number of output classes for classification.
        transformer_dim (int): Embedding size for the Vision Transformer. Default is 768.
    """
    def __init__(self, num_classes, transformer_dim=768):
        super(HybridCNNViT, self).__init__()
        
        # Initialize the CNN backbone
        # Extracts spatial features from images (like edges, textures)
        self.cnn_features, feature_dim = get_cnn_backbone("mobilenetv3_small")
        
        # Initialize the pre-trained Vision Transformer
        # Processes global relationships in the image
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Linear layer to project CNN output to the transformer dimension
        # Ensures CNN features and ViT embeddings are in the same space
        self.cnn_proj = nn.Linear(feature_dim, transformer_dim)
        
        # Fully connected classification head
        # Takes combined CNN and ViT features for final classification
        self.fc = nn.Linear(transformer_dim * 2, num_classes)

    def forward(self, x):
        """
        Forward pass of the hybrid model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """    
        # Step 1: Pass the input image through the CNN backbone
        # This extracts spatial features (B, C, H, W)
        cnn_out = self.cnn_features(x)
        
        # Step 2: Global average pooling to reduce spatial dimensions
        # Output shape: (B, C)
        if cnn_out.dim() == 4:  # Check if spatial dimensions exist
            cnn_out = torch.mean(cnn_out, dim=[2, 3])  # Pool over H and W
        elif cnn_out.dim() == 2:  # Already pooled (e.g., for some CNNs)
            pass
        else:
            raise ValueError(f"Unexpected CNN output dimensions: {cnn_out.shape}")
        
        # Step 3: Project CNN features to the transformer embedding dimension
        # Output shape: (B, transformer_dim)
        cnn_out = self.cnn_proj(cnn_out)

        # Step 4: Pass the input image through the Vision Transformer
        # ViT processes the input and extracts global relationships
        # Output shape: (B, seq_len, transformer_dim)
        vit_output = self.vit(pixel_values=x).last_hidden_state
        
        # Extract the [CLS] token embedding for classification
        # Shape: (B, transformer_dim)
        vit_cls_token = vit_output[:, 0]

        # Step 5: Concatenate CNN and ViT features
        # Combined feature shape: (B, transformer_dim * 2)
        combined_features = torch.cat([cnn_out, vit_cls_token], dim=1)

        # Step 6: Pass the combined features through the classification head
        # Output shape: (B, num_classes)
        x = self.fc(combined_features)
        
        return x
    
# The load_model function attempts to load the model from the given path and handles exceptions if the loading fails.
def load_model(model_path):
    try:
       # Instantiate the hybrid model
       model = HybridCNNViT(num_classes=2)

       # Step 3: Load the fine-tuned state_dict (which should be for 2 classes)
       state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
       # Load the state_dict into the model, ignoring the size mismatch in the classifier
       model.load_state_dict(state_dict, strict=False)

       # Set the model to evaluation mode for inference
       model.eval()

       print("Model loaded successfully.")
       return model
    except Exception as e:
       print("Error loading the model:", e)
       return None

# The preprocess_frame function resizes the frame to the required size, converts it to grayscale and 
# applies transformations to make it suitable for the model input.
def preprocess_frame(frame, frame_size):
    # Resize the frame to the target size
    resized_frame = cv2.resize(frame, frame_size)

    # Convert the RGB frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)

    # Duplicate the grayscale channel to create a 3-channel image
    three_channel_image = cv2.merge([gray_frame, gray_frame, gray_frame])

    # Define transformations: Convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # Define normalization parameters (ImageNet mean and std)
                                    ])
    input_tensor = transform(three_channel_image)  # Convert to tensor with CxHxW dimensions
   
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
        
        # Get the predicted class (index of the maximum probability)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    #print("Predicted class:", predicted_class)
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
    parser = argparse.ArgumentParser(description="Anomaly Detection from Live Stream using HybridCNNViT Model (MobileNetV3Small as base model)")
    parser.add_argument("--model_path", default="D:\Doktora\Tez Kaynak Kodlar\Real-Time-Anomaly-Detection\AI_Model\MobileNetV3Small_ViT.pth", type=str, help="Path of the MobileNetV3Small_ViT state dictionary")
    parser.add_argument("--video_url", default="test_video_2.mp4", type=str, help="URL of the video stream")
    #parser.add_argument("--video_url", default="rtsp://172.19.201.97/stream1", type=str, help="URL of the live video stream")
    parser.add_argument("--buffer_seconds", default=5, type=int, help="Buffer size in seconds for pre-anomaly recording")
    args = parser.parse_args()
    main(args)