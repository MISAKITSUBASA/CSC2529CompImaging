import argparse
import cv2
import torch
import torchvision.transforms as transforms
import os
# Initialize the model architecture
import torchvision.models as models

# Load the ResNet-50 model pre-trained on ImageNet
model = models.resnet50(pretrained=False)

# Load the state dictionary
model_path = 'fasterrcnn_pytorch_resnet50.pth'  # Replace with your model path
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Check for the correct key in the checkpoint (if necessary)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to output video")
args = vars(ap.parse_args())

# Initialize the video stream and writer
vs = cv2.VideoCapture(args["video"])
writer = None

# Directory for saving frames
save_dir = "frame_reco"
os.makedirs(save_dir, exist_ok=True)

# Function to prepare image for model prediction
def prepare_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img)

# Function to process and save frames
def process_and_save_frames(frame, frame_counter):
    # Convert frame to tensor
    tensor_frame = prepare_image(frame)

    # Get model predictions
    with torch.no_grad():
        predictions = model([tensor_frame])[0]

    # Process predictions
    for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
        # Apply threshold to predictions
        if score < 0.5:  # Threshold
            continue
        
        x0, y0, x1, y1 = [int(b) for b in box.tolist()]
        label = ...  # Replace with logic to determine label based on your model's output

        # Draw rectangle and label on frame
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the frame with drawn rectangles and labels
    cv2.imwrite(os.path.join(save_dir, f"frame_{frame_counter}.jpg"), frame)

# Process video frames
frame_counter = 0
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_counter += 1
    process_and_save_frames(frame, frame_counter)

    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Video writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    
    if writer is not None:
        writer.write(frame)

# Release resources
cv2.destroyAllWindows()
vs.release()
if writer is not None:
    writer.release()
