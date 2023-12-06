import torch
import torchvision.transforms as transforms
from pytorch.FasterRCNN.models.faster_rcnn import FasterRCNNModel
from pytorch.FasterRCNN.models.vgg16 import VGG16Backbone
from pytorch.FasterRCNN.datasets.voc import Dataset  # Assuming the dataset module is used for the number of classes
from pytorch.FasterRCNN.models.faster_rcnn import FasterRCNNModel
from pytorch.FasterRCNN.models.resnet import ResNetBackbone, Architecture
from pytorch.FasterRCNN.datasets.voc import Dataset  # Assuming the dataset module is used for the number of classes

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

resnet_architecture = Architecture.ResNet152  
backbone = ResNetBackbone(resnet_architecture)

# Initialize the Faster R-CNN model with the VGG16 backbone
# Make sure to use the correct number of classes as used during training
num_classes = 6 # This is an example, replace with the actual number of classes

model = FasterRCNNModel(num_classes=num_classes, backbone=backbone)
checkpoint = torch.load('fasterrcnn_pytorch_resnet152.pth')
class_index_to_name = {
    0: "background",
    1: 'Courteney Cox',
    2: 'David Schwimmer', 
    3: 'Jennifer Aniston', 
    4: 'Matt LeBlanc', 
    5: 'lisa Kudrow'}
  
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # If not, try loading the entire checkpoint as the model state dict
    model.load_state_dict(checkpoint)
    
    
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Define the transformation
# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # Add any other transformations that were used during training
])

# Load and transform the image
image_path = 'test2.jpg'  # Update with the path to your image
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

# Move the image tensor to the same device as the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)

# Set the model to evaluation mode and perform prediction
model.eval()
with torch.no_grad():
    predictions = model(image_tensor)

# Unpack the predictions
bounding_boxes, class_scores, _ = predictions

# Convert to CPU and numpy
bounding_boxes = bounding_boxes.cpu().numpy()
class_scores = class_scores.cpu().detach().numpy()

# Find the highest scoring box for each class
highest_scoring_boxes = {}
print(predictions)
for box, scores in zip(bounding_boxes, class_scores):
    class_idx = scores.argmax()
    score = scores[class_idx]
    print(class_idx)
    # Ignore background class (class_idx 0)
    if class_idx == 0:
        continue

    # Check if this is the highest score for this class
    if class_idx not in highest_scoring_boxes or score > highest_scoring_boxes[class_idx][1]:
        highest_scoring_boxes[class_idx] = (box, score)

# Load the image for visualization
original_image = Image.open(image_path)
plt.figure(figsize=(12, 8))
plt.imshow(original_image)

# Draw the highest scoring boxes
for class_idx, (box, score) in highest_scoring_boxes.items():
    y1, x1, y2, x2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x1, y1, f"Class {class_index_to_name[class_idx]}: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

# Save and show the result
plt.axis('off')
plt.savefig('predicted_highest_per_class.jpg', bbox_inches='tight')
plt.show()