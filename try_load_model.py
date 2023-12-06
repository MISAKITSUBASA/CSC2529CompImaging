import torch
from torchvision.models import resnet50

model_path = 'fasterrcnn_pytorch_resnet50.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Print the keys in the checkpoint
print(checkpoint.keys())

# Initialize the model architecture
model = resnet50(pretrained=False)

# Load the saved state dict
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # If the checkpoint is a raw state dict, load it directly
    model.load_state_dict(checkpoint)

model.eval()
