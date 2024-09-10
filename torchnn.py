import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 1. Model Definition with Adaptive Pooling
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Pool to 1x1 to avoid hardcoding dimensions
            nn.Flatten(), 
            nn.Linear(64, 10)  # Final layer to output 10 classes (0-9)
        )

    def forward(self, x): 
        return self.model(x)

# 2. Data Augmentation for Training
train_transform = transforms.Compose([
    transforms.RandomRotation(10),    # Random rotation between -10 to 10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Train the Model
def train_model(model, train_loader, num_epochs=10):
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')

            # Forward pass
            yhat = model(X)
            loss = loss_fn(yhat, y)

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Initialize model
clf = ImageClassifier().to('cpu')

# Train the model
train_model(clf, train_loader)

# 4. Noise Reduction and Preprocessing for Test Image
def preprocess_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 pixels to match MNIST
    img = cv2.resize(img, (28, 28))

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to PIL Image for compatibility with torchvision transforms
    img_pil = Image.fromarray(img_blur)

    # Apply transformations: normalize same as MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor

# 5. Test on Noisy Image
def test_model_on_image(model, image_path):
    # Preprocess the noisy image
    img_tensor = preprocess_image(image_path).to('cpu')

    # Model in evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output)

    # Get softmax probabilities
    probs = F.softmax(output, dim=1)
    confidence = probs[0][predicted].item()
    
    print(f"Predicted Label: {predicted.item()}, Confidence: {confidence}")

    # Visualize the processed image
    img_np = img_tensor.squeeze().cpu().numpy()
    plt.imshow(img_np, cmap='gray')
    plt.title(f"Predicted: {predicted.item()}, Confidence: {confidence}")
    plt.show()

# Test the model on img_4.jpg (the noisy outlier)
test_image_path = 'img_4.jpg'  # Path to the noisy image
test_model_on_image(clf, test_image_path)
