Changes from previous author:
- Updated Architecture: Using AdaptiveAvgPool2d ensures that the fully connected layer receives a consistent input size, regardless of the input dimensions.
- Data Augmentation: Training with rotated and shifted images ensures the model becomes more robust to variations, improving generalization.
- Noise Reduction: Preprocessing the image by removing noise helps the model focus on the digit itself.

