import torch
import torch.nn as nn
import numpy as np
import os
from models.classification_model import pipe

# CIFAR-10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10Classifier:
    def __init__(self, model_path="models/cifar10_linear_classifier.pth"):
        self.classifier = nn.Linear(384, 10)

        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            if 'model' in checkpoint:
                self.classifier.load_state_dict(checkpoint['model'])
            else:
                self.classifier.load_state_dict(checkpoint)
            print(f"Loaded weights from {model_path}")
        else:
            print(f"No weights found at {model_path}")

        self.classifier.eval()

    def predict(self, image_path):
        # Get DINO embedding
        features = pipe(image_path)
        embedding = np.array(features[0])[0]  # Class token (384,)

        # Classify
        with torch.no_grad():
            logits = self.classifier(torch.tensor(embedding).float())
            probs = torch.softmax(logits, dim=0)
            pred_idx = torch.argmax(logits).item()

        return {
            'class': CIFAR10_CLASSES[pred_idx],
            'confidence': probs[pred_idx].item(),
            'probabilities': probs.numpy()
        }

classifier = None

def classify_image(image_path):
    global classifier
    if classifier is None:
        classifier = CIFAR10Classifier()
    return classifier.predict(image_path)

if __name__ == "__main__":
    result = classify_image("tmp_imgs/test.png")
    print(f"Prediction: {result['class']} ({result['confidence']:.3f})")
