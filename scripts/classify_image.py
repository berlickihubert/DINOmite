import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classification_model import get_image_embedding

if len(sys.argv) < 2:
	print("Usage: python scripts/classify_image.py <image_path>")
	sys.exit(1)

image_path = sys.argv[1]
embedding = get_image_embedding(image_path)
print("Image embedding vector:", embedding)
print("Embedding shape:", embedding.shape)