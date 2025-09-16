import torch
from transformers import AutoModel, AutoImageProcessor
from sklearn.neighbors import KNeighborsClassifier
import joblib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("All dependencies imported successfully.")

class DINOv3KNNClassifier:
	def __init__(self, n_neighbors=5, knn_path=None):
		self.model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
		self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
		self.model.eval()
		if knn_path is not None:
			self.knn = joblib.load(knn_path)
		else:
			self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

	def get_DINO_embeddings(self, images):
		inputs = self.processor(images, return_tensors="pt")
		with torch.no_grad():
			outputs = self.model(**inputs)
		return outputs

	def fit(self, pil_images, labels):
		features = self.extract_features(pil_images)
		self.knn.fit(features, labels)

	def predict(self, pil_images):
		features = self.extract_features(pil_images)
		return self.knn.predict(features)