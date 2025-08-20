from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

# Should be changed to DINOv3 when later
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small")

def get_image_embedding(image_path):
	image = Image.open(image_path)
	inputs = processor(images=image, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**inputs)
		embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
	return embedding
