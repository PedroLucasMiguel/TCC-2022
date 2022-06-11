import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from PIL import Image
import Custom_ResNet50.fine_tunned_model as ftm


model = ftm.create(2, False, True)
model.load_state_dict(torch.load('checkpoints/best_model_41_f1=0.9018.pt'))
print(model.layer1[0].conv1.weight)

model2 = ftm.create(2, True, True)
print("-------------------------------------------")
print(model2.layer1[0].conv1.weight)