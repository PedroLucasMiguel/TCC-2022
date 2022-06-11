import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
from ResNet50 import fine_tunned_model as ftm
from Arfflib import Arfflib as arff
from ResNet50 import Resnet50_HP as hp

##########################################################################################
# Objetos dos ARFF files
#arff_max_pooling2d_1 = arff("max_pooling2d_1", 193600)
#arff_activation_4_relu = arff("activation_4_relu", 774400)
arff_activation_48_relu = arff("activation_48_relu", 25088)
#arff_activation_49_relu = arff("activation_49_relu", 100352)
arff_avg_pool = arff("avg_pool", 2048)

# Arrays temporarios
hook_result = {}

##########################################################################################

#Forward hooks
def get_features(name):
    def hook(model, input, output):
        if name == "max_pooling2d_1":
            aux_array = output.cpu().detach().numpy()
            aux_shape = aux_array.shape
            aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
            aux_array = aux_array.flatten()
            hook_result[name] = aux_array

        elif name == "activation_4_relu":
            aux_array = output.cpu().detach().numpy()
            aux_shape = aux_array.shape
            aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
            aux_array = aux_array.flatten()
            hook_result[name] = aux_array

        elif name == "activation_48_relu":
            relu = nn.ReLU(inplace=True)
            aux_array = relu(output)
            aux_array = aux_array.cpu().detach().numpy()
            aux_shape = aux_array.shape
            aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
            aux_array = aux_array.flatten()
            hook_result[name] = aux_array

        elif name == "activation_49_relu":
            aux_array = output.cpu().detach().numpy()
            aux_shape = aux_array.shape
            aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
            aux_array = aux_array.flatten()
            hook_result[name] = aux_array

        elif name == "avg_pool":
            aux_array = output.cpu().detach().numpy()
            aux_shape = aux_array.shape
            aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
            aux_array = aux_array.flatten()
            hook_result[name] = aux_array

    return hook


# Criando o modelo e carregando os pesos
model = ftm.create(2, False, False)
model.load_state_dict(torch.load('model_chekpoint/best_model_24_f1=0.8872.pt'))

# Registrando o método de forward hook
#model.maxpool.register_forward_hook(get_features('max_pooling2d_1'))
#model.layer1[0].relu.register_forward_hook(get_features('activation_4_relu'))
model.layer4[2].bn2.register_forward_hook(get_features('activation_48_relu'))
#model.layer4[2].relu.register_forward_hook(get_features('activation_49_relu'))
model.avgpool.register_forward_hook(get_features('avg_pool'))

# Criando o dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('DataSet', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=hp.BATCH_SIZE, num_workers=0)

output_value = None
model.eval()
print("Starting extraction...")
for batch_idx, (data, target) in enumerate(train_loader):
    # Se CUDA estiver disponivel, move os tensors para a GPU
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    # Foward pass
    output = model(data)
    if target[0] == 0:
        output_value = 0
    else:
        output_value = 1

    #arff_max_pooling2d_1.append(hook_result["max_pooling2d_1"], output_value)
    #arff_activation_4_relu.append(hook_result["activation_4_relu"], output_value)
    arff_activation_48_relu.append(hook_result["activation_48_relu"], output_value)
    #arff_activation_49_relu.append(hook_result["activation_49_relu"], output_value)
    arff_avg_pool.append(hook_result["avg_pool"], output_value)

print("Extraction completed")
