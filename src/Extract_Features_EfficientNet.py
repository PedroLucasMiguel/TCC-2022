import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from Arfflib.Arfflib import Arfflib
from EfficientNet import fine_tunned_model as ftm
from EfficientNet import EfficientNet_HP as hp

arff_avg_pool = Arfflib("efficientnet_avg_pool", 1920)

# Arrays temporarios
hook_result = {}

model = ftm.create()
model.load_state_dict(torch.load('D:\\git\\TCC-2022\\src\\model_chekpoint\\best-efficientnet_model_43_f1=0.8867.pt'))

def get_features(name):
    def hook(model, input, output):
        #print(len(output))
        aux_array = output.cpu().detach().numpy()
        aux_shape = aux_array.shape
        #print(aux_shape)
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        hook_result[name] = aux_array

    return hook

model.classifier.pooling.register_forward_hook(get_features('avg_pool'))

# Criando o dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('DataSet', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=hp.BATCH_SIZE, num_workers=0)

output_value = None
model.eval()

epoch = 1

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

    arff_avg_pool.append(hook_result["avg_pool"], output_value)

print("Extraction completed")