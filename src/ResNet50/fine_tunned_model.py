'''
Responsável pela criação do modelo customizado que atende as especificações do projeto
'''
import torch
import torch.nn as nn
from custom_model import resnet50_model as net

# Congela o cálculo de gradiente das outras camadas
def __freeze_trained_layers(model):
    for params in model.parameters():
        params.requires_grad = False
    return model

def create(n_classes: int, pre_trained: bool, train_just_fc: bool):
    model = net.resnet50(pre_trained)
    print(model)
    # Verifica se será necessário cogelar os calculos de gradientes
    if train_just_fc:
        model = __freeze_trained_layers(model)

    # Criando a nova fully connected
    model.fc = nn.Linear(2048, n_classes)

    # Verificando disponibilidade CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        print("CUDA available. Model optimized to use GPU")
    else:
        print("CUDA unavailable. Model optimized to use CPU")

    return model

create(2, False, False)