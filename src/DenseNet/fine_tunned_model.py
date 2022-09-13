'''
Responsável pela criação do modelo customizado que atende as especificações do projeto
'''
import torch
import torch.nn as nn

# Congela o cálculo de gradiente das outras camadas
def __freeze_trained_layers(model):
    for params in model.parameters():
        params.requires_grad = False
    return model

def create():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1, 1)))

    # Verifica se será necessário cogelar os calculos de gradientes
    model = __freeze_trained_layers(model)

    # Criando a nova fully connected
    model.classifier = nn.Linear(1920, 2)

    # Verificando disponibilidade CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        print("CUDA available. Model optimized to use GPU")
    else:
        print("CUDA unavailable. Model optimized to use CPU")

    return model

create()