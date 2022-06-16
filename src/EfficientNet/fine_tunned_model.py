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
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

    # Verifica se será necessário cogelar os calculos de gradientes
    model = __freeze_trained_layers(model)

    # Criando a nova fully connected
    model.classifier.fc = nn.Linear(1280, 2)

    # Verificando disponibilidade CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        print("CUDA available. Model optimized to use GPU")
    else:
        print("CUDA unavailable. Model optimized to use CPU")

    return model
