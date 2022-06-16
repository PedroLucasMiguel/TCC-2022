import csv
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import numpy as np
from EfficientNet import fine_tunned_model as ftm
from EfficientNet import EfficientNet_HP as hp
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import *
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
from torch.optim.lr_scheduler import StepLR

def writeCSVdata(file_name, dictionary):
    with open(file_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Accuracy", "Loss", "F1"])
        for i in range(1, hp.EPOCHS+1):
            writer.writerow([i, dictionary[i][0], dictionary[i][1], dictionary[i][2]])
        file.close()

data_to_csv = {}
completed_epoch = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criando as transformações do dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Criando dataset
train_data = datasets.ImageFolder('DataSet', transform=transform)

# Realizando o split do dataset
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

# Criando os loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=hp.BATCH_SIZE, sampler=train_sampler,
                                           num_workers=0)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=hp.BATCH_SIZE, sampler=val_sampler,
                                         num_workers=0)

# Model
model = ftm.create()

# Definindo otimizadores e loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=hp.LEARNING_RATE, momentum=hp.MOMENTUM, weight_decay=hp.WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=2, gamma=0.97) # Usado para diminuir a learning rate durante o treinamento

def train_step(engine, batch):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


# Define a trainer engine
trainer = Engine(train_step)

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to("cuda")

        y_pred = model(x)

        return y_pred, y

evaluator = Engine(validation_step)

# Show a message when the training begins
@trainer.on(Events.STARTED)
def start_message():
    print("Start training!")

# Handler can be want you want, here a lambda !
trainer.add_event_handler(
    Events.COMPLETED,
    lambda _: print("Training completed!")
)

# Run evaluator on val_loader every trainer's epoch completed
@trainer.on(Events.EPOCH_COMPLETED)
def run_validation_and_decrease_lr(engine):
    evaluator.run(val_loader)


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


# Accuracy and loss metrics are defined
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion),
    "precision": Precision(average=False, is_multilabel=False),
    "recall": Recall(average=False, is_multilabel=False)
}

# Attach metrics to the evaluator
for name, metric in val_metrics.items():
    metric.attach(evaluator, name)

# Build F1 score
precision = Precision(average=False)
recall = Recall(average=False)
F1 = precision * recall * 2 / (precision + recall + 1e-20)
F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

# and attach it to evaluator
F1.attach(evaluator, "f1")

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

def cvs_save_data(epoch, accuracy, loss, f1):
    data_to_csv[epoch] = [accuracy, loss, f1]

@trainer.on(Events.EPOCH_COMPLETED)
def run_train_validation(engine):
    train_evaluator.run(train_loader)

@evaluator.on(Events.COMPLETED)
def log_validation_results():
    metrics = evaluator.state.metrics
    print(
        "\nValidation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Avg F1: {:.2f} Precision: {} Recall {}"
        .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"], metrics["f1"], metrics["precision"],
                metrics["recall"]))
    cvs_save_data(trainer.state.epoch, metrics["accuracy"], metrics["loss"], metrics["f1"])

@train_evaluator.on(Events.COMPLETED)
def log_train_results():
    metrics = train_evaluator.state.metrics
    print("  Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))
    
    scheduler.step()
    print('Scheduler step! LR = {}'.format(scheduler.optimizer.param_groups[0]['lr']))

ProgressBar().attach(trainer, output_transform=lambda x: {'batch loss': x})

# Score function to select relevant metric, here f1
def score_function(engine):
    return engine.state.metrics["f1"]

# Checkpoint to store n_saved best models wrt score function
model_checkpoint = ModelCheckpoint(
    "model_chekpoint",
    filename_prefix="best-efficientnet",
    score_function=score_function,
    score_name="f1",
    require_empty=False,
    global_step_transform=global_step_from_engine(trainer),
)

# Save the model (if relevant) every epoch completed of evaluator
evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

trainer.run(train_loader, max_epochs=hp.EPOCHS)

writeCSVdata("data-effientnet", data_to_csv)