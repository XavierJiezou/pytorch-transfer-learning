# Transfer Learning for Computer vision tutorial
## why
In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.
## how
These two major transfer learning scenarios look as follows:
- Finetuning the convnet: Instead of random initializaion, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
- ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.
# introduce
The problem we’re going to solve today is to train a model to classify ants and bees. We have about 120 training images each for ants and bees. There are 75 validation images for each class. Usually, this is a very small dataset to generalize upon, if trained from scratch. Since we are using transfer learning, we should be able to generalize reasonably well.
# download dataset
> [https://ghgxj.lanzous.com/i9EGHiv97za](https://ghgxj.lanzous.com/i9EGHiv97za)

| class | train | val |
|:---:|:---:|:---:|
| ants | 123 | 70  |
| bees | 121 | 83  |
| **总计** | **244** | **153** |
# install module
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
```
# data augmentation
```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```
# load dataset
```python
image_datasets = {
    x: datasets.ImageFolder(
        root=os.path.join('./hymenoptera_data', x),
        transform=data_transforms[x]
    ) for x in ['train', 'val']
}
```
# make dataloaders
```python
dataloaders = {
    x: DataLoader(
        dataset=image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=0
    ) for x in ['train', 'val']
}
```
# others
```python
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''output
{'train': 244, 'val': 153}
['ants', 'bees']
device(type='cuda', index=0)
'''
```
`dataset_sizes`：size of datasets; training dataset: `244` images, validation dataset: `153` images。
`class_names`：names of class, `ants` and`bees`。
`device`：device to work on; GPU is recommended, if not CPU can replace.
# visualize a few images
```python
inputs, labels = next(iter(dataloaders['train']))
grid_images = torchvision.utils.make_grid(inputs)

def no_normalize(im):
    im = im.permute(1, 2, 0)
    im = im*torch.Tensor([0.229, 0.224, 0.225])+torch.Tensor([0.485, 0.456, 0.406])
    return im

grid_images = no_normalize(grid_images)
plt.title([class_names[x] for x in labels])
plt.imshow(grid_images)
plt.show()
```
![](https://img-blog.csdnimg.cn/20201128164559874.png#pic_center)
# train the model
```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    t1 = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        print(
            f'EPOCH: {epoch+1:0>{len(str(num_epochs))}}/{num_epochs}',
            f'LR: {lr:.4f}',
            end=' '
        )
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.argmax(1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print
            if phase == 'train':
                print(
                    f'LOSS: {epoch_loss:.4f}',
                    f'ACC: {epoch_acc:.4f} ',
                    end=' '
                )
            else:
                print(
                    f'VAL-LOSS: {epoch_loss:.4f}',
                    f'VAL-ACC: {epoch_acc:.4f} ',
                    end='\n'
                )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    t2 = time.time()
    total_time = t2-t1
    print('-'*10)
    print(
        f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s',
        f'BEST-VAL-ACC: {best_acc:.4f}'
    )
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```
## finetune the convnet
```python
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# training
model_ft = train_model(
    model_ft, 
    criterion, 
    optimizer_ft, 
    exp_lr_scheduler,
    num_epochs=10
)
```
## convNet as fixed feature extractor
```python
model_conv = models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as opposed to before
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=1e-3, momentum=0.9)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

# training
model_conv = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    exp_lr_scheduler,
    num_epochs=10
)
```
## comparison
- finetuning the convnet
```python
EPOCH: 01/10 LR: 0.0010 LOSS: 0.8586 ACC: 0.5902  VAL-LOSS: 0.2560 VAL-ACC: 0.9020
EPOCH: 02/10 LR: 0.0010 LOSS: 1.1052 ACC: 0.6803  VAL-LOSS: 0.3033 VAL-ACC: 0.8758
EPOCH: 03/10 LR: 0.0010 LOSS: 0.6706 ACC: 0.7910  VAL-LOSS: 0.9216 VAL-ACC: 0.8039
EPOCH: 04/10 LR: 0.0010 LOSS: 0.7949 ACC: 0.7623  VAL-LOSS: 0.2686 VAL-ACC: 0.8954
EPOCH: 05/10 LR: 0.0010 LOSS: 0.5725 ACC: 0.7500  VAL-LOSS: 0.3638 VAL-ACC: 0.8431
EPOCH: 06/10 LR: 0.0001 LOSS: 0.3003 ACC: 0.8525  VAL-LOSS: 0.2749 VAL-ACC: 0.8758
EPOCH: 07/10 LR: 0.0001 LOSS: 0.4123 ACC: 0.8197  VAL-LOSS: 0.2747 VAL-ACC: 0.8889
EPOCH: 08/10 LR: 0.0001 LOSS: 0.3650 ACC: 0.8361  VAL-LOSS: 0.2942 VAL-ACC: 0.8758
EPOCH: 09/10 LR: 0.0001 LOSS: 0.3748 ACC: 0.8279  VAL-LOSS: 0.2560 VAL-ACC: 0.9020
EPOCH: 10/10 LR: 0.0001 LOSS: 0.3523 ACC: 0.8361  VAL-LOSS: 0.2687 VAL-ACC: 0.9085
----------
TOTAL-TIME: 1m10s BEST-VAL-ACC: 0.9085
```
Training `10` epochs, total time `1m10s`, best val-acc `0.9085`
- convNet as fixed feature extractor
```python
EPOCH: 01/10 LR: 0.0010 LOSS: 0.7262 ACC: 0.6598  VAL-LOSS: 0.2515 VAL-ACC: 0.9085
EPOCH: 02/10 LR: 0.0010 LOSS: 0.5294 ACC: 0.7951  VAL-LOSS: 0.3064 VAL-ACC: 0.8627
EPOCH: 03/10 LR: 0.0010 LOSS: 0.5121 ACC: 0.7746  VAL-LOSS: 0.1943 VAL-ACC: 0.9346
EPOCH: 04/10 LR: 0.0010 LOSS: 0.4977 ACC: 0.7992  VAL-LOSS: 0.1751 VAL-ACC: 0.9477
EPOCH: 05/10 LR: 0.0010 LOSS: 0.5162 ACC: 0.7992  VAL-LOSS: 0.1880 VAL-ACC: 0.9412
EPOCH: 06/10 LR: 0.0001 LOSS: 0.4928 ACC: 0.7869  VAL-LOSS: 0.1695 VAL-ACC: 0.9542
EPOCH: 07/10 LR: 0.0001 LOSS: 0.3889 ACC: 0.8156  VAL-LOSS: 0.1952 VAL-ACC: 0.9412
EPOCH: 08/10 LR: 0.0001 LOSS: 0.3160 ACC: 0.8648  VAL-LOSS: 0.1897 VAL-ACC: 0.9412
EPOCH: 09/10 LR: 0.0001 LOSS: 0.4431 ACC: 0.7828  VAL-LOSS: 0.1689 VAL-ACC: 0.9542
EPOCH: 10/10 LR: 0.0001 LOSS: 0.2999 ACC: 0.8770  VAL-LOSS: 0.2250 VAL-ACC: 0.9346
----------
TOTAL-TIME: 0m45s BEST-VAL-ACC: 0.9542
```
Training `10` epochs, total time `0m46s`, best val-acc `0.9542`
> By comparison, the total time of convNet as fixed feature extractor is shorter and the accuracy rate is higher.
# evaluate
```python
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloaders['val']))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(1)

        plt.figure(figsize=(9, 9))
        for i in range(inputs.size(0)):
            plt.subplot(2,2,i+1)
            plt.axis('off')
            plt.title(f'pred: {class_names[preds[i]]}|true: {class_names[labels[i]]}')
            im = no_normalize(inputs[i].cpu())
            plt.imshow(im)
        plt.savefig('train.jpg')
		plt.show()
```
![](https://img-blog.csdnimg.cn/20201128210657638.png#pic_center)
# save model
> Want more details about save and load model of `pytorch`, click [here](https://blog.csdn.net/qq_42951560/article/details/109545302)
```python
torch.save(model_conv.state_dict(), 'model.pt')
```
# load model
```python
device = torch.device('cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('model.pt', map_location=device))
```
# test model
> Randomly find a few pictures of ants and bees in Bing images, then test the model loaded in the previous step

![](https://img-blog.csdnimg.cn/20201128210243533.png#pic_center)
# deploy online
You can also deploy your machine learning model online if you are interested, like this：
> [https://pytorch-cnn-mnist.herokuapp.com/](https://pytorch-cnn-mnist.herokuapp.com/)

![](https://img-blog.csdnimg.cn/20201116162845484.gif#pic_center)
# by the way
This article only talks about the application of transfer learning in Computer Vision. In fact, it can also be applied to natural language processing.
# cite
> [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)