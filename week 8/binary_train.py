#%% packages
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import accuracy_score
#%% Hyperparameters
BATCH_SIZE = 4
EPOCHS = 10

# %% transformation steps
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# %% create datasets
ds_train = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
ds_test = torchvision.datasets.ImageFolder(root='data/test', transform=transform)

# %% create dataloaders
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # (C, H, W) -> (H, W, C)
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
# %%
imshow(torchvision.utils.make_grid(images, nrow=2))
print(labels)
# %%
images.shape
# %%
labels.shape
# %% Neural Network setup
class BinaryClassificationModel(nn.Module):
    def __init__(self) -> None:
        super(BinaryClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc_out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)  # BS, 6, 30, 30
        x = self.relu(x)
        x = self.pool(x)  # BS, 6, 15, 15
        x = self.conv2(x)  # BS, 16, 13, 13
        x = self.relu(x)
        x = self.pool(x)  # BS, 16, 6, 6
        x = torch.flatten(x, 1) # BS, 576
        x = self.fc1(x)  # BS, 128
        x = self.relu(x)
        x = self.fc_out(x)  # BS, 1
        x = self.sigmoid(x)
        return x

#%% create a random input tensor
# model = BinaryClassificationModel()
# input = torch.rand((1, 1, 32, 32))   # BS, C, H, W
# model(input).shape
        

#%% create an instance of the model
model = BinaryClassificationModel()

#%% define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



#%% train the model
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, data in enumerate(train_loader, 0):

        input_imgs, true_labels = data
        # setting the gradients to zero
        optimizer.zero_grad()
        
        # forward pass
        pred_labels = model(input_imgs)
        
        # calculate the loss
        loss = loss_fn(pred_labels, true_labels.reshape(-1, 1).float())
        
        # backward pass
        loss.backward()
        
        # update the weights
        optimizer.step()
        
        # store the losses in a list
        loss_epoch += loss.item()
    losses.append(loss_epoch)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss_epoch:.4f}')


#%% plot the losses
plt.plot(losses)
#%% test the model (prediction and accuracy calculation)
# create prediction and true labels lists
y_test_true = []
y_test_pred = []
# iterate over the test data
for i, data in enumerate(test_loader, 0):
    input_imgs, true_labels = data
    with torch.no_grad():
        pred_labels = model(input_imgs).round()
    
    y_test_true.extend(true_labels.numpy())
    y_test_pred.extend(pred_labels.numpy())


# %%
acc = accuracy_score(y_test_true, y_test_pred)
acc
# %% confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_true, y_test_pred)

# %%
cm
# %%
