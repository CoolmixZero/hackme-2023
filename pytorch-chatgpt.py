import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data)  # Encode categorical variables
X = data.drop('breach', axis=1)
y = data['breach']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the dataloaders
train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = CustomDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the neural network and move it to the device
net = Net().to(device)

# Define the loss function, optimizer, and number of epochs
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
num_epochs = 50

# Train the neural network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch {} loss: {:.3f}'.format(epoch + 1, running_loss / len(train_dataloader)))

# Evaluate the neural network
y_pred = []
y_true = []
net.eval()
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        y_pred += list(outputs.cpu().detach().numpy().ravel())
        y_true += list(labels.cpu().detach().numpy().ravel())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Calculate the evaluation metrics

acc = accuracy_score(y_true, np.round(y_pred))
prec = precision_score(y_true, np.round(y_pred))
rec = recall_score(y_true, np.round(y_pred))
f1 = f1_score(y_true, np.round(y_pred))
auc_roc = roc_auc_score(y_true, y_pred)

print('Accuracy: {:.3f}'.format(acc))
print('Precision: {:.3f}'.format(prec))
print('Recall: {:.3f}'.format(rec))
print('F1-score: {:.3f}'.format(f1))
print('AUC-ROC: {:.3f}'.format(auc_roc))
