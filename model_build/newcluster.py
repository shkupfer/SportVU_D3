import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from model_build.nets import KMeansLoss

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
device = torch.device("cpu")

pklfiles_dir = '/dbdata/new_processed_possessions'
plot_file = 'test_loss.png'
validation_prop = 0.3


batch_size = 100
epochs = 20
cnn_hidden_kernels = 15
cnn_out_kernels = 15
lstm_hidden = 64
hid_lin_features = 128
out_features = 12
learning_rate = 0.001

img_x = 48
img_y = 51
n_imgs = 150
# poss_data_len = 7

try_n_clusters = [8, 10, 12]

all_pklfiles = os.listdir(pklfiles_dir)
train_pklfiles, validation_pklfiles = train_test_split(all_pklfiles, test_size=validation_prop, random_state=0)

print("About to initialize PossessionsDatasets")
train_dataset = PossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in train_pklfiles][:500])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# validation_dataset = PossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in validation_pklfiles])
# validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, cnn_hidden_kernels, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(cnn_hidden_kernels, cnn_out_kernels, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(20 * 8 * 9, 720)
        # self.fc2 = nn.Linear(720, 1)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = F.relu(x)
        x = self.pool1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = F.relu(x)
        x = self.pool2(x)
        print(x.size())
        x = x.view(-1, cnn_out_kernels * 9 * 10)
        print(x.size())
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        # self.cnns = [CNN().to(device) for _ in range(n_imgs)]
        self.cnn = CNN().to(device)
        self.rnn = nn.LSTM(input_size=cnn_out_kernels * 9 * 10,
                           hidden_size=lstm_hidden,
                           num_layers=2, bidirectional=True).to(device)
        self.linear = nn.Linear(2 * lstm_hidden * n_imgs, hid_lin_features).to(device)
        self.linear2 = nn.Linear(hid_lin_features, out_features)

    def forward(self, coords_data):
        x = []
        for timestep in range(n_imgs):
            x.append(self.cnn(coords_data[:, timestep, :, :, :]))
        x = torch.stack(x)
        x, (h_n, h_c) = self.rnn(x)
        x = x.view(x.size(1), -1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x

model = Combine()

params = list(model.cnn.parameters()) + list(model.rnn.parameters()) + list(model.linear.parameters()) + list(model.linear2.parameters())

optimizer = optim.Adam(params, lr=learning_rate)
criterion = KMeansLoss(try_n_clusters).to(device)

val_losses = []
for n in range(epochs):
    print("Starting epoch %s" % (n + 1))
    model.train()
    for batchnum, (coords_data, poss_data, targets) in enumerate(train_dataloader):
        coords_data = coords_data.to(device)
        # poss_data = poss_data.to(device)
        # targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(coords_data)

        loss = criterion(outputs)
        print("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
        loss.backward()
        optimizer.step()
