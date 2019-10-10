import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class CNN3D(nn.Module):
    def __init__(self, t_dim=120, n_chans=2, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.ch1, self.ch2 = 32, 32
        self.k1, self.k2 = (5, 3, 3), (5, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=n_chans, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden1, 1)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x_3d = x_3d.view(x_3d.size(0), x_3d.size(2), x_3d.size(1), x_3d.size(3), x_3d.size(4))
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        x = F.hardtanh(x, 0, 4)
        return x


img_x = 48
img_y = 51
n_imgs = 150
n_chans = 2
fc_hidden = 128
batch_size = 100
epochs = 10
learning_rate = .00001
# pklfiles_dir = '/dbdata/playerchans_no_fade'
pklfiles_dir = '/dbdata/faded_new_possessions'
validation_prop = 0.3
dropout_p = 0.2
plot_file = 'test_loss.png'

if __name__ == "__main__":
    all_pklfiles = os.listdir(pklfiles_dir)
    train_pklfiles, validation_pklfiles = train_test_split(all_pklfiles, test_size=validation_prop, random_state=999)
    train_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in train_pklfiles], playerchans=False, targets=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    validation_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in validation_pklfiles], playerchans=False, targets=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnn_3d = CNN3D(n_chans=n_chans, t_dim=n_imgs, img_x=img_x, img_y=img_y, drop_p=dropout_p, fc_hidden1=fc_hidden).to(device)

    optimizer = optim.Adam(cnn_3d.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)

    val_losses = []
    for n in range(epochs):
        print("Starting epoch %s" % (n + 1))
        cnn_3d.train()
        for batchnum, (coords_data, targets) in enumerate(train_dataloader):
            coords_data = coords_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = cnn_3d(coords_data)

            loss = criterion(outputs, targets)
            print("Batch %s MSE loss: %s" % (batchnum + 1, str(loss.item())))
            loss.backward()
            optimizer.step()

        all_test_targets = []
        all_test_outputs = []
        summed_val_loss = 0
        cnn_3d.eval()
        with torch.no_grad():
            for val_coords_data, val_targets in validation_dataloader:
                val_coords_data = val_coords_data.to(device)
                val_targets = val_targets.to(device)

                val_outputs = cnn_3d(val_coords_data)

                batch_test_loss = F.mse_loss(val_outputs, val_targets, reduction='sum').item()
                summed_val_loss += batch_test_loss

                all_test_targets.extend(val_targets)
                all_test_outputs.extend(val_outputs)

            avg_val_loss = summed_val_loss / len(validation_dataset)
            val_losses.append(avg_val_loss)

            print("Validation MSE loss for epoch %s: %s" % (n + 1, avg_val_loss))

    plt.figure(1)
    plt.plot(range(1, epochs + 1), val_losses, label='Training Set')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.xticks(range(0, epochs + 1))
    # plt.savefig(plot_file)
    try:
        plt.show()
    except:
        print("Unable to show plot")