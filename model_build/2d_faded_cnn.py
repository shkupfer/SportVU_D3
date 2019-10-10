import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset, PadSequence, PossessionsDatasetTorch, Simple2DPossSet
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from model_build.nets import ClusterLoss, EncoderCNN, VaryingLengthDecoderRNN, ResCNNEncoder, conv2D_output_size, ResNet, BasicBlock, Bottleneck
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
import torchvision
from math import floor
import random

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
print(device)


class CNN2D(nn.Module):
    def __init__(self, img_x=48, img_y=51, ksizes=((3, 3), (3, 3)), k_ns=(32, 64), dropout_p=0.25, n_chans=11, n_out=1, outfunc=nn.Identity()):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(n_chans, k_ns[0], kernel_size=ksizes[0])
        self.bn1 = nn.BatchNorm2d(k_ns[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(k_ns[0], k_ns[1], kernel_size=ksizes[1])
        self.bn2 = nn.BatchNorm2d(k_ns[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.drop_p = dropout_p

        outshape = conv2D_output_size((img_x, img_y), (0, 0), ksizes[0], (1, 1))
        outshape = (floor((outshape[0] - 2) / 2) + 1, floor((outshape[1] - 2) / 2) + 1)
        outshape = conv2D_output_size(outshape, (0, 0), ksizes[1], (1, 1))
        outshape = (floor((outshape[0] - 2) / 2) + 1, floor((outshape[1] - 2) / 2) + 1)

        self.fc1 = nn.Linear(k_ns[-1] * outshape[0] * outshape[1], n_out)
        self.outfunc = outfunc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = F.dropout2d(x, p=self.drop_p, training=self.training)

        return self.outfunc(x)


pklfiles_dir = '/dbdata/pchans_max24_onefadeimg/'
plot_file = 'test_loss.png'
validation_prop = 0.3
batch_size = 200

# scaler = torchvision.transforms.Resize((96, 102))
scaler = nn.Upsample(scale_factor=(5, 5))
# scaler = torchvision.transforms.Normalize((0, 0), (1, 1))
rflip = lambda tens: tens if random.random() < 0.5 else torch.flip(tens, dims=[1])

pklfile_names = [os.path.join(pklfiles_dir, pfile_name) for pfile_name in os.listdir(pklfiles_dir)]
train_pklfiles, validation_pklfiles = train_test_split(pklfile_names, test_size=validation_prop)

train_dataset = Simple2DPossSet(train_pklfiles, targets=True, transform=rflip, off_ball_only=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

validation_dataset = Simple2DPossSet(validation_pklfiles, targets=True, transform=rflip, off_ball_only=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

epochs = 50
learning_rate = 0.00001

img_x = 48
img_y = 51
in_chans = 6

drop_p = 0.25
ksizes = ((3, 3), (3, 3))
k_ns = (64, 128)
n_out = 1
outfunc = nn.Hardtanh(0, 4)

# model = CNN2D(img_x=img_x, img_y=img_y, n_chans=in_chans, dropout_p=drop_p,
#               ksizes=ksizes, k_ns=k_ns, n_out=n_out, outfunc=outfunc).to(device)

class MyRes(nn.Module):
    def __init__(self, drop_p=0.25, outfunc=nn.Identity(), **kwargs):
        super(MyRes, self).__init__()
        self.resnet = ResNet(**kwargs)
        self.dropout = nn.Dropout2d(p=drop_p)
        self.fc_out = nn.Linear(kwargs['num_classes'], 1)
        self.outfunc = outfunc

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc_out(x)

        x = self.outfunc(x)
        return x


# model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1, n_chans=11).to(device)
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1, n_chans=6).to(device)
# model = MyRes(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1, n_chans=11).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss().to(device)

val_losses = []
for n in range(epochs):
    sys.stdout.flush()
    logger.info("Starting epoch %s" % (n + 1))
    model.train()
    for batchnum, (coords_data, targets) in enumerate(train_dataloader):
        coords_data = coords_data.to(device)
        targets = targets.to(device)
        coords_data = scaler(coords_data)

        optimizer.zero_grad()

        outputs = model(coords_data)

        loss = criterion(outputs, targets)
        print("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
        sys.stdout.flush()
        loss.backward()
        optimizer.step()

    total_val_loss = 0
    model.eval()
    with torch.no_grad():
        for val_batchnum, (val_coords_data, val_targets) in enumerate(validation_dataloader):
            val_coords_data = val_coords_data.to(device)
            val_targets = val_targets.to(device)
            val_coords_data = scaler(val_coords_data)

            val_outputs = model(val_coords_data)

            batch_val_loss = F.mse_loss(val_outputs, val_targets, reduction='sum')
            total_val_loss += batch_val_loss.item()

    avg_val_loss = total_val_loss / len(validation_dataset)

    val_losses.append(avg_val_loss)
    print("Average MSE loss for epoch %s: %s" % (n + 1, avg_val_loss))
    if avg_val_loss > (val_losses[-2] if len(val_losses) >= 2 else 999):
        # break
        pass


plt.figure(1)
plt.plot(range(1, epochs + 1), val_losses, label='Training Set')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.xticks(range(0, epochs + 1))
plt.savefig(plot_file)

try:
    plt.show()
except:
    logger.info("Unable to show plot")
