import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from model_build.nets import ClusterLoss, EncoderCNN, DecoderRNN, ResCNNEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
import torchvision

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
# device = torch.device("cpu")

# pklfiles_dir = '/dbdata/new_processed_possessions'
pklfiles_dir = '/dbdata/faded_new_possessions/'
plot_file = 'test_loss.png'
validation_prop = 0.3
batch_size = 100

# scaler = torchvision.transforms.Resize((96, 102))
# scaler = nn.Upsample(scale_factor=5)
scaler = torchvision.transforms.Normalize((0, 0), (1, 1))

all_pklfiles = os.listdir(pklfiles_dir)
train_pklfiles, validation_pklfiles = train_test_split(all_pklfiles, test_size=validation_prop, random_state=0)

print("About to initialize possessions datasets")
train_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in train_pklfiles], transform=scaler, targets=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

validation_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in validation_pklfiles], transform=scaler, targets=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

epochs = 10
learning_rate = 0.001
# try_n_clusters = list(range(10, 15 + 1))
try_n_clusters = [10]

img_x = 48
img_y = 51
n_imgs = 150

fc_hidden1 = 512
fc_hidden2 = 512
cnn_drop_p = 0.3
CNN_embed_dim = 300

h_RNN_layers = 3
h_RNN = 256
h_FC_dim = 128
rnn_drop_p = 0.3
last_lin_out = 1
outfunc = nn.Hardtanh(0, 4)
# outfunc = nn.Identity()

cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2,
                         drop_p=cnn_drop_p, CNN_embed_dim=CNN_embed_dim).to(device)

# cnn_encoder = ResCNNEncoder(lin_1_in=2048)

rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN,
                         h_FC_dim=h_FC_dim, drop_p=rnn_drop_p, n_output=last_lin_out, outfunc=outfunc).to(device)

crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

optimizer = optim.Adam(crnn_params, lr=learning_rate)
# criterion = ClusterLoss(try_n_clusters, KMeans).to(device)
criterion = nn.MSELoss()

val_losses = []
for n in range(epochs):
    sys.stdout.flush()
    print("Starting epoch %s" % (n + 1))
    cnn_encoder.train()
    rnn_decoder.train()
    for batchnum, (coords_data, targets) in enumerate(train_dataloader):
        coords_data = coords_data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = rnn_decoder(cnn_encoder(coords_data))

        loss = criterion(outputs, targets)
        print("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
        sys.stdout.flush()
        loss.backward()
        optimizer.step()

    total_val_loss = 0
    cnn_encoder.eval()
    rnn_decoder.eval()
    with torch.no_grad():
        for val_batchnum, (val_coords_data, val_targets) in enumerate(validation_dataloader):
            val_coords_data = val_coords_data.to(device)
            val_targets = val_targets.to(device)

            val_outputs = rnn_decoder(cnn_encoder(val_coords_data))

            batch_val_loss = F.mse_loss(val_outputs, val_targets, reduction='sum')
            total_val_loss += batch_val_loss.item()

    avg_val_loss = total_val_loss / len(validation_dataset)

    val_losses.append(avg_val_loss)
    print("Average MSE loss for epoch %s: %s" % (n + 1, avg_val_loss))
    if avg_val_loss > (val_losses[-2] if len(val_losses) >= 2 else 999):
        break

torch.save(cnn_encoder, 'pred_cnn_enc.mdl')
torch.save(rnn_decoder, 'pred_rnn_dec.mdl')

#
# plt.figure(1)
# plt.plot(range(1, epochs + 1), val_losses, label='Training Set')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.xticks(range(0, epochs + 1))
# plt.savefig(plot_file)
#
# try:
#     plt.show()
# except:
#     logger.info("Unable to show plot")
