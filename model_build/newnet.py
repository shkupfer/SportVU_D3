import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from model_build.nets import EncoderCNN, DecoderRNN
import os

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU

pklfiles_dir = '/dbdata/new_processed_possessions'
plot_file = 'test_loss.png'

validation_prop = 0.3
batch_size = 100
epochs = 20
n_imgs = 150
img_x = 48
img_y = 51

CNN_fc_hidden1 = int(512 / 2)
CNN_fc_hidden2 = int(512 / 2)
# dropout_p = .1
dropout_p = 0.1
CNN_embed_dim = int(256 / 2)
RNN_hidden_layers = 2
RNN_hidden_nodes = int(256 / 2)
RNN_FC_dim = int(128 / 2)

learning_rate = .01

all_pklfiles = os.listdir(pklfiles_dir)
train_pklfiles, validation_pklfiles = train_test_split(all_pklfiles, test_size=validation_prop, random_state=0)

print("About to initialize PossessionsDatasets")
train_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in train_pklfiles])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

validation_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in validation_pklfiles])
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


if __name__ == "__main__":
    cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                             drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p).to(device)

    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    optimizer = optim.Adam(list(cnn_encoder.parameters()) + list(rnn_decoder.parameters()), lr=learning_rate)
    # optimizer = optim.SGD(list(cnn_encoder.parameters()) + list(rnn_decoder.parameters()), lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    val_losses = []
    for n in range(epochs):
        print("Starting epoch %s" % (n + 1))
        for batchnum, (coords_data, poss_data, targets) in enumerate(train_dataloader):
            cnn_encoder.train()
            rnn_decoder.train()

            coords_data = coords_data.to(device)
            poss_data = poss_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = rnn_decoder(cnn_encoder(coords_data), poss_data)

            loss = criterion(outputs, targets)
            print("Batch %s MSE loss: %s" % (batchnum + 1, str(loss.item())))
            loss.backward()
            optimizer.step()

        all_test_targets = []
        all_test_outputs = []
        summed_val_loss = 0
        with torch.no_grad():
            for val_coords_data, val_poss_data, val_targets in validation_dataloader:
                cnn_encoder.eval()
                rnn_decoder.eval()

                val_coords_data = val_coords_data.to(device)
                val_poss_data = val_poss_data.to(device)
                val_targets = val_targets.to(device)

                val_outputs = rnn_decoder(cnn_encoder(val_coords_data), val_poss_data)

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

    # torch.save(model.state_dict(), 'model.mdl')
