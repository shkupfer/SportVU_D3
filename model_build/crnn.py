import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset, PossessionsDatasetTorch, PadSequence
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
# device = torch.device("cpu")

# pklfiles_dir = '/dbdata/new_processed_possessions'
# pklfiles_dir = '/dbdata/faded_new_possessions'
# pklfiles_dir = '/dbdata/playerchans_no_fade'
coords_data_files_dir = '/dbdata/pchans_max24_nofade_cd'
targets_files_dir = '/dbdata/pchans_max24_nofade_t'
plot_file = 'test_loss.png'
validation_prop = 0.3

batch_size = 20
epochs = 10
cnn_hidden_1_kerns = 20
# cnn_hidden_2_kerns = 32
cnn_out_kerns = 20
lstm_inputs = 200
lstm_hidden = 64

learning_rate = 0.001
dropout_p = 0.1

img_x = 48
img_y = 51
poss_data_len = 7
n_chans = 11


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_chans, cnn_hidden_1_kerns, kernel_size=3)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(cnn_hidden_1_kerns, cnn_out_kerns, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(cnn_hidden_2_kerns, cnn_out_kerns, kernel_size=3)
        # self.pool3 = nn.MaxPool2d(kernel_size=210 * 114 * 4, lstm_inputs)
        # self.fc2 = nn.Linear(720, 1)
        self.fc1 = nn.Linear(cnn_out_kerns * 10 * 11, lstm_inputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.dropout2d(x, dropout_p)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.dropout2d(x, dropout_p)
        x = F.relu(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = F.dropout2d(x, dropout_p)
        # x = F.relu(x)
        # x = self.pool3(x)

        x = x.view(-1, cnn_out_kerns * 10 * 11)
        x = self.fc1(x)
        x = F.relu(x)

        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        # self.cnns = [CNN().to(device) for _ in range(n_imgs)]
        self.cnn = CNN().to(device)
        # self.cnn = CNN()
        self.rnn = nn.LSTM(input_size=lstm_inputs,
                           hidden_size=lstm_hidden,
                           num_layers=1,
                           batch_first=True).to(device)
        self.linear = nn.Linear(lstm_hidden, 1).to(device)

    def forward(self, coords_data, seq_lengths):
        # x = torch.stack([self.cnns[timestep](coords_data[:, timestep, :, :, :]) for timestep in range(n_imgs)])

        x_list = []
        for timestep in range(coords_data.size(1)):
            x_list.append(self.cnn(coords_data[:, timestep, :, :, :]))
        x = torch.stack(x_list, dim=1)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)

        x, (h_n, h_c) = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x[:, -1, :]  # choose RNN_out at the last time step
        x = self.linear(x)
        x = F.hardtanh(x, 0, 4)

        return x


if __name__ == "__main__":
    cd_files = os.listdir(coords_data_files_dir)
    t_files = os.listdir(targets_files_dir)
    cd_train, cd_validation, t_train, t_validation = train_test_split(cd_files, t_files, test_size=validation_prop, random_state=0)

    sequence_padder = PadSequence(targets=True)
    print("About to initialize possessions datasets")
    train_dataset = PossessionsDatasetTorch([os.path.join(coords_data_files_dir, pfile_name) for pfile_name in cd_train],
                                            [os.path.join(targets_files_dir, pfile_name) for pfile_name in t_train],
                                            transform=None, targets=True, playerchans=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=sequence_padder)

    validation_dataset = PossessionsDatasetTorch([os.path.join(coords_data_files_dir, pfile_name) for pfile_name in cd_validation],
                                                 [os.path.join(targets_files_dir, pfile_name) for pfile_name in t_validation],
                                                 transform=None, targets=True, playerchans=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=sequence_padder)

    model = Combine()

    # params = []
    # for cnn in model.cnns:
    #     params.extend(list(cnn.parameters()))
    # params.extend(list(model.rnn.parameters()))
    # params.extend(list(model.linear.parameters()))
    params = list(model.cnn.parameters()) + list(model.rnn.parameters()) + list(model.linear.parameters())

    optimizer = optim.Adam(params, lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    val_losses = []
    for n in range(epochs):
        print("Starting epoch %s" % (n + 1))
        model.train()
        for batchnum, (coords_data, seq_lengths, targets) in enumerate(train_dataloader):
            coords_data = coords_data.to(device)
            seq_lengths = seq_lengths.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(coords_data, seq_lengths)

            loss = criterion(outputs, targets)
            print("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
            loss.backward()
            optimizer.step()

        all_test_targets = []
        all_test_outputs = []
        summed_val_loss = 0
        model.eval()
        with torch.no_grad():
            for val_coords_data, val_targets in validation_dataloader:
                val_coords_data = val_coords_data.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_coords_data)

                batch_test_loss = F.mse_loss(val_outputs, val_targets, reduction='sum').item()
                summed_val_loss += batch_test_loss

                all_test_targets.extend(val_targets)
                all_test_outputs.extend(val_outputs)

        avg_val_loss = summed_val_loss / len(validation_dataset)
        val_losses.append(avg_val_loss)

        print("Validation MSE loss for epoch %s: %s" % (n + 1, avg_val_loss))

    torch.save(model.state_dict(), '/home/ubuntu/SportVU_D3/amodel.mdl')

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
