import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering, DBSCAN
from model_build.build_utils import ExtractedFeatures
from model_build.nets import MyLSTM, ClusterLoss
from itertools import product
import random

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
print(device)

datafile = '/dbdata/all_good_out.npy'
n_moments = 30
input_features = 57
batch_size = 500

dataset = ExtractedFeatures(datafile, n_moments, targetsfile='/home/ubuntu/SportVU_D3/targets.txt')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

epochs = 10
# try_n_clusters = range(4, 16 + 1)
try_n_clusters = range(8, 16)

learning_rate = 0.001
hidden_size = 128
num_layers = 3
drop_p = 0.0
linear_hidden = 128
linear_outputs = 1
outfunc = nn.Identity()
# lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hsizes = [4, 16, 64, 128]
# num_layers = [1, 2, 3]
# drop_ps = [0.0, 0.2]
l_hiddens = hsizes
# l_outputs = [1, 4, 16, 64]
# outfuncs = [nn.Hardtanh(), nn.LogSigmoid(), nn.LogSoftmax(),
#             nn.ReLU(), nn.Sigmoid()]
# , nn.Softplus()
# optimizers = optim.Adam, optim.Adagrad
# combos = list(product(lrs, drop_ps, l_outputs, outfuncs, optimizers))
# random.shuffle(combos)
# for combo in combos:
# learning_rate, drop_p, linear_outputs, outfunc, optimizer = combo
model = MyLSTM(input_features=input_features, hidden_size=hidden_size, num_layers=num_layers,
               drop_p=drop_p, linear_hidden=linear_hidden, linear_outputs=linear_outputs, outfunc=outfunc).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optimizer(model.parameters(), lr=learning_rate)
# criterion = ClusterLoss(try_n_clusters=try_n_clusters, clustering_algo=[KMeans], scale=True)

# print(model, optimizer)
# print(combo)
criterion = nn.MSELoss()
for n in range(epochs):
    # logger.info("Starting epoch %s" % (n + 1))
    sys.stdout.flush()

    model.train()

    for batchnum, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, targets)
        logger.info("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
        sys.stdout.flush()

        loss.backward()
        optimizer.step()
    # print("Epoch %s last batch loss: %s" % (n, loss.item()))
import ipdb;ipdb.set_trace()