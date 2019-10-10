import sys
import torch
import torch.nn as nn
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from model_build.build_utils import PossessionsDataset, OnLoadPossessionsDataset
import os
from sklearn.model_selection import train_test_split
from model_build.nets import ClusterLoss, CNN3D
from sklearn.cluster import KMeans
from itertools import product
from random import shuffle, choice
# import gc
import traceback

batch_size = 300

pklfiles_dir = '/dbdata/new_processed_possessions'
validation_prop = 0.3
all_pklfiles = os.listdir(pklfiles_dir)
train_pklfiles, validation_pklfiles = train_test_split(all_pklfiles, test_size=validation_prop, random_state=0)
print("About to initialize PossessionsDatasets")
train_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in train_pklfiles], rearrange=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

validation_dataset = OnLoadPossessionsDataset([os.path.join(pklfiles_dir, pfile_name) for pfile_name in validation_pklfiles], rearrange=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
sys.stdout.flush()

# @profile
def run_network(big_params_dct, plotname):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
    # device = torch.device("cpu")

    # in_channels = 2
    img_size_in = (150, 48, 51)

    epochs = 5
    try_n_clusters = [10, 12, 15, 18, 22, 26]

    model = CNN3D(big_params_dct, img_size_in).to(device)

    params = model.parameters()

    optimizer = optim.Adam(params, lr=big_params_dct['learning_rate'])
    criterion = ClusterLoss(try_n_clusters, big_params_dct['clustering_algo']).to(device)

    val_losses = []
    for n in range(epochs):
        sys.stdout.flush()
        print("Starting epoch %s" % (n + 1))
        model.train()
        for batchnum, coords_data in enumerate(train_dataloader):
            coords_data = coords_data.to(device)

            optimizer.zero_grad()

            loss = criterion(model(coords_data))
            print("Batch %s loss: %s" % (batchnum + 1, str(loss.item())))
            loss.backward()
            optimizer.step()

        summed_val_loss = 0
        model.eval()
        with torch.no_grad():
            for val_batchnum, val_coords_data in enumerate(validation_dataloader):
                val_coords_data = val_coords_data.to(device)

                batch_test_loss = criterion(model(val_coords_data), is_val=True).item()
                summed_val_loss += batch_test_loss

        avg_val_loss = summed_val_loss / (val_batchnum + 1)
        val_losses.append(avg_val_loss)
        print("Validation MSE loss for epoch %s: %s" % (n + 1, avg_val_loss))
        if avg_val_loss > (val_losses[-2] if len(val_losses) >= 2 else 999):
            break

    # plt.figure(1)
    # plot_x = list(range(len(val_losses)))
    # plt.plot(plot_x, val_losses, label='Training Set')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE Loss')
    # plt.xticks(plot_x)
    # plt.savefig("%s.png" % plotname)

    # try:
    #     plt.show()
    # except:
    #     print("Unable to show plot")

    return min(val_losses)


orig_params_dct = {'in_channels': 2,
                   'conv1': {'n_kernels': 24,
                             'kernel_size': (6, 4, 4),
                             'stride': (2, 2, 2),
                             'padding': (0, 0, 0)
                             },
                   'pool1': {'kernel_size': (2, 2, 2)
                             },
                   'conv2': {'n_kernels': 48,
                             'kernel_size': (4, 4, 4),
                             'stride': (1, 1, 1),
                             'padding': (0, 0, 0)
                             },
                   'pool2': {'kernel_size': (2, 2, 2)
                             },
                   'out_features': 128,
                   'learning_rate': 0.001,
                   'clustering_algo': KMeans
                   }

def run_stuff():
    conv1_kernels = [64]
    conv1_ksizes = [(4, 4, 4), (8, 4, 4), (16, 8, 8)]
    conv1_strides = [(2, 1, 1), (2, 2, 2)]

    conv2_kernels = [32, 64, 128]
    conv2_ksizes = [(2, 2, 2), (4, 2, 2), (4, 4, 4)]
    conv2_strides = [(2, 1, 1), (2, 2, 2)]

    pool1_ksizes = [(2, 2, 2)]
    pool1_types = [nn.AvgPool3d, nn.MaxPool3d]

    pool2_ksizes = [(2, 2, 2)]
    pool2_types = [nn.MaxPool3d]

    out_features = [64]

    learning_rates = [0.01, 0.001]

    # clustering_algos = [KMeans, SpectralClustering, AgglomerativeClustering]
    clustering_algos = [KMeans]

    output_funcs = [nn.LogSoftmax]

    combos = list(product(conv1_kernels, conv1_ksizes, conv1_strides, conv2_kernels, conv2_ksizes, conv2_strides,
                          pool1_ksizes, pool1_types, pool2_ksizes, pool2_types,
                          out_features, learning_rates, clustering_algos, output_funcs))

    # shuffle(combos)
    combos = [choice(combos)]

    for run_num, (c1_kerns, c1_ksize, c1_stride, c2_kerns, c2_ksize, c2_stride, p1_ksize, p1_type, p2_ksize, p2_type, out_feats, lr, clusterer, outfunc) in enumerate(combos):
        sys.stdout.flush()
        params_dict = orig_params_dct.copy()

        params_dict['conv1'].update({'n_kernels': c1_kerns,
                                     'kernel_size': c1_ksize,
                                     'stride': c1_stride})

        params_dict['conv2'].update({'n_kernels': c2_kerns,
                                     'kernel_size': c2_ksize,
                                     'stride': c2_stride})

        params_dict['pool1'].update({'kernel_size': p1_ksize,
                                     'pool_type': p1_type})

        params_dict['pool2'].update({'kernel_size': p2_ksize,
                                     'pool_type': p2_type})

        params_dict['out_features'] = out_feats
        params_dict['learning_rate'] = .1
        params_dict['clustering_algo'] = clusterer
        params_dict['output_func'] = outfunc

        print("RUN #%s, PARAMS: %s" % (run_num, str(params_dict)))

        min_loss = run_network(params_dict, 'loss_plot_%s' % run_num)
        print("MIN LOSS: %s" % min_loss)

        # try:
        #     min_loss = run_network(params_dict, 'loss_plot_%s' % run_num)
        #     print("RUN #%s, PARAMS: %s. MIN LOSS: %s" % (run_num, str(params_dict), min_loss))
        # except (RuntimeError, OSError) as exc:
        #     # errcounter += 1
        #     exc_type, exc_value, exc_traceback = sys.exc_info()
        #     print("RUN #%s, PARAMS: %s. ERRORED: %s" % (run_num, str(params_dict), traceback.format_tb(exc_traceback)))


if __name__ == "__main__":
    run_stuff()
