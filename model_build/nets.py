# Some of these network classes copied from: https://github.com/HHTseng/video-classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score
from torch.autograd import Variable
from math import floor
from torchvision.models.resnet import conv3x3
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

poss_data_len = 7

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, in_chans=2, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        # self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 32, 64, 64
        # self.k1, self.k2, self.k3, self.k4 = (4, 4), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        # self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.k1, self.k2, self.k3, self.k4 = (3, 3), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        # self.conv1_outshape = (floor((self.conv1_outshape[0] - 2) / 2) + 1, floor((self.conv1_outshape[1] - 2) / 2) + 1)
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        # self.conv2_outshape = (floor((self.conv2_outshape[0] - 2) / 2) + 1, floor((self.conv2_outshape[1] - 2) / 2) + 1)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        # self.conv3_outshape = (floor((self.conv3_outshape[0] - 2) / 2) + 1, floor((self.conv3_outshape[1] - 2) / 2) + 1)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
        # print(self.conv1_outshape, self.conv2_outshape, self.conv3_outshape)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        # self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        # self.fc1 = nn.Linear(self.ch3 * self.conv3_outshape[0] * self.conv3_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        # self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.CNN_embed_dim)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class MyLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, drop_p, linear_hidden, linear_outputs, outfunc=None, bidirectional=False):
        super(MyLSTM, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=drop_p,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc1 = nn.Linear(2 * hidden_size if bidirectional else hidden_size, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, linear_outputs)
        self.outfunc = nn.Identity() if outfunc is None else outfunc

    def forward(self, x):
        lstm_out, _ = self.LSTM(x)
        x = lstm_out[:, -1, :]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.outfunc(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, n_output=32, outfunc=nn.LogSoftmax):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.n_output = n_output

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.n_output)
        # self.fc1 = nn.Linear(2 * self.h_RNN, self.n_output)
        self.outfunc = outfunc

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        # x = RNN_out[:, -1, :]  # choose RNN_out at the last time step
        x = torch.cat([RNN_out[:, -1, :self.h_RNN], RNN_out[:, 0, self.h_RNN:]], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        x = self.outfunc(x)

        return x


class VaryingLengthDecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, n_output=32, outfunc=nn.LogSoftmax):
        super(VaryingLengthDecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.n_output = n_output

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.n_output)
        # self.fc1 = nn.Linear(2 * self.h_RNN, self.n_output)
        self.outfunc = outfunc

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x, _ = nn.utils.rnn.pad_packed_sequence(RNN_out, batch_first=True)
        x = x[:, -1, :]  # choose RNN_out at the last time step
        # x = torch.cat([RNN_out[:, -1, :self.h_RNN], RNN_out[:, 0, self.h_RNN:]], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        x = self.outfunc(x)

        return x

class ClusterLoss(nn.Module):
    def __init__(self, try_n_clusters, clustering_algo=[KMeans], out_device=torch.device("cpu"), scale=False, **kwargs):
        super(ClusterLoss, self).__init__()
        self.try_n_clusters = try_n_clusters
        self.clustering_algo = clustering_algo
        self.out_device = out_device
        self.scale = scale
        self.kwargs = kwargs

    def forward(self, img_repr, is_val=False):
        img_repr = img_repr.detach().cpu().numpy()
        if self.scale:
            scaler = StandardScaler()
            img_repr = scaler.fit_transform(img_repr)
        best_sil_score = -1
        best_algo = self.clustering_algo[0]
        best_n_clusters = self.try_n_clusters[0]
        for algo in self.clustering_algo:
            if 'n_clusters' in algo._get_param_names():
                for n_clusters in self.try_n_clusters:
                    clusterer = algo(n_clusters=n_clusters)
                    cluster_labels = clusterer.fit_predict(img_repr)
                    sil_score = silhouette_score(img_repr, cluster_labels)
                    # sil_score = clusterer.inertia_
                    if sil_score > best_sil_score:
                        best_sil_score = sil_score
                        best_n_clusters = n_clusters
                        best_algo = algo

            else:
                clusterer = algo(**self.kwargs)
                cluster_labels = clusterer.fit_predict(img_repr)
                sil_score = silhouette_score(img_repr, cluster_labels)
                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_algo = algo

        print("Best silhouette score: %s, with algo: %s and %s clusters" % (best_sil_score, best_algo.__name__, best_n_clusters))

        return Variable(torch.Tensor([1 - best_sil_score]), requires_grad=False if is_val else True)


def cnn_shape_calc(img_size_in, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0)):
    outs = []
    for size, ksize, strd, pad in zip(img_size_in, kernel_size, stride, padding):
        out = floor((size + 2 * pad - ksize) / strd) + 1
        outs.append(out)
    return tuple(outs)


class CNN3D(nn.Module):
    def __init__(self, layers_dict, in_size):
        super(CNN3D, self).__init__()
        conv1_dict, pool1_dict, conv2_dict, pool2_dict = layers_dict['conv1'], layers_dict['pool1'], layers_dict['conv2'], layers_dict['pool2']

        self.conv1 = nn.Conv3d(layers_dict['in_channels'], conv1_dict['n_kernels'], kernel_size=conv1_dict['kernel_size'], stride=conv1_dict['stride'], padding=conv1_dict['padding'])
        size_tracker = cnn_shape_calc(in_size, conv1_dict['kernel_size'], conv1_dict['stride'], conv1_dict['padding'])
        self.bn1 = nn.BatchNorm3d(conv1_dict['n_kernels'])

        if pool1_dict['kernel_size'] is not None:
            self.pool1 = pool1_dict['pool_type'](pool1_dict['kernel_size'])
            size_tracker = cnn_shape_calc(size_tracker, pool1_dict['kernel_size'], pool1_dict['kernel_size'])
        else:
            self.pool1 = nn.Identity()

        self.conv2 = nn.Conv3d(conv1_dict['n_kernels'], conv2_dict['n_kernels'], kernel_size=conv2_dict['kernel_size'], stride=conv2_dict['stride'], padding=conv2_dict['padding'])
        size_tracker = cnn_shape_calc(size_tracker, conv2_dict['kernel_size'], conv2_dict['stride'], conv2_dict['padding'])
        self.bn2 = nn.BatchNorm3d(conv2_dict['n_kernels'])

        if pool2_dict['kernel_size'] is not None:
            self.pool2 = pool2_dict['pool_type'](pool2_dict['kernel_size'])
            size_tracker = cnn_shape_calc(size_tracker, pool2_dict['kernel_size'], pool2_dict['kernel_size'])
        else:
            self.pool2 = nn.Identity()

        self.output_func = layers_dict.get('output_func', nn.Identity)()

        # print("Shape before linear layer should be: %s" % str(size_tracker))

        lin_infeats = conv2_dict['n_kernels']
        for dim in size_tracker:
            lin_infeats *= dim

        self.fc1 = nn.Linear(lin_infeats, layers_dict['out_features'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.output_func(x)

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
        # x = torch.cat((x, poss_data), dim=1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = F.hardtanh(x, 0, 4)
        x = F.log_softmax(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, n_chans=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_chans, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(4 * 512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, lin_1_in=None):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # resnet = models.resnet152(pretrained=True)
        resnet = ResNet(BasicBlock, [2, 2, 2, 2])  # resnet18
        # resnet = ResNet(BasicBlock, [3, 4, 6, 3])  # resnet34
        # resnet = ResNet(Bottleneck, [3, 4, 6, 3])  # resnet50
        # resnet = ResNet(Bottleneck, [3, 8, 36, 3])  # resnet152
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        if not lin_1_in:
            lin_1_in = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(lin_1_in, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                # print(x_3d[:, t, :, :, :].size())
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq