import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        pass
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        n = len(self.filters)
        for i in range(n):
            f = self.filters[i]
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = f

            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        # Note that we use dimension-preserving 3x3 convolutions, except the 2x2 Max pooling,
        # So in_h and in_w would be divided by 2 - pool_every times.
        number_of_pooling = int(len(self.filters) / self.pool_every)
        f_h, f_w = in_h / (2 ** number_of_pooling), in_w / (2 ** number_of_pooling)
        f_c = self.filters[-1]
        in_dim = int(f_h * f_w * f_c)

        for hid_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.ReLU())
            in_dim = hid_dim
        layers.append(nn.Linear(in_dim, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        cls_scores = self.classifier(features)
        # out = F.log_softmax(cls_scores, dim=1) # out = F.softmax(cls_scores, dim=1)

        # ========================
        return cls_scores


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, block_size):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv: Add stride and padding...
    # filter sizes etc.
    # ====== YOUR CODE: ======

    """
        We observed in the training of ConvClassifier the following main problem:
        1. For relatively shallow and large net, we saw over-fitting. 
            - Batch Norm
            - Dropout
        2. In dipper network we got the problem of vanishing gradient. 
            - Skip-connection
            - Batch Norm
        
        Each block size (between skip-connection) should be according to the size of K and
         the number of block will be set by L.
       
        # Note: I disabled the pooling currently.
        
    """
    def _make_feature_extractor(self):
        # will return a list of Residual-blocks on them, we will iterate.
        # Note there is no activation function (Relu) in the end of every block.

        # Lets assume pool_every and block_size are equal (if I would have time I would add pulling)
        b_size = self.pool_every
        in_channels, _, _, = tuple(self.in_size)
        blocks = []

        i = 0
        for b in range(int(len(self.filters) / 2)):
            # The block: (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm)
            layers = []
            for j in range(b_size):
                f = self.filters[i]
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(f))
                if j + 1 < b_size:
                    layers.append(nn.ReLU())
                in_channels = f
                i += 1
            blocks.append(nn.Sequential(*layers))

        def _extract_feature(x):
            out = x
            first_block = True
            for i_, func in enumerate(blocks):
                f_x = func(out)
                if first_block:
                    out = f_x
                    first_block = False
                else:
                    if f_x.shape != out.shape:
                        raise NotImplementedError("Error! Please select the hyperparameter of the model such "
                                                  "that the residual connection will be in the same size."
                                                  "In Pytorch, they do support non-equal dimension using 1x1 conv, "
                                                  "but we currently not. \n")
                    out = f_x + out
            return out

        return _extract_feature

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # (Linear -> ReLU -> Dropout)*M -> Linear
        # As this time I didn't implement pooling
        in_dim = in_h * in_w * self.filters[-1]

        for hid_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.Dropout(p=0.4))
            layers.append(nn.ReLU())
            in_dim = hid_dim
        layers.append(nn.Linear(in_dim, self.out_classes))

        seq = nn.Sequential(*layers)
        return seq
