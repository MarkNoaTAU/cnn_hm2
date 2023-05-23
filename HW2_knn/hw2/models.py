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
        if len(hidden_features) == 0:
            blocks.append(Linear(in_features, num_classes))
           
        else:
            blocks.append(Linear(in_features, hidden_features[0]))
            
            for i in range(len(hidden_features)-1):
                if activation == 'relu':
                    blocks.append(ReLU())
                else:
                    blocks.append(Sigmoid())
                if(dropout> 0):
                    blocks.append(Dropout(dropout))
                blocks.append(Linear(hidden_features[i], hidden_features[i+1]))
            
            if activation == 'relu':
                blocks.append(ReLU())
            else:
                blocks.append(Sigmoid())
            if(dropout> 0):
                blocks.append(Dropout(dropout))
            
            blocks.append(Linear(hidden_features[-1], num_classes))
                    
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
        self.h = in_h
        self.w = in_w
        
        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        C = self.in_size[0]
        filters = self.filters
        filters.insert(0,C)
        k = 1
        
        padding = (0, 0)
        kernel_size = (2, 2)
        stride = (2, 2)
        
        stride_conv = (1,1)
        kernel_size_conv = (3,3)
        padding_conv = (1,1)
        
        assert (N / self.pool_every) == (N // self.pool_every)
        for i in range(N//P):
            for j in range(P):
                layers.append(torch.nn.Conv2d(filters[k-1], filters[k], stride=stride_conv, padding=padding_conv, kernel_size=kernel_size_conv))
                
                self.h = (((self.h + 2 * padding_conv[0] - (kernel_size_conv[0] - 1) - 1) // stride_conv[0]) + 1)
                self.w = (((self.w + 2 * padding_conv[1] - (kernel_size_conv[1] - 1) - 1) // stride_conv[1]) + 1)
                
                layers.append(nn.ReLU(inplace = True))
                k += 1
            layers.append(torch.nn.MaxPool2d(kernel_size = kernel_size, stride=stride, padding=padding))
            self.h = (((self.h + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0]) + 1)
            self.w = (((self.w + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1]) + 1)
            
        filters.pop(0)

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Flatten())
        N = len(self.filters)
        P = self.pool_every
        M = len(self.hidden_dims)
        C = self.in_size[0]
        
        
        hidden_dims = self.hidden_dims
        hidden_dims.insert(0,int(self.filters[-1] * self.h * self.w))
        
        for i in range(M):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU(inplace = True))
        
        layers.append(nn.Linear(hidden_dims[M],self.out_classes))
        
        hidden_dims.pop(0)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        in_channels, in_h, in_w, = tuple(self.in_size)
        
        N = len(self.filters)
        P = self.pool_every
        C = self.in_size[0]
        
            
        x = self.feature_extractor(x)
        out = self.classifier(x)
    
        # ========================
        return out

class ResNetResidualBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs): 
        
        # filters is [in_channels, filter, out_channels ]
         # or
        #  [in_channels, out_channels] 
             
        super().__init__()
        self.filters = filters
        self.should_apply_shortcut = self.filters[0] != self.filters[-1]
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.filters[0], self.filters[-1], kernel_size=1, stride=(1,1)), 
            nn.BatchNorm2d(self.filters[-1], track_running_stats=False)
        ) if self.should_apply_shortcut else None 
        
        stride_conv = (1,1)
        kernel_size_conv = (3,3)
        padding_conv = (1,1)
        
        layers = []
        
        for i in range(len(self.filters)-1): 
        
            ## only one ReLU
            if i!=0:   
                layers.append(nn.ReLU(inplace = True))
                
            ## add first conv
            layers.append(torch.nn.Conv2d(self.filters[i], self.filters[i+1], stride=stride_conv, padding=padding_conv, kernel_size=kernel_size_conv))

            ##first BN
            layers.append(nn.BatchNorm2d(self.filters[i+1]))

                                  
        self.block_seq = nn.Sequential(*layers)
        

                          
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.block_seq(x)
        x += residual
        x =   F.relu(x) #nn.ReLU(x)
        return x
    



class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

                    
    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
        
    def update_dims(self, h,w, padding, kernel_size, stride):
        h_ret = (((h + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0]) + 1)
        w_ret = (((w + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1]) + 1)
        return h_ret, w_ret

     
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        self.h = in_h
        self.w = in_w
        
        layers = []
        
        N = len(self.filters)
        P = self.pool_every
        C = self.in_size[0]
        filters = self.filters
        filters.insert(0,C)
        
        padding = (0, 0)
        kernel_size = (2, 2)
        stride = (2, 2)
        k = 0
        
        stride_conv = (1,1)
        kernel_size_conv = (3,3)
        padding_conv = (1,1)
        
        assert (N / self.pool_every) == (N // self.pool_every)
        for i in range(N//P):
            for j in range(0,P-1,2):  
                layers.append(ResNetResidualBlock(filters[k:k+3]))
                self.h, self.w = self.update_dims(self.h,self.w, padding_conv, kernel_size_conv, stride_conv)
                self.h, self.w = self.update_dims(self.h,self.w, padding_conv, kernel_size_conv, stride_conv)

                k += 3
                
                
            if P % 2 == 1:   ## we need to add one more block with only one conv
                layers.append(ResNetResidualBlock(filters[k:k+2]))
                self.h, self.w = update_dims(self.h,self.w, padding_conv, kernel_size_conv, stride_conv)
                k += 2
                              
            layers.append(torch.nn.MaxPool2d(kernel_size = kernel_size, stride=stride, padding=padding))
            self.h = (((self.h + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0]) + 1)
            self.w = (((self.w + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1]) + 1)

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # =======================