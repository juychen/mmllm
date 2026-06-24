import torch
from flashfftconv import FlashFFTConv

def MyModel(torch.nn.Module):
    def __init__(self, H, seqlen, num_layers):
        super().__init__()

        self.H = H
        self.seqlen = seqlen
        self.num_layers = num_layers
        self.flashfftconv = FlashFFTConv(seqlen, dtype=torch.bfloat16)

        # create your conv layers
        self.long_conv_layers = torch.nn.ModuleList([
            ConvLayer(H, seqlen)
            for i in range(num_layers)
        ])

        # add a pointer to the flashfft object in each layer
        for layer in self.long_conv_layers:
            layer.flashfftconv = self.flashfftconv

        ...
    
    def forward(self, x):
        for layer in self.long_conv_layers:
            x = layer(x)

        return x

def ConvLayer(torch.nn.Module):
    def __init__(self, H, seqlen):
        self.k = torch.nn.Parameter(torch.randn(H, seqlen, dtype=torch.float32))
        ...

    def forward(self, x):
        return self.flashfftconv(x, self.k) # self.flashfftconv comes from the wrapper model!