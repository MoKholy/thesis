import torch 
import torch.nn as nn

class DeepLDA(nn.Module):
    
    def __init__(self, n_hidden: list , input_size, out_size, dropout=0.5):
        super(DeepLDA, self).__init__()
        self.out_size = out_size
        self.layers = nn.Sequential()
        for i in range(len(n_hidden)):
            self.layers.add_module(f"FC {i}", nn.Linear(input_size, n_hidden[i], bias=False))
            self.layers.add_module(f"BatchNorm {i}", nn.BatchNorm1d(n_hidden[i]))
            self.layers.add_module(f"ReLU {i}", nn.ReLU())
            self.layers.add_module(f"Dropout {i}", nn.Dropout(dropout))
            input_size = n_hidden[i]
        self.layers.add_module("FC out", nn.Linear(input_size, self.out_size))

        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance (m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance (m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            


def test():
    hidden = [64, 128, 256, 128, 64]
    model = DeepLDA(hidden, 40, 14, 0.3)
    print(model)
    # input size = 40, output size = 2

    # check if cuda 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # create random input with normal distribution
    x = torch.randn(128, 40).to(device)
    # create random target
    y = torch.randint(0, 15, (128,)).to(device)
    # forward pass
    output = model(x)
    # print loss dimension
    print(output.shape)

if __name__ == "__main__":
    test()