from architectures.ArchitecturesUtils import *

class DisBasic(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=64):
        super(DisBasic, self).__init__()
        self.conv1 = self.basicBlock(in_channels, n_filters)
        self.conv2 = self.basicBlock(n_filters, n_filters)
        self.conv3 = self.basicBlock(n_filters, n_filters * 2)
        self.conv4 = self.basicBlock(n_filters * 2, n_filters * 4)
        self.fc1 = nn.Linear(15 * 15 * 256, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    @staticmethod
    def basicBlock(in_channels, features):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=features,
                                       kernel_size=3, stride=2, bias=False),
                             nn.BatchNorm2d(num_features=features),
                             nn.LeakyReLU(0.2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return self.fc2(self.leaky_relu(out))


if __name__ == '__main__':
    print("Test Dis")
    input_c = 1
    output_c = 1

    device = torch.device(("cpu", "cuda")[torch.cuda.is_available()])
    D = DisBasic(in_channels=input_c, out_channels=output_c)
    x = torch.zeros((6, input_c, 256, 256), dtype=torch.float32)
    print("input shape = ", x.shape)
    y = D(x.to(device))
    print("output shape = ", y.shape)