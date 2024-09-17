from torch import nn

class SmallNetwork(nn.Module):
    """
    Small CNN architecture for MNIST (1x28x28 images) inspired by 
    
    Simard, P. Y., Steinkraus, D., and Platt, J. C. Best Practices for Convolutional Neural Networks Applied to Visual
    Document Analysis. In Proceedings of the Seventh International Conference on Document Analysis and Recognition,
    volume 3. IEEE Computer Society, 2003.

    Code taken with permission from 

    Lindström, M., Rodríguez-Gálvez, B., Thobaben, R., and Skoglund, M., 
    A Coding-Theoretic Analysis of Hyperspherical Prototypical Learning Geometry
    To appear in the Proceedings of Machine Learning Research (PMLR), vol 251.
    Code at: https://github.com/martinlindstrom/coding_theoretic_hpl 
    """
    def __init__(self, out_dim):
        super(SmallNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=50, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=50*5*5, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=out_dim),
        )
    def forward(self, x):
        out = self.net(x)
        return out