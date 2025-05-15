import torch
from torch import nn
import torch.nn.functional as F
from kan import KANLayer
from kan import KAN

class ChannelWiseConv(nn.Module):
    def __init__(self,grid_size: int = 5, in_channels: int = 13, num_classes: int = 10, device='cpu'):
        """Initialize the KANClassification model.
        Args:
            grid_size (int): Size of the grid for KAN convolutional layers.
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super().__init__()

        self.channel_wise_encoding = []
        for _ in range(in_channels):
            this_channel_convs = [
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1), 
                nn.ReLU(),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(), 
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
                # Final shape: [B, 13, 1, 1]
            ]
            self.channel_wise_encoding.append(
                nn.Sequential(*this_channel_convs).to(device)
            )

        # last KAN linear layer
        self.kan_bands_aggregator = KAN(
            width=[in_channels, 20, num_classes], 
            grid=grid_size, 
            k=3, 
            seed=42, 
            device=device
        )

        self.name = 'ChannelWiseConv'


    def forward(self, x):
        """Forward pass of the KANClassification model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        channel_wise_outputs = []
        for i in range(x.shape[1]):
            channel_output = self.channel_wise_encoding[i](x[:, i:i+1, :, :])
            channel_wise_outputs.append(channel_output)
        x = torch.cat(channel_wise_outputs, dim=1)
        x = x.view(x.size(0), -1)  # Flatten [B, C, 1, 1] to [B, C]
        x = self.kan_bands_aggregator(x)
        x = F.log_softmax(x, dim=1)
        return x

