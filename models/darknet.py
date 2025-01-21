import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot


class DarknetBlock(nn.Module):
    """
    A Single Darknet Block with two Convolutional Layers
    """
    def __init__(self, in_channels, out_channels):
        super(DarknetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # padding=1 for same padding
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1) # Use with each bn1, bn2 layer but use One for easy to change it.
    
    def forward(self, x):
        residual = x # Step 1: Save the input (residual connection).
        x = self.relu(self.bn1(self.conv1(x))) # Step 2: Apply Conv1, BatchNorm1, and LeakyReLU.
        x = self.relu(self.bn2(self.conv2(x))) # Step 3: Apply Conv2, BatchNorm2, and LeakyReLU.
        return x + residual # Step 4: Add the residual connection to the processed output.
    
class Darknet53(nn.Module):
    """
    A Darknet-53 Model
    """
    def __init__(self, num_classes=80):
        super(Darknet53, self).__init__() 
        self.layers = self._make_layers()
        
    def _make_layers(self):
        layers = []
        in_channels = 3 # RGB Image
        layer_config = [
            (64, 1),
            (128, 2),
            (256, 8),
            (512, 8),
            (1024, 4)
        ]

        # First Convolutional Layer
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(32)) 
        layers.append(nn.LeakyReLU(0.1))
        in_channels = 32 # Update in_channels for next layers.
        
        for out_channels, num_blocks in layer_config:
            # Down sample (Stride = 2)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.1))
            
            # Add Darknet Blocks
            for _ in range (num_blocks):
                layers.append(DarknetBlock(out_channels, out_channels))
            
            in_channels = out_channels # Update in_channels for next layers.
            
        return nn.Sequential(*layers) # Unpack the list of layers and return as a Sequential model.
    
    def forward(self, x):
        return self.layers(x)
    

# Test the Darknet53 Model
if __name__ == "__main__":
    model = Darknet53()
    x = torch.randn(1, 3, 416, 416) # Batch size 1, 3 channels, 416x416 image size
    output = model(x)
    y = model(x)
    print(summary(model, input_size=(3, 416, 416)))
    print(output.shape)  # torch.Size([1, 1024, 13, 13])
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render("/home/vasu/Documents/yoloV3/darknet53_structure", format="png")