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
    def __init__(self):
        super(Darknet53, self).__init__() 
        self.stages = self._make_layers()
        
    def _make_layers(self):
        stages = []
        in_channels = 3  # RGB Image
        
        # Layer configuration for Darknet53
        layer_config = [
            (64, 1),
            (128, 2),
            (256, 8),
            (512, 8),
            (1024, 4)
        ]

        # First Convolutional Layer
        stages.append(nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        ))
        in_channels = 32  # Update in_channels for next layers.
        
        for out_channels, num_blocks in layer_config:
            stage = []
            # Down sample (Stride=2)
            stage.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            stage.append(nn.BatchNorm2d(out_channels))
            stage.append(nn.LeakyReLU(0.1))
            
            # Add Darknet Blocks
            for _ in range(num_blocks):
                stage.append(DarknetBlock(out_channels, out_channels))
            
            stages.append(nn.Sequential(*stage)) # Flatten the list and add to stages
            in_channels = out_channels  # Update in_channels for next layers.
            
        return nn.ModuleList(stages)  # Convert list to ModuleList
    
    def forward(self, x):
        features_small = None
        features_medium = None
        features_large = None

        for idx, stage in enumerate(self.stages):
            #print(f"Stage {idx}: {stage}")
            x = stage(x)
            if idx == 3:  # Small feature map (52 x 52 x 256)
                features_small = x
            elif idx == 4:  # Medium feature map (26 x 26 x 512)
                features_medium = x
            elif idx == 5:  # Large feature map (13 x 13 x 1024)
                features_large = x

        return features_large, features_medium, features_small
        


# Test the Darknet53 Model
if __name__ == "__main__":
    model = Darknet53()
    
    # Test with 416x416 input
    x_416 = torch.randn(1, 3, 416, 416) # Batch size 1, 3 channels, 416x416 image size
    features_large, features_medium, features_small = model(x_416)
    print("For 416 x 416 input:")
    print("Small Feature Map Shape:", features_small.shape)   # Expected: (1, 256, 52, 52)
    print("Medium Feature Map Shape:", features_medium.shape) # Expected: (1, 512, 26, 26)
    print("Large Feature Map Shape:", features_large.shape)   # Expected: (1, 1024, 13, 13)

    # Test with 640x640 input
    x_640 = torch.randn(1, 3, 640, 640) # Batch size 1, 3 channels, 640x640 image size
    features_large, features_medium, features_small = model(x_640)
    print("\nFor 640 x 640 input:")
    print("Small Feature Map Shape:", features_small.shape)   # Expected: (1, 256, 80, 80)
    print("Medium Feature Map Shape:", features_medium.shape) # Expected: (1, 512, 40, 40)
    print("Large Feature Map Shape:", features_large.shape)   # Expected: (1, 1024, 20, 20)