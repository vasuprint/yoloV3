import torch
import torch.nn as nn
from models.darknet import Darknet53

class YOLOv3(nn.Module):
    def __init__(self, num_classes): 
        super(YOLOv3, self).__init__() # Call the __init__ method of the parent class nn.Module.
        self.backbone = Darknet53()
        self.num_class = num_classes
        
        # Init Output Convolutional Layers for three scales
        self.output_conv_small = self._make_output_layer(1024) # For the small feature map 13x13 For detect large Object. 
        self.output_conv_medium = self._make_output_layer(512) # For the medium feature map 26x26 For detect medium Object.
        self.output_conv_large = self._make_output_layer(256) # For the large feature map 52x52 For detect small Object.
        
        # Upsampling layer for feature fusion
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest') # Upsample the small and medium feature map.
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest') # Upsample the medium and large feature map.
        
        # Convolution layers for feature fusion
        self.conv1 = self._make_convolution_layer(512, 256) # For the medium feature map 26x26
        self.conv2 = self._make_convolution_layer(256, 128) # For the large feature map 52x52
        
    def forward(self, x):
        # Extract feature maps from the backbone53
        feature_large, feature_medium, feature_small = self.backbone(x)
        
        