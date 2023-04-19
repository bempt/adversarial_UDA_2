"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
import torchvision

import modular.segmentation.utils_segmentation as utils_segmentation
from modular.segmentation import data_setup_segmentation as seg_data
import modular.utils

device = modular.utils.set_device()

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion


class Discriminator(nn.Module):
    def __init__(self, in_channels=seg_data.N_CLASSES, out_channels=1, num_filters=64):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_filters * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_filters * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=num_filters * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters * 8, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(out)
        return out.view(-1, 1)

def create_discriminator():
    # 1. Get the base model with pretrained weights and send to target device
    model = Discriminator().to(device)

    # 2. Set the seeds
    modular.utils.set_seeds()

    # 3. Give the model a name
    model.name = "discriminator"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an EffNetB0 feature extractor
def create_effnetb0(out_features):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    utils_segmentation.set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_features)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an EffNetB2 feature extractor
def create_effnetb2(out_features):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    utils_segmentation.set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=out_features)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model