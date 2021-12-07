import torch
import torch.nn as nn
from torchvision import transforms as tfs
from PIL import Image
from PIL.ImageOps import invert


# CNN for MNIST problem
class MnistCnn(nn.Module):
    def __init__(self):
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        LIN_IN = 5 * 5 * 16
        self.lin = nn.Sequential(
            nn.Linear(LIN_IN, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.out = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(-1, 5 * 5 * 16)
        
        x = self.lin(x)
        logits = self.out(x)
        
        return logits


# a class for linking MnistCnn and bot
class PredictiveModel:
    def __init__(self, saved_model_path):
        # loading and setting up model
        self.model = MnistCnn() 
        checkpoint = torch.load(saved_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        # transforms for input images
        self.trans = tfs.Compose([
            tfs.PILToTensor(),
            tfs.Resize(28),
            tfs.ConvertImageDtype(torch.float),
            tfs.Normalize((0.5), (0.5))
        ])
    
    def tf_image_to_tensor(self, im_path):
        im = Image.open(im_path) # read file
        im = im.convert('L') # convert to gray
        im = invert(im) # inverse color (black and white)
        return self.trans(im) # transform and return

    def predict(self, im_path):
        x = self.tf_image_to_tensor(im_path)
        logits = self.model.forward(x[None, :])
        return logits.argmax(-1).item()
