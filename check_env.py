import torch
import torchvision
import torchaudio
import cv2
import numpy
import scipy
import PIL
import matplotlib
import yaml
import tqdm
import sam2

print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('torchvision:', torchvision.__version__)
print('torchaudio:', torchaudio.__version__)
print('opencv:', cv2.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('pillow:', PIL.__version__)
print('matplotlib:', matplotlib.__version__)
print('sam2: OK')
print()
print('All packages imported successfully')
