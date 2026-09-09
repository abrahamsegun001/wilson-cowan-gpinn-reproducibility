#!/usr/bin/env python3
import sys
import numpy, scipy, pandas, matplotlib, torch
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
