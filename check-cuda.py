import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import sys
import os

# Verifying installed packages and dependencies
try:
    import bitsandbytes as bnb
    print("Successfully imported bitsandbytes")
except ImportError:
    print("Error importing bitsandbytes. Attempting to install again...")
    #!pip install -q bitsandbytes --upgrade
    os.exit(1)

# Installing required packages (you may want to comment the cell below if you already got these installed)
# !pip install -q transformers accelerate bitsandbytes einops

# Set device, prioritizing GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
