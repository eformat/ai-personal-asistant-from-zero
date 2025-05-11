import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import sys
import os

# Load quantized lmstudio-community/Qwen3-1.7B-GGUF runs with 8G NVIDIA GeForce RTX 4070 Laptop GPU
model_name = "Qwen/Qwen3-1.7B"
model_file = 'Qwen3-1.7B-Q8_0.gguf'

print(f"Loading {model_name}...")
start_time = time.time()

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=False
)

# Trying to load the model with 4-bit quantization for efficiency
try:
    print("Attempting to load model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map="auto",
        quantization_config={"load_in_4bit": True}  # 4-bit quantization for memory efficiency
    )
except Exception as e:
    print(f"Load failed failed with error: {str(e)}")
    os._exit(1)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")
