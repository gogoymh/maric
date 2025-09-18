import os
import torch

# Set up GPU environment before importing models
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # For MPT model to use GPU instead of CPU
    os.environ['INIT_DEVICE'] = 'cuda'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'
    print(f"GPU setup complete. Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")