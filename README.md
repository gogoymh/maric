# MARIC: Multi-Agent Reasoning for Image Classification

MARIC is a multi-agent framework that reformulates image classification as a collaborative reasoning process using Vision-Language Models (VLMs).
Paper: https://arxiv.org/pdf/2509.14860

## Overview

MARIC decomposes image classification into three specialized agents:
- **Outliner Agent**: Analyzes global context and generates targeted prompts
- **Aspect Agents**: Extract fine-grained descriptions from complementary visual perspectives  
- **Reasoning Agent**: Synthesizes outputs through contrastive integration for final classification

## Installation

### Docker Setup

1. Build the Docker image:
```bash
chmod +x docker_build.sh
./docker_build.sh
```

This will:
- Build a Docker image named `maric:latest`
- Start a container with GPU support and mount the current directory

2. Access the container:
```bash
docker exec -it maric bash
```

## Dataset Setup

### OOD-CV Dataset
The OOD-CV dataset must be downloaded manually:
1. Download from https://bzhao.me/OOD-CV/
2. Extract and place the files in `./OOD-CV-Cls/` directory with the following structure:
   - `./OOD-CV-Cls/phase-1/`
   - `./OOD-CV-Cls/phase-2/`

Other datasets (CIFAR-10, Weather, Skin Cancer) will be downloaded automatically when running experiments.

## Running Experiments

Use the `experiment.sh` script to run experiments on different datasets:

```bash
bash experiment.sh
```

The script runs experiments with different configurations:
- **Datasets**: skin_cancer, weather, cifar10, oodcv
- **Models**: llava-7b, llava-13b
- **GPU settings**: Uses 4 GPUs by default
- **Batch sizes**: 32 for 7B model, 16 for 13B model

Results are saved to `experiments/` directory.

### Custom Experiments

You can also run experiments manually:
```bash
python3 main.py --datasets [dataset] --methods maric --vlm [model] --batch_size [size] --gpu_ids [gpus]
```

Example:
```bash
python3 main.py --datasets cifar10 --methods maric --vlm llava-7b --batch_size 32 --gpu_ids 0 1 2 3
```