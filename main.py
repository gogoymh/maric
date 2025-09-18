#!/usr/bin/env python3
"""
MARIC: Multi-Agent based Reasoning for Image Classification
Main execution script
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import torch
import gc
import time
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import hashlib

# Core modules
from maric_framework import MARIC
from vlm_models import create_vlm_model
from baseline_methods import create_baseline_method
from test_all_datasets_random import (
    load_cifar10_test, load_ood_cv_test, 
    load_weather_dataset, load_skin_cancer_dataset,
    download_datasets
)
from experiment_logger import ExperimentLogger


def get_balanced_indices(labels: np.ndarray, samples_per_class: int, 
                        seed: int = 42, cache_dir: str = "indices_cache") -> np.ndarray:
    """
    Get balanced indices for sampling, with caching support.
    
    Args:
        labels: Array of labels
        samples_per_class: Number of samples to select per class
        seed: Random seed for reproducibility
        cache_dir: Directory to cache indices
        
    Returns:
        Array of selected indices
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique identifier for this configuration
    config_str = f"{len(labels)}_{samples_per_class}_{seed}_{np.unique(labels).tolist()}"
    cache_key = hashlib.md5(config_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"balanced_indices_{cache_key}.json")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            print(f"Loaded cached indices from {cache_file}")
            return np.array(cached['indices'])
    
    # Generate balanced indices
    np.random.seed(seed)
    unique_labels = np.unique(labels)
    selected_indices = []
    
    for label in unique_labels:
        # Get all indices for this class
        class_indices = np.where(labels == label)[0]
        
        # Sample from this class
        if len(class_indices) >= samples_per_class:
            # Sample without replacement
            sampled = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            # If not enough samples, use all available
            sampled = class_indices
            print(f"Warning: Class {label} has only {len(class_indices)} samples, less than requested {samples_per_class}")
        
        selected_indices.extend(sampled)
    
    # Shuffle the final indices
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    # Save to cache
    cache_data = {
        'config': {
            'total_samples': len(labels),
            'samples_per_class': samples_per_class,
            'seed': seed,
            'num_classes': len(unique_labels),
            'total_selected': len(selected_indices)
        },
        'indices': selected_indices.tolist()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Saved indices to cache: {cache_file}")
    
    return selected_indices


def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
    """Get GPU memory information using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total,memory.free',
            '--format=csv,nounits,noheader',
            '--id=' + str(gpu_id)
        ], capture_output=True, text=True, check=True)
        
        used, total, free = map(float, result.stdout.strip().split(','))
        return {
            'used': used,
            'total': total,
            'free': free,
            'utilization': (used / total) * 100
        }
    except:
        return None


def select_best_gpu(gpu_ids: List[int] = None) -> int:
    """Select GPU with most free memory"""
    if gpu_ids is None:
        gpu_ids = list(range(6))  # 0-5
    
    best_gpu = None
    max_free_memory = 0
    
    for gpu_id in gpu_ids:
        mem_info = get_gpu_memory_info(gpu_id)
        if mem_info and mem_info['free'] > max_free_memory:
            max_free_memory = mem_info['free']
            best_gpu = gpu_id
    
    return best_gpu if best_gpu is not None else 0


def get_available_gpus(min_free_memory_gb: float = 5.0) -> List[int]:
    """Get list of available GPUs with sufficient free memory"""
    available_gpus = []
    
    for gpu_id in range(6):  # 0-5
        mem_info = get_gpu_memory_info(gpu_id)
        if mem_info and (mem_info['free'] / 1024) >= min_free_memory_gb:
            available_gpus.append(gpu_id)
    
    return available_gpus


def estimate_batch_size(model_name: str, gpu_id: int) -> int:
    """Estimate optimal batch size based on available GPU memory"""
    mem_info = get_gpu_memory_info(gpu_id)
    if not mem_info:
        return 1
    
    total_memory_gb = mem_info['total'] / 1024
    free_memory_gb = mem_info['free'] / 1024
    
    # Detect if it's an A100 GPU (80GB total memory)
    is_a100 = total_memory_gb > 70  # A100 has ~80GB
    
    # Memory estimates per image for different models (with 224x224 images)
    # Updated based on actual usage patterns
    memory_per_image = {
        'llava-7b': 0.3,  # GB per image for 224x224
        'llava-13b': 0.5,
        'qwen-3b': 0.2,
        'qwen-7b': 0.3,
        'gemma-4b': 0.25,  # Estimated based on model size
        'gemma-12b': 0.4   # Estimated based on model size
    }
    
    mem_per_img = memory_per_image.get(model_name, 0.3)
    
    if is_a100:
        # For A100, we can be more aggressive
        # Model takes about 14-15GB, leaving ~65GB for data
        usable_memory = min(free_memory_gb * 0.9, 60)  # Use up to 60GB for data
    else:
        # For other GPUs, be more conservative
        usable_memory = free_memory_gb * 0.7
    
    batch_size = max(1, int(usable_memory / mem_per_img))
    
    # Higher cap for A100 but not too high to maintain GPU utilization
    # Smaller batches can actually improve GPU utilization for generation tasks
    max_batch = 32 if is_a100 else 16
    
    estimated_batch = min(batch_size, max_batch)
    print(f"GPU {gpu_id}: Total {total_memory_gb:.1f}GB, Free {free_memory_gb:.1f}GB, "
          f"Estimated batch size: {estimated_batch}")
    
    return estimated_batch


class ExperimentRunner:
    """Main experiment runner with evaluation capabilities"""
    
    def __init__(self, vlm_type: str = "llava-7b", device: str = "cuda", output_dir: str = "results", 
                 log_samples: int = 5, gpu_id: int = None, gpu_ids: List[int] = None, 
                 batch_size: int = None, use_multi_gpu: bool = False):
        self.vlm_type = vlm_type
        self.output_dir = output_dir
        self.log_samples = log_samples
        os.makedirs(output_dir, exist_ok=True)
        self.use_multi_gpu = use_multi_gpu
        
        # Multi-GPU setup
        if use_multi_gpu and gpu_ids:
            self.gpu_ids = gpu_ids
            self.device = None  # Will be set per worker
            # Estimate batch size based on first GPU if not specified
            if batch_size is None:
                self.batch_size = estimate_batch_size(vlm_type, gpu_ids[0])
            else:
                self.batch_size = batch_size
            print(f"Using multiple GPUs: {self.gpu_ids} with batch size: {self.batch_size}")
        else:
            # Single GPU setup (original logic)
            if gpu_id is None:
                self.gpu_id = select_best_gpu()
            else:
                self.gpu_id = gpu_id
                
            # Set device
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpu_id}"
                torch.cuda.set_device(self.gpu_id)
            else:
                self.device = "cpu"
                
            # Determine batch size if not specified
            if batch_size is None:
                self.batch_size = estimate_batch_size(vlm_type, self.gpu_id) if self.device != "cpu" else 1
            else:
                self.batch_size = batch_size
                
            print(f"Using device: {self.device}, Batch size: {self.batch_size}")
        
        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.timestamp_dir, exist_ok=True)
        
        # Create logger with timestamp-based directory
        self.logger = ExperimentLogger(log_dir=os.path.join(self.timestamp_dir, "logs"), 
                                     max_samples_to_log=log_samples)
        
        # For multi-GPU: model instances per GPU
        self.gpu_models = {} if use_multi_gpu else None
        self.model_locks = {} if use_multi_gpu else None
        
    def prepare_image(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image and resize to 224x224"""
        size = len(img_array)
        if size == 3 * 32 * 32:
            img_shape = (32, 32)
        elif size == 3 * 64 * 64:
            img_shape = (64, 64)
        elif size == 3 * 128 * 128:
            img_shape = (128, 128)
        else:
            side = int(np.sqrt(size // 3))
            img_shape = (side, side)
            
        img_array = img_array.reshape(3, img_shape[0], img_shape[1]).transpose(1, 2, 0)
        img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array.astype(np.uint8)
        
        # Create PIL image and resize to 224x224
        img = Image.fromarray(img_array)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        return img
    
    def _get_or_create_model(self, gpu_id: int, class_names: List[str]):
        """Get or create model for specific GPU (thread-safe)"""
        if gpu_id not in self.gpu_models:
            # Use a global lock for model creation to prevent simultaneous loading
            global_lock = self.model_locks.setdefault('global', threading.Lock())
            with global_lock:
                if gpu_id not in self.gpu_models:
                    device = f"cuda:{gpu_id}"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating model on {device}...")
                    torch.cuda.set_device(gpu_id)
                    vlm_model = create_vlm_model(self.vlm_type, device=device)
                    self.gpu_models[gpu_id] = {
                        'vlm_model': vlm_model,
                        'device': device
                    }
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model on {device} ready!")
        return self.gpu_models[gpu_id]
    
    def _process_batch_on_gpu(self, gpu_id: int, batch_data: List[Tuple[int, np.ndarray, int]], 
                              method_name: str, class_names: List[str]) -> List[Tuple[int, int]]:
        """Process a batch of images on a specific GPU"""
        thread_id = threading.current_thread().name
        print(f"[Thread {thread_id}] GPU {gpu_id}: Starting processing...", flush=True)
        
        try:
            # Set the current CUDA device for this thread
            torch.cuda.set_device(gpu_id)
            
            # Get or create model for this GPU
            model_info = self._get_or_create_model(gpu_id, class_names)
            vlm_model = model_info['vlm_model']
            
            # Create method instance
            if method_name == 'maric':
                method = MARIC(vlm_model=vlm_model, class_names=class_names, 
                              num_aspect_agents=3, logger=self.logger)
            else:
                method = create_baseline_method(method_name, vlm_model, class_names, logger=self.logger)
            
            results = []
            correct_count = 0  # Track correct predictions
            processed_count = 0  # Track processed samples
            total_samples = len(batch_data)
            print(f"[Thread {thread_id}] GPU {gpu_id}: Processing {total_samples} samples...", flush=True)
            
            start_time = time.time()
        except Exception as e:
            print(f"[Thread {thread_id}] GPU {gpu_id}: Error during initialization: {e}", flush=True)
            return [(idx, 0) for idx, _ in batch_data]
        
        # Process in batches for better GPU utilization
        batch_size = self.batch_size
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            current_batch_size = batch_end - batch_start
            
            # Progress reporting - adjust frequency based on batch size
            report_frequency = max(10, batch_size * 2)  # Report every 10 samples or 2 batches, whichever is larger
            if batch_start % report_frequency == 0:
                if batch_start > 0:
                    elapsed = time.time() - start_time
                    processed = batch_start
                    avg_time_per_sample = elapsed / processed
                    remaining_samples = total_samples - processed
                    estimated_remaining = avg_time_per_sample * remaining_samples
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id}: {processed}/{total_samples} samples ({100*processed/total_samples:.1f}%) - "
                          f"Est. remaining: {estimated_remaining:.1f}s", flush=True)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id}: Starting batch processing (batch size: {batch_size})...", flush=True)
            
            # Prepare batch
            batch_imgs = []
            batch_indices = []
            batch_labels = []
            
            # Pre-allocate tensors for better memory efficiency
            with torch.cuda.stream(torch.cuda.Stream()):
                for i in range(batch_start, batch_end):
                    if i < len(batch_data):
                        idx, img_array, label = batch_data[i]
                        batch_imgs.append(self.prepare_image(img_array))
                        batch_indices.append(idx)
                        batch_labels.append(label)
            
            if not batch_imgs:
                continue
                
            try:
                # Check if method supports batch processing
                if hasattr(method, 'classify_batch'):
                    # Determine which samples to log
                    sample_ids_for_batch = []
                    image_paths_for_batch = []
                    true_labels_for_batch = []
                    for i, idx in enumerate(batch_indices):
                        should_log = self.logger.mark_sample_for_logging()
                        sample_ids_for_batch.append(idx if should_log else None)
                        # Use index as image path and get true label
                        image_paths_for_batch.append(f"sample_{idx}" if should_log else None)
                        true_labels_for_batch.append(class_names[batch_labels[i]] if should_log else None)
                    
                    batch_results = method.classify_batch(batch_imgs, 
                                                        sample_ids=sample_ids_for_batch,
                                                        image_paths=image_paths_for_batch,
                                                        true_labels=true_labels_for_batch)
                    for j, result in enumerate(batch_results):
                        # Handle both baseline and MARIC fast results
                        if 'answer' in result:  # MARIC fast/hybrid format
                            pred_class = result.get('answer', class_names[0])
                        else:  # Baseline format
                            pred_class = result.get('prediction', class_names[0])
                        pred_class_lower = pred_class.lower()
                        pred_idx = None
                        for cls_idx, cls in enumerate(class_names):
                            if cls.lower() == pred_class_lower:
                                pred_idx = cls_idx
                                break
                        if pred_idx is None:
                            pred_idx = 0
                        results.append((batch_indices[j], pred_idx))
                        # Update accuracy tracking
                        processed_count += 1
                        if pred_idx == batch_labels[j]:
                            correct_count += 1
                    # Print accuracy after each batch - always print for small batch sizes
                    if processed_count > 0 and (batch_size <= 2 or processed_count % 10 == 0 or processed_count == total_samples):
                        current_accuracy = correct_count / processed_count
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id}: {method_name.upper()} - Processed: {processed_count}/{total_samples}, Accuracy: {current_accuracy:.3f}", flush=True)
                else:
                    # Process individually (for MARIC or methods without batch support)
                    for j, (img, idx) in enumerate(zip(batch_imgs, batch_indices)):
                        if hasattr(method, 'classify') and not isinstance(method, MARIC):
                            # Check if we should log this sample
                            should_log = self.logger.mark_sample_for_logging()
                            sample_id = idx if should_log else None
                            image_path = f"sample_{idx}" if should_log else None
                            true_label = class_names[batch_labels[j]] if should_log else None
                            result = method.classify(img, sample_id=sample_id, 
                                                   image_path=image_path, true_label=true_label)
                            pred_class = result.get('prediction', class_names[0])
                        else:  # MARIC
                            # Check if we should log this sample
                            should_log = self.logger.mark_sample_for_logging()
                            sample_id = idx if should_log else None
                            image_path = f"sample_{idx}" if should_log else None
                            true_label = class_names[batch_labels[j]] if should_log else None
                            
                            if hasattr(method, 'classify'):
                                result = method.classify(img, sample_id=sample_id, 
                                                       image_path=image_path, true_label=true_label)
                                if isinstance(result, dict):
                                    pred_class = result.get('answer', class_names[0])
                                else:
                                    pred_class = result
                            else:
                                result = method.predict(img, sample_id=sample_id)
                                if isinstance(result, dict):
                                    pred_class = result.get('final_answer', result.get('answer', class_names[0]))
                                else:
                                    pred_class = result
                        
                        # Convert prediction to index
                        pred_class_lower = pred_class.lower()
                        pred_idx = None
                        for cls_idx, cls in enumerate(class_names):
                            if cls.lower() == pred_class_lower:
                                pred_idx = cls_idx
                                break
                        if pred_idx is None:
                            pred_idx = 0
                            
                        results.append((idx, pred_idx))
                        # Update accuracy tracking
                        processed_count += 1
                        if pred_idx == batch_labels[j]:
                            correct_count += 1
                        # Print accuracy - more frequently for small batch sizes
                        print_frequency = 5 if batch_size <= 2 else 10
                        if processed_count % print_frequency == 0 or processed_count == total_samples:
                            current_accuracy = correct_count / processed_count
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id}: {method_name.upper()} - Processed: {processed_count}/{total_samples}, Accuracy: {current_accuracy:.3f}", flush=True)
            except Exception as e:
                error_msg = str(e)
                if "expected scalar type" in error_msg or "dtype" in error_msg:
                    print(f"\nDtype error on GPU {gpu_id}, sample {idx}: {error_msg}", flush=True)
                    # Try to recover by clearing cache and retrying once
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            # Retry once after clearing cache
                            if hasattr(method, 'classify') and not isinstance(method, MARIC):
                                # Note: Not re-checking should_log since this is a retry
                                result = method.classify(img, sample_id=sample_id if 'sample_id' in locals() else None)
                                pred_class = result.get('prediction', class_names[0])
                            else:  # MARIC
                                result = method.predict(img)
                                if isinstance(result, dict):
                                    pred_class = result.get('final_answer', result.get('answer', class_names[0]))
                                else:
                                    pred_class = result
                            
                            # Convert prediction to index
                            pred_class_lower = pred_class.lower()
                            pred_idx = None
                            for i, cls in enumerate(class_names):
                                if cls.lower() == pred_class_lower:
                                    pred_idx = i
                                    break
                            if pred_idx is None:
                                pred_idx = 0
                                
                            results.append((idx, pred_idx))
                            # Update accuracy tracking for retry
                            processed_count += 1
                            if pred_idx == batch_labels[j]:
                                correct_count += 1
                            print(f"  ✓ Retry successful for sample {idx}", flush=True)
                            continue
                        except Exception as retry_e:
                            print(f"  ✗ Retry failed for sample {idx}: {retry_e}", flush=True)
                            results.append((idx, 0))
                            processed_count += 1
                else:
                    print(f"Error processing sample {idx} on GPU {gpu_id}: {e}", flush=True)
                    results.append((idx, 0))
                    processed_count += 1
        
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        final_accuracy = correct_count / processed_count if processed_count > 0 else 0
        print(f"GPU {gpu_id}: Completed {total_samples} samples in {total_time:.1f}s "
              f"(avg: {avg_time_per_sample:.2f}s/sample) - Final Accuracy: {final_accuracy:.3f}", flush=True)
        return results
    
    def evaluate_method(self, method, test_images: np.ndarray, test_labels: np.ndarray, 
                       class_names: List[str], method_name: str, num_samples: int = None,
                       indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate a single method on test data with batch processing"""
        print(f"\nEvaluating {method_name}...")
        start_time = time.time()
        
        # Use provided indices or generate new ones
        if indices is not None:
            # Use provided indices
            indices = indices
            num_samples = len(indices)
        else:
            # Use all samples if num_samples is None
            if num_samples is None:
                num_samples = len(test_labels)
            else:
                num_samples = min(num_samples, len(test_labels))
            
            indices = np.random.choice(len(test_labels), num_samples, replace=False)
        
        predictions = []
        correct_count = 0  # Track correct predictions
        processed_count = 0  # Track processed samples
        
        # Check if method supports batch processing
        supports_batch = hasattr(method, 'classify_batch') or hasattr(method, 'predict_batch')
        
        if supports_batch and self.batch_size > 1:
            # Batch processing
            print(f"Using batch size: {self.batch_size}")
            for i in tqdm(range(0, len(indices), self.batch_size), desc=f"{method_name} (batches)"):
                batch_indices = indices[i:i + self.batch_size]
                batch_images = [self.prepare_image(test_images[idx]) for idx in batch_indices]
                
                try:
                    # Batch classification
                    if hasattr(method, 'classify_batch'):
                        # Prepare logging information
                        sample_ids_for_batch = []
                        image_paths_for_batch = []
                        true_labels_for_batch = []
                        for j, idx in enumerate(batch_indices):
                            should_log = self.logger.mark_sample_for_logging()
                            sample_ids_for_batch.append(idx if should_log else None)
                            image_paths_for_batch.append(f"sample_{idx}" if should_log else None)
                            true_labels_for_batch.append(class_names[test_labels[idx]] if should_log else None)
                        
                        batch_results = method.classify_batch(batch_images, 
                                                            sample_ids=sample_ids_for_batch,
                                                            image_paths=image_paths_for_batch,
                                                            true_labels=true_labels_for_batch)
                        for result in batch_results:
                            # Handle both baseline and MARIC results
                            if 'answer' in result:  # MARIC format
                                pred_class = result.get('answer', class_names[0])
                            else:  # Baseline format
                                pred_class = result.get('prediction', class_names[0])
                            pred_class_lower = pred_class.lower()
                            pred_idx = None
                            for idx, cls in enumerate(class_names):
                                if cls.lower() == pred_class_lower:
                                    pred_idx = idx
                                    break
                            predictions.append(pred_idx if pred_idx is not None else 0)
                            # Update accuracy tracking
                            processed_count += 1
                            if pred_idx == test_labels[batch_indices[j]]:
                                correct_count += 1
                        # Print accuracy after each batch
                        current_accuracy = correct_count / processed_count if processed_count > 0 else 0
                        tqdm.write(f"{method_name}: Processed {processed_count}/{num_samples}, Accuracy: {current_accuracy:.3f}")
                    else:  # Fallback for methods without classify_batch
                        # Process individually
                        for idx in batch_indices:
                            img = self.prepare_image(test_images[idx])
                            result = method.predict(img)
                            if isinstance(result, dict):
                                pred_class = result.get('final_answer', result.get('answer', class_names[0]))
                            else:
                                pred_class = result
                            pred_class_lower = pred_class.lower()
                            pred_idx = None
                            for cls_idx, cls in enumerate(class_names):
                                if cls.lower() == pred_class_lower:
                                    pred_idx = cls_idx
                                    break
                            predictions.append(pred_idx if pred_idx is not None else 0)
                            # Update accuracy tracking
                            processed_count += 1
                            if pred_idx == test_labels[idx]:
                                correct_count += 1
                        # Print accuracy after each individual processing in fallback
                        current_accuracy = correct_count / processed_count if processed_count > 0 else 0
                        tqdm.write(f"{method_name}: Processed {processed_count}/{num_samples}, Accuracy: {current_accuracy:.3f}")
                except Exception as e:
                    print(f"\nError processing batch: {e}")
                    # Fall back to individual processing for this batch
                    for idx in batch_indices:
                        predictions.append(0)
        else:
            # Individual processing (original code)
            for idx in tqdm(indices, desc=method_name):
                img = self.prepare_image(test_images[idx])
                
                try:
                    # Determine if we should log this sample
                    should_log = self.logger.mark_sample_for_logging()
                    sample_id = idx if should_log else None
                    
                    if hasattr(method, 'classify') and not isinstance(method, MARIC):
                        image_path = f"sample_{idx}"
                        true_label = class_names[test_labels[idx]]
                        if sample_id is not None:
                            result = method.classify(img, sample_id=sample_id, 
                                                   image_path=image_path, true_label=true_label)
                        else:
                            result = method.classify(img)
                        # Handle both MARIC fast and baseline formats
                        if isinstance(result, dict) and 'answer' in result:
                            pred_class = result.get('answer', class_names[0])
                        else:
                            pred_class = result.get('prediction', class_names[0])
                    else:  # Original MARIC
                        image_path = f"sample_{idx}"
                        true_label = class_names[test_labels[idx]]
                        if hasattr(method, 'classify'):
                            result = method.classify(img, sample_id=sample_id, 
                                                   image_path=image_path, true_label=true_label)
                            if isinstance(result, dict):
                                pred_class = result.get('answer', class_names[0])
                            else:
                                pred_class = result
                        else:
                            if sample_id is not None:
                                result = method.predict(img, sample_id=sample_id)
                            else:
                                result = method.predict(img)
                            # Handle both string and dict results
                            if isinstance(result, dict):
                                pred_class = result.get('final_answer', result.get('answer', class_names[0]))
                            else:
                                pred_class = result
                        
                    # Convert prediction to lowercase for comparison (MARIC returns lowercase)
                    pred_class_lower = pred_class.lower()
                    # Find matching class index (case-insensitive)
                    pred_idx = None
                    for cls_idx, cls in enumerate(class_names):
                        if cls.lower() == pred_class_lower:
                            pred_idx = cls_idx
                            break
                    if pred_idx is None:
                        pred_idx = 0
                    predictions.append(pred_idx)
                    # Update accuracy tracking
                    processed_count += 1
                    if pred_idx == test_labels[idx]:
                        correct_count += 1
                    
                    # Debug: print first few predictions (commented out)
                    # if processed_count <= 5:
                    #     true_class = class_names[test_labels[idx]]
                    #     tqdm.write(f"  Sample {idx}: True={true_class}, Predicted={pred_class}, Correct={pred_idx == test_labels[idx]}")
                    # Print accuracy - more frequently for small datasets or batch size 1
                    print_frequency = 1 if self.batch_size == 1 else (5 if num_samples < 100 else 10)
                    if processed_count % print_frequency == 0 or processed_count == len(indices):
                        current_accuracy = correct_count / processed_count
                        tqdm.write(f"{method_name}: Processed {processed_count}/{num_samples}, Accuracy: {current_accuracy:.3f}")
                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    predictions.append(0)  # Default prediction on error
                    processed_count += 1
                
        # Calculate metrics - fix array indexing
        test_labels_array = np.array(test_labels)
        true_labels_subset = test_labels_array[indices]
        
        accuracy = accuracy_score(true_labels_subset, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_subset, predictions, average='macro', zero_division=0
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n{method_name} completed in {elapsed_time:.2f} seconds")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time_seconds': elapsed_time
        }
    
    def evaluate_method_multi_gpu(self, test_images: np.ndarray, test_labels: np.ndarray,
                                 class_names: List[str], method_name: str, num_samples: int = None,
                                 indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate a method using multiple GPUs in parallel"""
        print(f"\nEvaluating {method_name} using GPUs: {self.gpu_ids}...")
        start_time = time.time()
        
        # Pre-load models on all GPUs sequentially to avoid resource contention
        print("Pre-loading models on all GPUs...")
        for gpu_id in self.gpu_ids:
            self._get_or_create_model(gpu_id, class_names)
        print("All models loaded successfully!\n")
        
        # Use provided indices or generate new ones
        if indices is not None:
            # Use provided indices
            indices = indices
            num_samples = len(indices)
        else:
            # Use all samples if num_samples is None
            if num_samples is None:
                num_samples = len(test_labels)
            else:
                num_samples = min(num_samples, len(test_labels))
            
            indices = np.random.choice(len(test_labels), num_samples, replace=False)
        
        # Prepare batches for each GPU
        num_gpus = len(self.gpu_ids)
        samples_per_gpu = len(indices) // num_gpus
        gpu_batches = []
        
        for i, gpu_id in enumerate(self.gpu_ids):
            start_idx = i * samples_per_gpu
            end_idx = start_idx + samples_per_gpu if i < num_gpus - 1 else len(indices)
            batch_indices = indices[start_idx:end_idx]
            batch_data = [(idx, test_images[idx], test_labels[idx]) for idx in batch_indices]
            gpu_batches.append((gpu_id, batch_data))
        
        # Process batches in parallel
        all_results = []
        print(f"\nDistributing {len(indices)} samples across {num_gpus} GPUs:")
        for gpu_id, batch_data in gpu_batches:
            print(f"  GPU {gpu_id}: {len(batch_data)} samples")
        print()
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for gpu_id, batch_data in gpu_batches:
                future = executor.submit(self._process_batch_on_gpu, gpu_id, batch_data, 
                                       method_name.lower(), class_names)
                futures[future] = gpu_id
            
            # Collect results
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                gpu_id = futures[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    print(f"\n✓ GPU {gpu_id} completed ({completed_count}/{num_gpus} GPUs finished)")
                except Exception as e:
                    print(f"\n✗ Error in GPU {gpu_id} batch processing: {e}")
        
        # Sort results by original index and extract predictions
        all_results.sort(key=lambda x: x[0])
        predictions = [pred for _, pred in all_results]
        sorted_indices = [idx for idx, _ in all_results]
        
        # Calculate metrics
        test_labels_array = np.array(test_labels)
        true_labels_subset = test_labels_array[sorted_indices]
        
        accuracy = accuracy_score(true_labels_subset, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_subset, predictions, average='macro', zero_division=0
        )
        
        elapsed_time = time.time() - start_time
        print(f"{method_name} completed in {elapsed_time:.2f} seconds using {num_gpus} GPUs")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time_seconds': elapsed_time,
            'gpus_used': num_gpus
        }
    
    def run_experiment(self, dataset_name: str, num_samples: int = None, 
                      methods_to_run: List[str] = None) -> Dict[str, Any]:
        """Run experiment on a single dataset"""
        # Start experiment logging
        experiment_config = {
            'dataset': dataset_name,
            'vlm_model': self.vlm_type,
            'num_samples': num_samples,
            'log_samples': self.log_samples,
            'methods': methods_to_run or ['direct', 'cot', 'savr', 'maric']
        }
        self.logger.start_experiment(f"{dataset_name}_{self.vlm_type}", experiment_config)
        
        # Load dataset
        print(f"\nLoading {dataset_name} dataset...")
        
        if dataset_name == 'cifar10':
            test_data, test_labels, class_names, _ = load_cifar10_test()
        elif dataset_name == 'oodcv':
            test_data, test_labels, class_names, _ = load_ood_cv_test()
        elif dataset_name == 'weather':
            weather_path, _ = download_datasets()
            test_data, test_labels, class_names, _ = load_weather_dataset(weather_path)
        elif dataset_name == 'skin_cancer':
            _, skin_cancer_path = download_datasets()
            test_data, test_labels, class_names, _ = load_skin_cancer_dataset(skin_cancer_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        print(f"Dataset: {len(test_labels)} samples, {len(class_names)} classes")
        print(f"Classes: {', '.join(class_names)}")
        
        # Determine indices to use
        indices = None
        if dataset_name in ['cifar10', 'oodcv']:
            # Always use balanced sampling with 100 samples per class for CIFAR10 and OOD-CV
            samples_per_class = 100
            indices = get_balanced_indices(test_labels, samples_per_class, seed=42)
            print(f"Using balanced sampling for {dataset_name}: {samples_per_class} samples per class, total {len(indices)} samples")
        elif num_samples is not None:
            # Use random sampling for other datasets
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(test_labels), min(num_samples, len(test_labels)), replace=False)
            print(f"Using random sampling: {len(indices)} samples")
        
        results = {}
        
        # Default to all methods if not specified
        if methods_to_run is None:
            methods_to_run = ['direct', 'cot', 'savr', 'maric']
        
        if self.use_multi_gpu:
            # Multi-GPU mode: use evaluate_method_multi_gpu
            for method_name in methods_to_run:
                results[method_name] = self.evaluate_method_multi_gpu(
                    test_data, test_labels, class_names,
                    method_name.upper(), num_samples, indices
                )
                # Print results after each method
                self._print_method_results(dataset_name, method_name, results[method_name])
        else:
            # Single GPU mode: original logic
            print(f"\nLoading VLM model: {self.vlm_type} on {self.device}")
            vlm_model = create_vlm_model(self.vlm_type, device=self.device)
            
            # Evaluate baselines
            baseline_methods = [m for m in ['direct', 'cot', 'savr'] if m in methods_to_run]
            for baseline_name in baseline_methods:
                method = create_baseline_method(baseline_name, vlm_model, class_names, logger=self.logger)
                results[baseline_name] = self.evaluate_method(
                    method, test_data, test_labels, class_names, 
                    baseline_name.upper(), num_samples, indices
                )
                # Print results after each method
                self._print_method_results(dataset_name, baseline_name, results[baseline_name])
            
            # Evaluate MARIC if requested
            if 'maric' in methods_to_run:
                print("\nInitializing MARIC...")
                maric = MARIC(vlm_model=vlm_model, class_names=class_names, 
                             num_aspect_agents=3, logger=self.logger)
                results['maric'] = self.evaluate_method(
                    maric, test_data, test_labels, class_names, 
                    'MARIC', num_samples, indices
                )
                # Print results after MARIC
                self._print_method_results(dataset_name, 'maric', results['maric'])
            
            # Clean up
            del vlm_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # Save experiment logs
        self.logger.save_experiment()
            
        return results
    
    def _print_method_results(self, dataset: str, method: str, metrics: Dict[str, float]):
        """Print results for a single method"""
        print(f"\n{'='*60}")
        print(f"✓ Completed: {dataset.upper()} - {method.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:    {metrics['accuracy']:.3f}")
        print(f"Precision:   {metrics['precision']:.3f}")
        print(f"Recall:      {metrics['recall']:.3f}")
        print(f"F1-Score:    {metrics['f1_score']:.3f}")
        print(f"Time:        {metrics.get('time_seconds', 0):.1f}s")
        if 'gpus_used' in metrics:
            print(f"GPUs Used:   {metrics['gpus_used']}")
        print(f"{'='*60}\n")
    
    def save_results(self, all_results: Dict[str, Any], config: Dict[str, Any] = None):
        """Save results to files"""
        # Use the timestamp directory created in __init__
        
        # Prepare results with metadata
        results_with_metadata = {
            'configuration': config or {},
            'results': all_results
        }
        
        # Save JSON
        json_path = os.path.join(self.timestamp_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        print(f"\nResults saved to {json_path}")
        
        # Create summary table
        summary_data = []
        for dataset, methods in all_results.items():
            for method, metrics in methods.items():
                summary_data.append({
                    'Dataset': dataset,
                    'Method': method.upper(),
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1_score']:.3f}",
                    'Time (s)': f"{metrics.get('time_seconds', 0):.2f}"
                })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.timestamp_dir, 'summary.csv')
        df.to_csv(csv_path, index=False)
        
        print("\nSummary Table:")
        print(df.to_string(index=False))
        print(f"\nSummary saved to {csv_path}")


def main():
    # Disable output buffering for real-time progress updates
    import sys
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # Force unbuffered output by setting PYTHONUNBUFFERED
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    parser = argparse.ArgumentParser(description='MARIC Experiments')
    parser.add_argument('--datasets', nargs='+', 
                       default=['cifar10', 'oodcv', 'weather', 'skin_cancer'],
                       choices=['cifar10', 'oodcv', 'weather', 'skin_cancer'],
                       help='Datasets to evaluate')
    parser.add_argument('--methods', nargs='+',
                       default=['direct', 'cot', 'savr', 'maric'],
                       choices=['direct', 'cot', 'savr', 'maric'],
                       help='Methods to evaluate (default: all)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples per dataset (default: use all samples)')
    parser.add_argument('--vlm', type=str, default='llava-7b',
                       choices=['llava-7b', 'llava-13b'],
                       help='VLM model to use')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--log_samples', type=int, default=5,
                       help='Number of samples to log in detail (default: 5)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use (0-5). If not specified, selects GPU with most free memory')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                       help='Multiple GPU IDs to use for parallel processing (e.g., --gpu_ids 0 1 2)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing. If not specified, automatically determined based on GPU memory')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multiple GPUs for parallel processing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MARIC: Multi-Agent based Reasoning for Image Classification")
    print("="*60)
    print(f"Configuration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Methods: {args.methods}")
    print(f"  Samples per dataset: {'all' if args.num_samples is None else args.num_samples}")
    print(f"  VLM model: {args.vlm}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Detailed logging: first {args.log_samples} samples")
    
    # Determine GPU configuration
    if args.multi_gpu or args.gpu_ids:
        # Multi-GPU mode
        if args.gpu_ids:
            gpu_ids = args.gpu_ids
        else:
            # Auto-detect available GPUs
            gpu_ids = get_available_gpus(min_free_memory_gb=5.0)
            if not gpu_ids:
                print("No GPUs with sufficient free memory found. Falling back to single GPU.")
                gpu_ids = None
        
        if gpu_ids and len(gpu_ids) > 1:
            print(f"  Multi-GPU mode: Using GPUs {gpu_ids}")
            runner = ExperimentRunner(vlm_type=args.vlm, output_dir=args.output_dir, 
                                    log_samples=args.log_samples, gpu_ids=gpu_ids,
                                    batch_size=args.batch_size, use_multi_gpu=True)
        else:
            # Fall back to single GPU
            runner = ExperimentRunner(vlm_type=args.vlm, output_dir=args.output_dir, 
                                    log_samples=args.log_samples, gpu_id=args.gpu_id,
                                    batch_size=args.batch_size)
    else:
        # Single GPU mode
        runner = ExperimentRunner(vlm_type=args.vlm, output_dir=args.output_dir, 
                                log_samples=args.log_samples, gpu_id=args.gpu_id,
                                batch_size=args.batch_size)
    
    all_results = {}
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        results = runner.run_experiment(dataset, args.num_samples, 
                                      methods_to_run=args.methods)
        all_results[dataset] = results
    
    # Save results with configuration
    config = {
        'vlm_model': args.vlm,
        'datasets': args.datasets,
        'methods': args.methods,
        'num_samples': args.num_samples,
        'output_dir': args.output_dir,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'CUDA' if torch.cuda.is_available() else 'CPU'
    }
    runner.save_results(all_results, config)
    
    # Print final comprehensive summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print("="*80)
    
    for dataset in all_results:
        print(f"\n{dataset.upper()} Dataset Results:")
        print("-" * 60)
        print(f"{'Method':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time (s)':<10}")
        print("-" * 60)
        
        for method, metrics in all_results[dataset].items():
            print(f"{method.upper():<10} "
                  f"{metrics['accuracy']:.3f}      "
                  f"{metrics['precision']:.3f}       "
                  f"{metrics['recall']:.3f}     "
                  f"{metrics['f1_score']:.3f}      "
                  f"{metrics.get('time_seconds', 0):.1f}")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to: {runner.timestamp_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
