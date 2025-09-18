"""
Logging utilities for MARIC experiments
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import threading


class ExperimentLogger:
    """Logger for detailed experiment tracking"""
    
    def __init__(self, log_dir: str = "logs", max_samples_to_log: int = 5):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            max_samples_to_log: Maximum number of samples to log in detail
        """
        self.log_dir = log_dir
        self.max_samples_to_log = max_samples_to_log
        self.current_experiment = None
        self.logs = []
        self.sample_count = 0
        self.saved_early = False
        self.lock = threading.Lock()
        self.log_file_path = None
        self.appended_samples = set()  # Track which samples have been appended
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Start a new experiment"""
        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'samples': []
        }
        self.sample_count = 0
        self.saved_early = False
        self.appended_samples = set()
        
        # Create log file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment_name}_{timestamp}_streaming.json"
        self.log_file_path = os.path.join(self.log_dir, filename)
        
        # Initialize file with experiment metadata
        initial_data = {
            'name': experiment_name,
            'config': config,
            'timestamp': self.current_experiment['timestamp'],
            'samples': []
        }
        with open(self.log_file_path, 'w') as f:
            json.dump(initial_data, f, indent=2, default=str)
        
        # print(f"[Logger] Initialized streaming log file: {self.log_file_path}")
        
    def should_log_sample(self) -> bool:
        """Check if we should log this sample"""
        return self.sample_count < self.max_samples_to_log
        
    def mark_sample_for_logging(self, sample_idx: Optional[int] = None) -> bool:
        """Check if we should log this sample and increment counter if yes"""
        with self.lock:
            if self.saved_early:
                return False
            
            # If sample_idx is provided, use index-based decision for consistency
            if sample_idx is not None:
                should_log = sample_idx < self.max_samples_to_log
                if should_log and sample_idx not in self.appended_samples:
                    # print(f"[Logger] Marking sample index {sample_idx} for logging")
                    return True
                return False
            
            # Original counter-based logic (for backward compatibility)
            if self.sample_count < self.max_samples_to_log:
                self.sample_count += 1
                # print(f"[Logger] Marking sample {self.sample_count}/{self.max_samples_to_log} for logging")
                return True
            return False
    
    def log_sample(self, sample_id: int, method_name: str, data: Dict[str, Any]):
        """Log a single sample's processing"""
        # Note: Don't check should_log_sample here - that decision should be made
        # by the caller who knows if this sample should be logged
        with self.lock:
            sample_log = {
                'sample_id': sample_id,
                'method': method_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': data
            }
            
            self.current_experiment['samples'].append(sample_log)
            # print(f"[Logger] Added sample {sample_id} for method {method_name} to log (total: {len(self.current_experiment['samples'])})")
            
            # Immediately append to file
            self._append_sample_to_file(sample_log)
            
            # Check if we've reached the limit
            if len(self.current_experiment['samples']) >= self.max_samples_to_log and not self.saved_early:
                # print(f"[Logger] Reached max samples ({self.max_samples_to_log})")
                self.saved_early = True
        
    def log_maric_step(self, sample_id: int, step_name: str, step_data: Dict[str, Any]):
        """Log a specific MARIC processing step"""
        # Note: We don't check should_log_sample here because the main loop already decided to log this sample
        
        with self.lock:
            # Find or create sample entry
            sample_entry = None
            for sample in self.current_experiment['samples']:
                if sample['sample_id'] == sample_id and sample['method'] == 'MARIC':
                    sample_entry = sample
                    break
                    
            if sample_entry is None:
                sample_entry = {
                    'sample_id': sample_id,
                    'method': 'MARIC',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data': {
                        'steps': {}
                    }
                }
                self.current_experiment['samples'].append(sample_entry)
                
            # Ensure steps dict exists
            if 'steps' not in sample_entry['data']:
                sample_entry['data']['steps'] = {}
                
            # Add step data
            sample_entry['data']['steps'][step_name] = step_data
            
            # Check if this is the final step (reasoning_agent)
            if step_name == 'reasoning_agent':
                # Count MARIC samples that are complete (have reasoning_agent step)
                complete_maric_samples = sum(1 for sample in self.current_experiment['samples'] 
                                           if sample['method'] == 'MARIC' 
                                           and 'steps' in sample.get('data', {})
                                           and 'reasoning_agent' in sample['data']['steps'])
                
                # print(f"[Logger] MARIC sample {sample_id} completed (total complete: {complete_maric_samples})")
                
                # Immediately append to JSON file
                self._append_sample_to_json_file(sample_entry)
                
                # Check if we've reached the limit
                if complete_maric_samples >= self.max_samples_to_log and not self.saved_early:
                    # print(f"[Logger] Reached max MARIC samples ({self.max_samples_to_log})")
                    self.saved_early = True
        
    def log_prediction_result(self, sample_id: int, prediction_data: Dict[str, Any]):
        """Log prediction results for a sample that was already selected for logging"""
        # Find the sample entry
        sample_entry = None
        for sample in self.current_experiment['samples']:
            if sample['sample_id'] == sample_id and sample['method'] == 'MARIC':
                sample_entry = sample
                break
                
        if sample_entry is None:
            # This shouldn't happen if MARIC steps were logged properly
            sample_entry = {
                'sample_id': sample_id,
                'method': 'MARIC',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': {}
            }
            self.current_experiment['samples'].append(sample_entry)
            
        # Add prediction results to the sample data
        sample_entry['data']['prediction'] = prediction_data
        
    def save_experiment_early(self):
        """Save experiment early when sample limit is reached"""
        if self.current_experiment is None or self.saved_early:
            return
            
        # Create filename with timestamp and early marker
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.current_experiment['name']}_{timestamp}_early.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"\n[Logger] Saving early log with {len(self.current_experiment['samples'])} samples...")
        if len(self.current_experiment['samples']) == 0:
            print(f"[Logger] WARNING: No samples to save! This shouldn't happen.")
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
            
        print(f"[Logger] Early log saved (first {self.max_samples_to_log} samples): {filepath}\n")
        
        # Mark as saved early but don't reset experiment
        self.saved_early = True
        
    def save_experiment(self):
        """Save current experiment to file"""
        if self.current_experiment is None:
            return
            
        # If already saved early, skip
        if self.saved_early:
            # print("Logs already saved early. Skipping final save.")
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.current_experiment['name']}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
            
        # print(f"Detailed logs saved to: {filepath}")
        
        # Reset
        self.current_experiment = None
        self.sample_count = 0
        self.saved_early = False
        self.appended_samples = set()
    
    def _append_sample_to_file(self, sample_log: Dict[str, Any]):
        """Append a single sample to the log file immediately"""
        if not self.log_file_path:
            return
        
        # Create unique key for this sample
        sample_key = f"{sample_log['method']}_{sample_log['sample_id']}"
        
        # For MARIC, only append when complete (has reasoning_agent step)
        if sample_log['method'] == 'MARIC':
            if 'steps' not in sample_log.get('data', {}) or 'reasoning_agent' not in sample_log['data']['steps']:
                return  # Don't append incomplete MARIC samples
        
        # Check if already appended
        if sample_key in self.appended_samples:
            return
        
        # Use a more robust append method that doesn't require reading the entire file
        try:
            # Create a separate file for each sample to avoid conflicts
            sample_filename = f"{sample_key}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            sample_path = os.path.join(self.log_dir, "samples", sample_filename)
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            
            # Write sample to individual file
            with open(sample_path, 'w') as f:
                json.dump(sample_log, f, indent=2, default=str)
            
            self.appended_samples.add(sample_key)
            # print(f"[Logger] Saved {sample_key} to individual file: {sample_filename}")
            
            # Also append to main log file (with file locking for thread safety)
            import fcntl
            try:
                with open(self.log_file_path, 'r+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        # Read current content
                        f.seek(0)
                        data = json.load(f)
                        
                        # Add new sample
                        data['samples'].append(sample_log)
                        
                        # Write back
                        f.seek(0)
                        f.truncate()
                        json.dump(data, f, indent=2, default=str)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                print(f"[Logger] Error updating main log file: {e}")
                
        except Exception as e:
            print(f"[Logger] Error saving sample: {e}")
    
    def _append_sample_to_json_file(self, sample_log: Dict[str, Any]):
        """Append a single sample to the JSON file immediately (simple version)"""
        if not self.log_file_path:
            return
        
        try:
            # Simple approach: append to a line-delimited JSON file
            sample_file = self.log_file_path.replace('_streaming.json', '_samples.jsonl')
            with open(sample_file, 'a') as f:
                json.dump(sample_log, f, default=str)
                f.write('\n')
            # print(f"[Logger] Appended sample {sample_log['sample_id']} to {sample_file}")
        except Exception as e:
            print(f"[Logger] Error appending sample: {e}")
        
    def log_baseline_inference(self, sample_id: int, method_name: str, 
                             prompt: str, response: str, prediction: str,
                             image_path: Optional[str] = None, true_label: Optional[str] = None):
        """Log baseline method inference"""
        # Note: Don't check should_log_sample here - that decision should be made
        # by the main loop when it decides which samples to log
        # print(f"[Logger] log_baseline_inference called for sample {sample_id}, method {method_name}")
        # if true_label:
        #     print(f"[Logger] True: {true_label}, Predicted: {prediction}, Correct: {prediction.lower() == true_label.lower()}")
            
        data = {
            'prompt': prompt,
            'response': response,
            'prediction': prediction
        }
        
        # Add optional fields if provided
        if image_path is not None:
            data['image_path'] = image_path
        if true_label is not None:
            data['true_label'] = true_label
            # Add whether prediction was correct
            data['is_correct'] = (prediction.lower() == true_label.lower())
            
        self.log_sample(sample_id, method_name, data)
    
    def save_incorrect_case(self, sample_id: int, prediction_data: Dict[str, Any], 
                           maric_steps: Optional[Dict[str, Any]] = None):
        """Immediately save an incorrect prediction case to a separate file"""
        # Create incorrect cases directory
        incorrect_dir = os.path.join(self.log_dir, "incorrect_cases")
        os.makedirs(incorrect_dir, exist_ok=True)
        
        # Create filename based on experiment name and sample ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.current_experiment['name']}_sample_{sample_id}_{timestamp}.json"
        filepath = os.path.join(incorrect_dir, filename)
        
        # Prepare data to save
        incorrect_case = {
            'experiment_name': self.current_experiment['name'],
            'experiment_config': self.current_experiment['config'],
            'sample_id': sample_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_data': prediction_data
        }
        
        # Include MARIC steps if provided directly
        if maric_steps is not None:
            incorrect_case['maric_steps'] = maric_steps
        else:
            # Find and include all MARIC steps for this sample from existing logs
            for sample in self.current_experiment['samples']:
                if sample['sample_id'] == sample_id and sample['method'] == 'MARIC':
                    incorrect_case['maric_steps'] = sample.get('data', {}).get('steps', {})
                    break
        
        # Save to file immediately
        with open(filepath, 'w') as f:
            json.dump(incorrect_case, f, indent=2, default=str)
        
        print(f"\nIncorrect case saved: {filepath}")