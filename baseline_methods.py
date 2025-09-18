import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
from vlm_models import VLMModel, create_vlm_model
from experiment_logger import ExperimentLogger
from prompt_manager import PromptManager
import re

class BaselineMethod:
    """Base class for baseline methods"""
    
    def __init__(self, vlm_model: VLMModel, class_names: List[str], logger: Optional[ExperimentLogger] = None):
        self.vlm_model = vlm_model
        self.class_names = class_names
        self.logger = logger
        self.current_sample_id = None
        self.current_image_path = None
        self.current_true_label = None
        self.prompt_manager = PromptManager(prompts_dir="prompts")
        
    def classify(self, image: Image.Image, sample_id: Optional[int] = None, 
                image_path: Optional[str] = None, true_label: Optional[str] = None) -> Dict[str, Any]:
        self.current_sample_id = sample_id
        self.current_image_path = image_path
        self.current_true_label = true_label
        return self._classify_internal(image)
        
    def _classify_internal(self, image: Image.Image) -> Dict[str, Any]:
        raise NotImplementedError
        
    def classify_batch(self, images: List[Image.Image], sample_ids: Optional[List[Optional[int]]] = None,
                      image_paths: Optional[List[Optional[str]]] = None, 
                      true_labels: Optional[List[Optional[str]]] = None) -> List[Dict[str, Any]]:
        """Batch classification - use VLM's batch processing if available"""
        # Check if VLM supports true batch processing
        if hasattr(self.vlm_model, 'generate_batch') and len(images) > 1:
            # Get prompts for all images
            prompts = []
            for _ in images:
                prompts.append(self._get_prompt())
            
            # Generate responses for all images at once
            responses = self.vlm_model.generate_batch(images, prompts, max_new_tokens=self._get_max_tokens())
            
            # Process all responses
            results = []
            for i, response in enumerate(responses):
                result = self._process_response(response, i)
                results.append(result)
                
                # Log if sample_id provided
                if sample_ids and i < len(sample_ids) and sample_ids[i] is not None and self.logger:
                    image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                    true_label = true_labels[i] if true_labels and i < len(true_labels) else None
                    self.logger.log_baseline_inference(
                        sample_ids[i],
                        result.get("method", "Unknown"),
                        prompts[i],
                        response,
                        result["prediction"],
                        image_path=image_path,
                        true_label=true_label
                    )
            
            return results
        else:
            # Fallback to one-by-one processing
            results = []
            for i, image in enumerate(images):
                sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else None
                image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                true_label = true_labels[i] if true_labels and i < len(true_labels) else None
                results.append(self.classify(image, sample_id=sample_id, 
                                           image_path=image_path, true_label=true_label))
            return results
    
    def _get_prompt(self) -> str:
        """Get the prompt for this baseline method"""
        raise NotImplementedError
    
    def _get_max_tokens(self) -> int:
        """Get max tokens for generation"""
        return 200
    
    def _process_response(self, response: str, batch_idx: int = None) -> Dict[str, Any]:
        """Process the response from VLM"""
        raise NotImplementedError


class DirectGeneration(BaselineMethod):
    """Direct Generation baseline - VLM directly predicts class"""
    
    def _get_prompt(self) -> str:
        # Load prompt template from file
        template = self.prompt_manager.load_prompt("baseline/direct", "v1")
        
        # Format with class mapping
        class_mapping = '\n'.join([f"{i}: {cls}" for i, cls in enumerate(self.class_names)])
        num_classes_minus_1 = len(self.class_names) - 1
        
        return template.format(
            class_mapping=class_mapping,
            num_classes_minus_1=num_classes_minus_1
        )
    
    def _get_max_tokens(self) -> int:
        return 200  # Fixed token limit for all baselines
    
    def _process_response(self, response: str, batch_idx: int = None) -> Dict[str, Any]:
        numbers = re.findall(r'\d+', response)
        predicted_class = None
        
        for num in numbers:
            pred_idx = int(num)
            if 0 <= pred_idx < len(self.class_names):
                predicted_class = self.class_names[pred_idx]
                break
        
        # If no valid number found, default to first class
        if predicted_class is None:
            predicted_class = self.class_names[0]
            
        return {
            "method": "Direct Generation",
            "prediction": predicted_class,
            "raw_response": response
        }
    
    def _classify_internal(self, image: Image.Image) -> Dict[str, Any]:
        # save image
        # image.save(f"image_{self.current_sample_id}.png")
        # import pdb; pdb.set_trace()
        
        prompt = self._get_prompt()
        response = self.vlm_model.generate(image, prompt, max_new_tokens=self._get_max_tokens())
        result = self._process_response(response)
        
        # Log if available
        if self.logger and self.current_sample_id is not None:
            self.logger.log_baseline_inference(
                self.current_sample_id,
                "Direct Generation",
                prompt,
                response,
                result["prediction"],
                image_path=self.current_image_path,
                true_label=self.current_true_label
            )
            
        return result


class ChainOfThought(BaselineMethod):
    """Chain-of-Thought (CoT) baseline - step-by-step reasoning"""
    
    def _get_prompt(self) -> str:
        # Load prompt template from file
        template = self.prompt_manager.load_prompt("baseline/cot", "v1")
        
        # Format with class mapping
        class_mapping = '\n'.join([f"{i}: {cls}" for i, cls in enumerate(self.class_names)])
        num_classes_minus_1 = len(self.class_names) - 1
        
        return template.format(
            class_mapping=class_mapping,
            num_classes_minus_1=num_classes_minus_1
        )
    
    def _get_max_tokens(self) -> int:
        return 300  # Increased from 200 to avoid truncation
    
    def _process_response(self, response: str, batch_idx: int = None) -> Dict[str, Any]:
        reasoning = response
        if "final answer" in response.lower():
            reasoning = response.lower().split("final answer")[0].strip()
            number = response.lower().split("final answer")[1].strip()
        elif "Final answer" in response.lower():
            reasoning = response.lower().split("Final answer")[0].strip()
            number = response.lower().split("Final answer")[1].strip()
        else:
            number = None
            # print(response)
        
        if number is None:
            predicted_class = self.class_names[0]
        else:
            # Extract number from the response
            numbers = re.findall(r'\d+', number)
            if numbers:
                pred_idx = int(numbers[0])
                # Convert index to class name
                if 0 <= pred_idx < len(self.class_names):
                    predicted_class = self.class_names[pred_idx]
                else:
                    predicted_class = self.class_names[0]
            else:
                predicted_class = self.class_names[0]
                    
        return {
            "method": "Chain-of-Thought",
            "prediction": predicted_class,
            "reasoning": reasoning,
            "raw_response": response
        }
    
    def _classify_internal(self, image: Image.Image) -> Dict[str, Any]:
        # print("\n[DEBUG] ChainOfThought._classify_internal called!")
        prompt = self._get_prompt()
        response = self.vlm_model.generate(image, prompt, max_new_tokens=self._get_max_tokens())        
        # print(f"[DEBUG] CoT response: {response}")
        # import pdb; pdb.set_trace()
        result = self._process_response(response)
        
        # Log if available
        if self.logger and self.current_sample_id is not None:
            self.logger.log_baseline_inference(
                self.current_sample_id,
                "Chain-of-Thought",
                prompt,
                response,
                result["prediction"],
                image_path=self.current_image_path,
                true_label=self.current_true_label
            )
            
        return result


class SingleAgentVisualReasoning(BaselineMethod):
    """SAVR baseline - single handcrafted prompt for reasoning + classification"""
    
    def _get_prompt(self) -> str:
        # Load prompt template from file
        template = self.prompt_manager.load_prompt("baseline/savr", "v1")
        
        # Format with class mapping
        class_mapping = '\n'.join([f"{i}: {cls}" for i, cls in enumerate(self.class_names)])
        num_classes_minus_1 = len(self.class_names) - 1
        
        return template.format(
            class_mapping=class_mapping,
            num_classes_minus_1=num_classes_minus_1
        )
    
    def _get_max_tokens(self) -> int:
        return 300  # Fixed token limit for all baselines
    
    def _process_response(self, response: str, batch_idx: int = None) -> Dict[str, Any]:
        reasoning = ""
        
        if "<reasoning>" in response:
            reasoning = response.split("<reasoning>")[1]
            if "</reasoning>" in reasoning:
                reasoning = reasoning.split("</reasoning>")[0].strip()
        else:
            reasoning = response

        if "<answer>" in response:
            answer = response.split("<answer>")[1]
            if "</answer>" in answer:
                answer = answer.split("</answer>")[0].strip()
        else:
            answer = response
        
        numbers = re.findall(r'\d+', answer)

        if numbers:
            pred_idx = int(numbers[0])
            if 0 <= pred_idx < len(self.class_names):
                predicted_class = self.class_names[pred_idx]
            else:
                predicted_class = self.class_names[0]
        else:
            predicted_class = self.class_names[0]
                    
        return {
            "method": "SAVR",
            "prediction": predicted_class,
            "reasoning": reasoning,
            "raw_response": response
        }
    
    def _classify_internal(self, image: Image.Image) -> Dict[str, Any]:
        prompt = self._get_prompt()
        response = self.vlm_model.generate(image, prompt, max_new_tokens=self._get_max_tokens())
        result = self._process_response(response)
        
        # Log if available
        if self.logger and self.current_sample_id is not None:
            self.logger.log_baseline_inference(
                self.current_sample_id,
                "SAVR",
                prompt,
                response,
                result["prediction"],
                image_path=self.current_image_path,
                true_label=self.current_true_label
            )
            
        return result


def create_baseline_method(method_name: str, vlm_model: VLMModel, 
                          class_names: List[str], logger: Optional[ExperimentLogger] = None, **kwargs) -> BaselineMethod:
    """Factory function to create baseline methods"""
    
    methods = {
        "direct": DirectGeneration,
        "cot": ChainOfThought,
        "savr": SingleAgentVisualReasoning
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")
        
    method_class = methods[method_name]
    
    return method_class(vlm_model, class_names, logger=logger)