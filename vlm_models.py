import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class VLMModel(ABC):
    """Abstract base class for Vision-Language Models"""
    
    @abstractmethod
    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200) -> str:
        pass
    
    @abstractmethod
    def load_model(self):
        pass
        
    def generate_batch(self, images: List[Optional[Image.Image]], prompts: List[str], max_new_tokens: int = 200) -> List[str]:
        """Default batch generation - process one by one"""
        results = []
        for image, prompt in zip(images, prompts):
            results.append(self.generate(image, prompt, max_new_tokens))
        return results


class LLaVAModel(VLMModel):
    """LLaVA model wrapper"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = None):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        print(f"Loading LLaVA model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Set padding side to left for decoder-only models
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        
        # Determine dtype based on device
        if self.device != "cpu":
            # Force float16 instead of bfloat16 to avoid triu_tril_cuda_template error
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        if self.device != "cpu":
            self.model = self.model.to(self.device, dtype=dtype)
            print(f"LLaVA model loaded on {self.device} with dtype {dtype}")
            # Ensure all model parameters have the correct dtype
            for param in self.model.parameters():
                param.data = param.data.to(dtype=dtype)
        
    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200) -> str:
        if image is not None:
            # Create conversation format for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            # Apply chat template
            formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=formatted_prompt,
                return_tensors="pt"
            )
            
            # Move to device with appropriate dtype
            if self.device != "cpu":
                # Get model dtype
                model_dtype = next(self.model.parameters()).dtype
                # Convert each tensor in inputs to the correct device and dtype
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        if key == 'pixel_values' or inputs[key].dtype.is_floating_point:
                            inputs[key] = inputs[key].to(self.device, dtype=model_dtype)
                        else:
                            inputs[key] = inputs[key].to(self.device)
        else:
            print("Text-only generation")
            # Text-only generation
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
            formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            print(f"Formatted prompt: {formatted_prompt}")
            
            inputs = self.processor(
                text=formatted_prompt,
                images=None,
                return_tensors="pt"
            )
            if self.device != "cpu":
                # Get model dtype
                model_dtype = next(self.model.parameters()).dtype
                # Convert each tensor in inputs to the correct device and dtype
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        if key == 'pixel_values' or inputs[key].dtype.is_floating_point:
                            inputs[key] = inputs[key].to(self.device, dtype=model_dtype)
                        else:
                            inputs[key] = inputs[key].to(self.device)
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for more consistent results
                    # repetition_penalty=1.2,  # Prevent repetitive text
                    # no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
                )
            except RuntimeError as e:
                error_msg = str(e)
                if "dtype" in error_msg or "expected scalar type" in error_msg:
                    # Debug: print current dtypes
                    print(f"Dtype error encountered: {error_msg}")
                    model_dtype = next(self.model.parameters()).dtype
                    print(f"Model dtype: {model_dtype}")
                    for key in inputs:
                        if hasattr(inputs[key], 'dtype'):
                            print(f"  {key} dtype: {inputs[key].dtype}")
                    
                    # Fallback: convert all inputs to model dtype more aggressively
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to(dtype=model_dtype)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        # repetition_penalty=1.2,
                        # no_repeat_ngram_size=3
                    )
                else:
                    raise
        
        # Decode response - skip the first 2 tokens as in the example
        response = self.processor.decode(outputs[0][2:], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
            
        return response
    
    def generate_batch(self, images: List[Optional[Image.Image]], prompts: List[str], max_new_tokens: int = 200) -> List[str]:
        """True batch generation for LLaVA"""
        if not images or not prompts:
            return []
        
        batch_size = len(images)
        conversations = []
        
        # Prepare all conversations
        for i in range(batch_size):
            if images[i] is not None:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts[i]},
                            {"type": "image"},
                        ],
                    },
                ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompts[i]}],
                    },
                ]
            conversations.append(conversation)
        
        # Apply chat template to all
        formatted_prompts = [self.processor.apply_chat_template(conv, add_generation_prompt=True) 
                            for conv in conversations]
        
        # Process all inputs together
        inputs = self.processor(
            images=[img for img in images if img is not None] if any(images) else None,
            text=formatted_prompts,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device with appropriate dtype
        if self.device != "cpu":
            # Get model dtype
            model_dtype = next(self.model.parameters()).dtype
            # Convert each tensor in inputs to the correct device and dtype
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    if key == 'pixel_values' or inputs[key].dtype.is_floating_point:
                        inputs[key] = inputs[key].to(self.device, dtype=model_dtype)
                    else:
                        inputs[key] = inputs[key].to(self.device)
        
        # Generate for all inputs at once
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                    num_beams=1,     # Use greedy decoding
                    # repetition_penalty=1.2,
                    # no_repeat_ngram_size=3,
                )
            except RuntimeError as e:
                error_msg = str(e)
                if "dtype" in error_msg or "expected scalar type" in error_msg:
                    # Fallback: convert all inputs to model dtype more aggressively
                    model_dtype = next(self.model.parameters()).dtype
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to(dtype=model_dtype)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,  # Enable KV cache for faster generation
                        num_beams=1,     # Use greedy decoding
                        # repetition_penalty=1.2,
                        # no_repeat_ngram_size=3
                    )
                else:
                    raise
        
        # Decode all responses
        responses = []
        for i in range(len(outputs)):
            response = self.processor.decode(outputs[i][2:], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            elif "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
            responses.append(response)
        
        return responses


def simple_process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[Optional[List[Image.Image]], None]:
    """
    Simplified version of process_vision_info for Qwen models
    Only handles single images, not videos
    """
    image_inputs = []
    
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # For now, we'll only handle PIL Images passed directly
                        # In production, you'd want to handle URLs and file paths too
                        if "image" in item and isinstance(item["image"], Image.Image):
                            image_inputs.append(item["image"])
    
    return image_inputs if image_inputs else None, None

def create_vlm_model(model_type: str, device: str = None) -> VLMModel:
    """Factory function to create VLM models mentioned in the MARIC paper"""
    
    if model_type == "llava-7b":
        return LLaVAModel("llava-hf/llava-1.5-7b-hf", device=device)
    elif model_type == "llava-13b":
        return LLaVAModel("llava-hf/llava-1.5-13b-hf", device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available models: llava-7b, llava-13b)