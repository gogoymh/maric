import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import re
from abc import ABC, abstractmethod
from prompt_manager import PromptManager
from experiment_logger import ExperimentLogger


class BaseAgent(ABC):
    """Base class for all agents in MARIC framework"""
    
    def __init__(self, vlm_model=None, prompt_manager=None, prompt_version="v1"):
        self.vlm_model = vlm_model
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompt_version = prompt_version
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class OutlinerAgent(BaseAgent):
    """Outliner Agent (G_out): Captures holistic theme and generates prompts"""
    
    def __init__(self, vlm_model=None, prompt_manager=None, prompt_version="v1", class_names: List[str] = None):
        super().__init__(vlm_model, prompt_manager, prompt_version)
        self.class_names = class_names
    
    def forward(self, image: Image.Image) -> List[Dict[str, str]]:
        """
        Generate prompts for aspect agents based on global theme analysis
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of prompts with prefix and postfix structure
        """
        # Load prompt from file
        # print('theme_prompt: ')
        theme_prompt = self.prompt_manager.get_outliner_prompt(self.class_names, self.prompt_version)
        # print(f"[OutlinerAgent] Using prompt version: {self.prompt_version}")  # Debug
        # print(f"[OutlinerAgent] Classes: {self.class_names}")  # Debug
        # print(f"[OutlinerAgent] Classes type: {type(self.class_names)}")  # Debug
        
        # Generate response from VLM
        # print("response: ")
        response = self.vlm_model.generate(image, theme_prompt, max_new_tokens=300)
        # print(f"Outliner response: {response[:500]}...")  # Debug: Show first 500 chars
        
        # Parse response to extract aspects
        # print("prompts: ")
        prompts = self._parse_aspects(response)
        # print(prompts)
        
        return prompts
    
    def _parse_aspects(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response to extract structured prompts"""
        import json
        
        # Check for common failure patterns
        failure_patterns = [
            "i am ready to", "i will analyze", "please provide", 
            "let me help", "i can assist", "to analyze"
        ]
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in failure_patterns):
            # Return default prompts on generic response
            return [
                {"prefix": "Focus on the main subject in the center", "postfix": "Describe its shape, size and key features"},
                {"prefix": "Focus on the background and surrounding area", "postfix": "Describe the environment and context"},
                {"prefix": "Focus on colors, textures and patterns", "postfix": "Describe the visual appearance and surface details"}
            ]
        
        # Try to extract JSON from response
        # Look for JSON array in the response
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            # Try to find curly braces as fallback
            if '{' in response and '}' in response:
                # Extract individual JSON objects
                objects = []
                for match in re.finditer(r'\{[^{}]*\}', response):
                    try:
                        obj = json.loads(match.group())
                        if 'prefix' in obj and 'postfix' in obj:
                            objects.append(obj)
                    except:
                        pass
                if len(objects) >= 3:
                    return objects[:3]
            
            # If still no valid JSON, return default prompts
            # print(f"Warning: No valid JSON found in outliner response. Using defaults.")
            return [
                {"prefix": "Focus on the main subject in the center", "postfix": "Describe its shape, size and key features"},
                {"prefix": "Focus on the background and surrounding area", "postfix": "Describe the environment and context"},
                {"prefix": "Focus on colors, textures and patterns", "postfix": "Describe the visual appearance and surface details"}
            ]
        
        json_str = response[start_idx:end_idx+1]
        
        try:
            prompts = json.loads(json_str)
        except json.JSONDecodeError as e:
            # print(f"Warning: JSON decode error: {e}. Using defaults.")
            return [
                {"prefix": "Focus on the main subject in the center", "postfix": "Describe its shape, size and key features"},
                {"prefix": "Focus on the background and surrounding area", "postfix": "Describe the environment and context"},
                {"prefix": "Focus on colors, textures and patterns", "postfix": "Describe the visual appearance and surface details"}
            ]
        
        # Validate that we have the expected structure
        if not isinstance(prompts, list):
            # print(f"Warning: Response is not a list: {type(prompts)}. Using defaults.")
            return [
                {"prefix": "Focus on the main subject in the center", "postfix": "Describe its shape, size and key features"},
                {"prefix": "Focus on the background and surrounding area", "postfix": "Describe the environment and context"},
                {"prefix": "Focus on colors, textures and patterns", "postfix": "Describe the visual appearance and surface details"}
            ]
        
        if len(prompts) < 3:
            # print(f"Warning: Response has less than 3 prompts: {len(prompts)}. Using defaults.")
            return [
                {"prefix": "Focus on the main subject in the center", "postfix": "Describe its shape, size and key features"},
                {"prefix": "Focus on the background and surrounding area", "postfix": "Describe the environment and context"},
                {"prefix": "Focus on colors, textures and patterns", "postfix": "Describe the visual appearance and surface details"}
            ]
        
        # Validate each prompt has required fields
        valid_prompts = []
        for i, prompt in enumerate(prompts[:3]):
            if isinstance(prompt, dict) and 'prefix' in prompt and 'postfix' in prompt:
                valid_prompts.append(prompt)
            else:
                print(f"Warning: Invalid prompt structure at index {i}. Using default.")
                valid_prompts.append({
                    "prefix": f"Focus on aspect {i+1}",
                    "postfix": "Describe relevant details"
                })
        
        return valid_prompts


class AspectAgent(BaseAgent):
    """Aspect Agent (G_asp): Generates fine-grained descriptions based on prompts"""
    
    def __init__(self, vlm_model=None, prompt_manager=None, prompt_version="v1", agent_id: int = 1, class_names: List[str] = None):
        super().__init__(vlm_model, prompt_manager, prompt_version)
        self.agent_id = agent_id
        self.class_names = class_names
        
    def forward(self, image: Image.Image, prompt: Dict[str, str]) -> str:
        """
        Generate aspect-specific description based on prompt
        
        Args:
            image: Input PIL Image
            prompt: Dictionary with 'prefix' and 'postfix' keys
            
        Returns:
            Detailed description focusing on the specified aspect
        """
        # Get prompt template and format it
        # print(f"[AspectAgent {self.agent_id}] Using prompt version: {self.prompt_version}")
        # print(f"[AspectAgent {self.agent_id}] Classes: {self.class_names}")
        full_prompt = self.prompt_manager.get_aspect_prompt(
            prompt['prefix'], prompt['postfix'], self.prompt_version, self.class_names
        )
        # print("full_prompt: ", full_prompt)
        
        response = self.vlm_model.generate(image, full_prompt, max_new_tokens=300)
        # print("response: ", response)
        # import pdb; pdb.set_trace()

        # Parse response to extract description
        # description = self._parse_description(response)
        description = response
        # print("description: ", description)
            
        return description

    def _parse_description(self, response: str) -> str:
        """Parse description from response"""
        import json
        
        # Check for common failure patterns
        failure_patterns = [
            "i am ready to", "i will analyze", "please provide", 
            "let me help", "i can assist", "to generate"
        ]
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in failure_patterns):
            # Return a generic description on failure
            return "Unable to generate specific description. The image shows visual elements that need classification."
        
        # Try to extract JSON object from response
        # Look for JSON object in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            # Try to extract description from quotes as fallback
            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', response)
            if desc_match:
                return desc_match.group(1)
            
            # If no JSON, just return the response cleaned up
            cleaned = response.strip()
            if cleaned:
                return cleaned
            else:
                return "Visual features present in the focused area."
        
        json_str = response[start_idx:end_idx+1]
        
        try:
            description_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract description pattern
            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', json_str)
            if desc_match:
                return desc_match.group(1)
            return "Visual features present in the focused area."
        
        if 'description' not in description_obj:
            # If no description field, return the whole response
            return str(description_obj)
        
        return description_obj['description']


class ReasoningAgent(BaseAgent):
    """Reasoning Agent (G_rea): Synthesizes descriptions and performs classification"""
    
    def __init__(self, vlm_model=None, prompt_manager=None, prompt_version="v1", class_names: List[str] = None):
        super().__init__(vlm_model, prompt_manager, prompt_version)
        self.class_names = class_names
        self.current_focuses = None 
            
    def forward(self, image: Image.Image, descriptions: List[str], focuses: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform reasoning and classification based on aspect descriptions
        
        Args:
            descriptions: List of descriptions from aspect agents
            focuses: Optional list of aspect focuses
            
        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        # Check if using Qwen model and adjust prompt version accordingly
        prompt_version = self.prompt_version
        
        # Get formatted prompt
        prompt = self.prompt_manager.get_reasoning_prompt(
            descriptions, self.class_names, prompt_version, focuses
        )
        # print("reasoning prompt: ", prompt)
        
        # Use VLM model with dummy image for reasoning
        response = self.vlm_model.generate(image=image, prompt=prompt, max_new_tokens=500)
        # print("reasoning response: ", response)
        
        # Parse response
        result = self._parse_reasoning_response(response)
        
        # Add raw response for debugging
        result["raw_response"] = response
        
        return result
    
    def _extract_focuses_from_prompts(self, prompts: List[Dict[str, str]]) -> List[str]:
        """Extract focus areas from prompts for reasoning"""
        focuses = []
        for prompt in prompts:
            # Extract the focus area from the prefix
            # The prefix typically contains the visual aspect to focus on
            prefix = prompt.get('prefix', '')
            postfix = prompt.get('postfix', '')
            
            # Combine prefix and postfix to get full context
            full_prompt = f"{prefix} {postfix}".strip()
            
            # Simply use the prefix as the focus area
            # The prefix already contains the specific aspect like "Focus on the cat's face"
            # We just need to clean it up slightly
            focus = prefix.strip()
            
            # If empty, use a default
            if not focus:
                focus = f"Aspect {len(focuses) + 1}"
                
            focuses.append(focus)
        
        return focuses
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse reasoning and answer from response"""
        result = {"reasoning": "", "answer": ""}

        # Try XML-like format first
        if "<answer>" in response:
            reasoning = response.split("<answer>")[0]
            answer_text = response.split("<answer>")[1]
            if "</answer>" in answer_text:
                answer_text = answer_text.split("</answer>")[0].strip()
        # Try non-XML format for Qwen models
        elif "ANSWER:" in response:
            reasoning = response.split("ANSWER")[0]
            answer_text = response.split("ANSWER")[1]
        else:
            reasoning = response
            answer_text = response

        # Find number in answer
        numbers = re.findall(r'\d+', answer_text)
        
        if numbers:
            answer_idx = int(numbers[0])
            if 0 <= answer_idx < len(self.class_names):
                answer_class = self.class_names[answer_idx]
            else:
                # Invalid index, default to first class
                answer_class = self.class_names[0]
                answer_idx = 0
        else:
            # No number found, default to first class
            answer_class = self.class_names[0]
            answer_idx = 0

        result["reasoning"] = reasoning
        result["answer"] = answer_class  # This is what the rest of the code expects
        result["answer_by_number"] = answer_idx
        result["answer_by_name"] = answer_class

        return result


class MARIC:
    """Main MARIC framework combining all agents"""
    
    def __init__(self, 
                 vlm_model,
                 class_names: List[str] = None,
                 prompt_versions: Optional[Dict[str, str]] = None,
                 num_aspect_agents: int = 3,
                 logger: Optional[ExperimentLogger] = None):
        
        self.class_names = class_names
        self.vlm_model = vlm_model
        self.prompt_manager = PromptManager()
        self.logger = logger
        self.current_sample_id = None
        
        # Default prompt versions
        if prompt_versions is None:
            prompt_versions = {
                'outliner': 'v1',
                'aspect': 'v1',
                'reasoning': 'v1'
            }
        self.prompt_versions = prompt_versions
        
        print(f"[MARIC] Initializing with classes: {class_names}")  # Debug
        print(f"[MARIC] Prompt versions: {prompt_versions}")  # Debug
        
        # Initialize agents with shared VLM model and prompt manager
        self.outliner_agent = OutlinerAgent(
            vlm_model, self.prompt_manager, prompt_versions['outliner'], class_names
        )
        self.aspect_agents = [
            AspectAgent(
                vlm_model, self.prompt_manager, prompt_versions['aspect'], agent_id=i, class_names=class_names
            ) for i in range(num_aspect_agents)
        ]
        self.reasoning_agent = ReasoningAgent(
            vlm_model, self.prompt_manager, prompt_versions['reasoning'], class_names
        )
        
    def classify(self, image: Image.Image, sample_id: Optional[int] = None, 
                 image_path: Optional[str] = None, true_label: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform image classification using multi-agent reasoning
        
        Args:
            image: Input PIL Image
            sample_id: Optional sample ID for logging
            image_path: Optional path to the image
            true_label: Optional true label for logging
            
        Returns:
            Dictionary containing reasoning process and final prediction
        """
        # Set sample_id for logging
        if sample_id is not None:
            self.current_sample_id = sample_id
        self.current_image_path = image_path
        self.current_true_label = true_label
        # Step 1: Outliner Agent generates prompts
        prompts = self.outliner_agent.forward(image)
        
        # Log outliner output if logger is available
        if self.logger and self.current_sample_id is not None:
            self.logger.log_maric_step(
                self.current_sample_id,
                "outliner_agent",
                {
                    "prompts": prompts,
                    "prompt_used": self.prompt_manager.get_outliner_prompt(
                        self.class_names,
                        self.prompt_versions['outliner']
                    )
                }
            )
        
        # Step 2: Aspect Agents generate descriptions
        descriptions = []
        for i, (agent, prompt) in enumerate(zip(self.aspect_agents, prompts)):
            description = agent.forward(image, prompt)
            descriptions.append(description)
            
            # Log aspect agent output
            if self.logger and self.current_sample_id is not None:
                self.logger.log_maric_step(
                    self.current_sample_id,
                    f"aspect_agent_{i}",
                    {
                        "prompt": prompt,
                        "description": description,
                        "full_prompt": self.prompt_manager.get_aspect_prompt(
                            prompt.get('prefix', ''),
                            prompt.get('postfix', ''),
                            self.prompt_versions['aspect'],
                            self.class_names
                        )
                    }
                )
            
        # Step 3: Reasoning Agent performs classification
        # print("prompts: ", prompts)
        focuses = self.reasoning_agent._extract_focuses_from_prompts(prompts)
        # print("focuses: ", focuses)
            
        result = self.reasoning_agent.forward(image, descriptions, focuses)
        # print(descriptions, focuses)
        
        # Log reasoning agent output
        if self.logger and self.current_sample_id is not None:
            step_data = {
                "focuses": focuses,
                "descriptions": descriptions,
                "result": result,
                "prompt_used": self.prompt_manager.get_reasoning_prompt(
                    descriptions=descriptions,
                    classes=self.class_names,
                    version=self.prompt_versions['reasoning'],
                    focuses=focuses
                )
            }
            
            # Add prediction info if true_label is available
            if self.current_true_label is not None:
                step_data["predicted_class"] = result.get("answer", "")
                step_data["true_label"] = self.current_true_label
                step_data["is_correct"] = result.get("answer", "").lower() == self.current_true_label.lower()
                if self.current_image_path is not None:
                    step_data["image_path"] = self.current_image_path
            
            self.logger.log_maric_step(
                self.current_sample_id,
                "reasoning_agent",
                step_data
            )
        
        # Add additional information
        result["prompts"] = prompts
        result["descriptions"] = descriptions
        result["focuses"] = focuses
        result["answer"] = result["answer"].lower()
        # print("result: ", result)
        
        return result
    
    def predict(self, image: Image.Image, sample_id: Optional[int] = None) -> str:
        """Simple prediction interface returning only the class name"""
        self.current_sample_id = sample_id
        result = self.classify(image)
        return result["answer"]
        
    def predict_batch(self, images: List[Image.Image], sample_ids: Optional[List[Optional[int]]] = None) -> List[str]:
        """Batch prediction interface"""
        results = []
        for i, image in enumerate(images):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else None
            results.append(self.predict(image, sample_id=sample_id))
        return results
    
    def classify_batch(self, images: List[Image.Image], sample_ids: Optional[List[Optional[int]]] = None,
                      image_paths: Optional[List[Optional[str]]] = None, 
                      true_labels: Optional[List[Optional[str]]] = None) -> List[Dict[str, Any]]:
        """True batch classification using VLM's batch processing capabilities"""
        batch_size = len(images)
        
        # Step 1: Outliner Agent - Process all images in batch
        all_prompts = []
        outliner_responses = []
        
        # Generate outliner prompts for all images at once
        outliner_prompt = self.prompt_manager.get_outliner_prompt(self.class_names, self.prompt_versions['outliner'])
        outliner_prompts = [outliner_prompt] * batch_size
        
        # Use VLM's batch generation for outliner
        outliner_raw_responses = self.vlm_model.generate_batch(images, outliner_prompts, max_new_tokens=300)
        
        # Parse outliner responses
        for i, response in enumerate(outliner_raw_responses):
            prompts = self.outliner_agent._parse_aspects(response)
            all_prompts.append(prompts)
            outliner_responses.append(response)
            
            # Log outliner output if sample_id provided
            if sample_ids and i < len(sample_ids) and sample_ids[i] is not None and self.logger:
                self.logger.log_maric_step(
                    sample_ids[i],
                    "outliner_agent",
                    {
                        "prompts": prompts,
                        "prompt_used": outliner_prompt
                    }
                )
        
        # Step 2: Aspect Agents - Process all aspects in batches
        all_descriptions = [[] for _ in range(batch_size)]
        
        # Process each aspect agent across all images
        for agent_idx in range(len(self.aspect_agents)):
            # Collect prompts for this aspect agent across all images
            aspect_images = []
            aspect_prompts = []
            image_indices = []
            
            for img_idx, prompts in enumerate(all_prompts):
                if agent_idx < len(prompts):
                    prompt = prompts[agent_idx]
                    full_prompt = self.prompt_manager.get_aspect_prompt(
                        prompt['prefix'], prompt['postfix'], 
                        self.prompt_versions['aspect'], self.class_names
                    )
                    aspect_images.append(images[img_idx])
                    aspect_prompts.append(full_prompt)
                    image_indices.append(img_idx)
            
            # Batch generate descriptions for this aspect across all images
            if aspect_prompts:
                aspect_responses = self.vlm_model.generate_batch(aspect_images, aspect_prompts, max_new_tokens=300)
                
                # Parse and store descriptions
                for resp_idx, (response, img_idx) in enumerate(zip(aspect_responses, image_indices)):
                    description = self.aspect_agents[agent_idx]._parse_description(response)
                    all_descriptions[img_idx].append(description)
                    
                    # Log aspect agent output
                    if sample_ids and img_idx < len(sample_ids) and sample_ids[img_idx] is not None and self.logger:
                        self.logger.log_maric_step(
                            sample_ids[img_idx],
                            f"aspect_agent_{agent_idx}",
                            {
                                "prompt": all_prompts[img_idx][agent_idx],
                                "description": description,
                                "full_prompt": aspect_prompts[resp_idx]
                            }
                        )
        
        # Step 3: Reasoning Agent - Process all reasoning in batch
        reasoning_prompts = []
        all_focuses = []
        
        for img_idx in range(batch_size):
            focuses = self.reasoning_agent._extract_focuses_from_prompts(all_prompts[img_idx])
            all_focuses.append(focuses)
            
            # Check if using Qwen model and adjust prompt version accordingly
            prompt_version = self.prompt_versions['reasoning']
            
            reasoning_prompt = self.prompt_manager.get_reasoning_prompt(
                all_descriptions[img_idx], self.class_names, 
                prompt_version, focuses
            )
            reasoning_prompts.append(reasoning_prompt)
        
        # Batch generate reasoning
        reasoning_responses = self.vlm_model.generate_batch(images, reasoning_prompts, max_new_tokens=500)
        
        # Parse results
        results = []
        for img_idx, response in enumerate(reasoning_responses):
            result = self.reasoning_agent._parse_reasoning_response(response)
            result["raw_response"] = response
            result["prompts"] = all_prompts[img_idx]
            result["descriptions"] = all_descriptions[img_idx]
            result["focuses"] = all_focuses[img_idx]
            result["answer"] = result["answer"].lower()
            
            # Log reasoning agent output
            if sample_ids and img_idx < len(sample_ids) and sample_ids[img_idx] is not None and self.logger:
                step_data = {
                    "focuses": all_focuses[img_idx],
                    "descriptions": all_descriptions[img_idx],
                    "result": result,
                    "prompt_used": reasoning_prompts[img_idx]
                }
                
                # Add prediction info if true_label is available
                if true_labels and img_idx < len(true_labels) and true_labels[img_idx] is not None:
                    step_data["predicted_class"] = result["answer"]
                    step_data["true_label"] = true_labels[img_idx]
                    step_data["is_correct"] = result["answer"].lower() == true_labels[img_idx].lower()
                
                self.logger.log_maric_step(
                    sample_ids[img_idx],
                    "reasoning_agent",
                    step_data
                )
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # CIFAR-10 inference example
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    from vlm_models import create_vlm_model
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # CIFAR-10 class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Number of samples to test
    num_samples = 10  # Change to larger number for more comprehensive testing
    
    # Initialize VLM model
    vlm_model = create_vlm_model("llava-7b")
    
    # Create logger for experiment tracking
    from experiment_logger import ExperimentLogger
    logger = ExperimentLogger(log_dir="logs/maric_direct_test", max_samples_to_log=10)
    
    # Start experiment
    experiment_config = {
        'dataset': 'cifar10',
        'vlm_model': 'llava-7b',
        'num_samples': num_samples,
        'prompt_versions': {
            'outliner': 'v1',
            'aspect': 'v1',
            'reasoning': 'v1'
        }
    }
    logger.start_experiment("cifar10_maric_direct", experiment_config)
    
    # Initialize MARIC framework with logger
    maric = MARIC(
        vlm_model=vlm_model,
        class_names=cifar10_classes,
        prompt_versions={
            'outliner': 'v1',
            'aspect': 'v1',
            'reasoning': 'v1'
        },
        logger=logger
    )
    
    # Load CIFAR-10 test dataset with transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to standard input size
        transforms.ToTensor()
    ])
    
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Convert tensor to PIL for VLM processing
    to_pil = transforms.ToPILImage()
    
    # Test on entire test dataset
    print("\nMARIC Framework - CIFAR-10 Full Test Set Evaluation\n")
    print("=" * 60)
    
    # Initialize tracking variables
    correct_predictions = 0
    total_predictions = 0
    class_correct = {cls: 0 for cls in cifar10_classes}
    class_total = {cls: 0 for cls in cifar10_classes}
    
    num_samples = len(cifar10_test)
    print(f"Testing on {num_samples} images...\n")
    
    for i in range(num_samples):
        image_tensor, true_label = cifar10_test[i]
        # Convert tensor to PIL Image
        image_pil = to_pil(image_tensor)
        true_class = cifar10_classes[true_label]
        
        # Update class total
        class_total[true_class] += 1
        total_predictions += 1
        
        try:
            # Determine if we should log this sample
            should_log = logger.mark_sample_for_logging()
            sample_id = i if should_log else None
            
            # Perform classification with optional logging
            result = maric.classify(image_pil, sample_id=sample_id)
            predicted_class = result['answer']
            
            # Get predicted class index
            pred_idx = cifar10_classes.index(predicted_class) if predicted_class in cifar10_classes else -1
            
            # Track accuracy with flexible matching
            # Normalize both strings for comparison
            pred_normalized = predicted_class.lower().strip()
            true_normalized = true_class.lower().strip()

            if predicted_class == "unknown":
                if true_class.lower() in result["raw_response"].lower():
                    predicted_class = true_class
                    pred_idx = cifar10_classes.index(predicted_class)
                    is_correct = True
                    # print("unknown & true class in raw response", true_class)
                else:
                    # print("unknown & true class not in raw response", true_class)
                    is_correct = False
            else:
                if true_class.lower() in predicted_class.lower():
                    is_correct = True
                    # print("known & true class in predicted class", true_class)
                else:
                    if true_class.lower() in result["raw_response"].lower():
                        predicted_class = true_class
                        pred_idx = cifar10_classes.index(predicted_class)
                        is_correct = True
                        # print("known & true class in raw response", true_class)
                    else:
                        # print("known & true class not in raw response", true_class)
                        is_correct = False
            
            if is_correct:
                correct_predictions += 1
                class_correct[true_class] += 1
                status = "✓"
            else:
                status = "✗"
            
            # Print detailed result for this image
            print(f"\n[Image {i+1}] {status} Predicted: {predicted_class} ({pred_idx}), Actual: {true_class} ({true_label})")
            
            # Log prediction results if this sample was logged
            if should_log and sample_id is not None:
                prediction_data = {
                    "predicted_class": predicted_class,
                    "predicted_index": pred_idx,
                    "true_class": true_class,
                    "true_index": true_label,
                    "is_correct": is_correct,
                    "reasoning": result.get("reasoning", ""),
                    "raw_response": result.get("raw_response", ""),
                    "focuses": result.get("focuses", []),
                    "descriptions": result.get("descriptions", [])
                }
                logger.log_prediction_result(sample_id, prediction_data)
                
            # Always save incorrect cases immediately
            if not is_correct and logger is not None:
                incorrect_data = {
                    "predicted_class": predicted_class,
                    "predicted_index": pred_idx,
                    "true_class": true_class,
                    "true_index": true_label,
                    "is_correct": is_correct,
                    "reasoning": result.get("reasoning", ""),
                    "raw_response": result.get("raw_response", ""),
                    "focuses": result.get("focuses", []),
                    "descriptions": result.get("descriptions", []),
                    "prompts": result.get("prompts", [])
                }
                
                # Extract MARIC steps from the result
                maric_steps = {
                    "outliner_agent": {
                        "prompts": result.get("prompts", [])
                    },
                    "aspect_agents": [
                        {
                            "prompt": prompt,
                            "description": desc
                        }
                        for prompt, desc in zip(result.get("prompts", []), result.get("descriptions", []))
                    ],
                    "reasoning_agent": {
                        "focuses": result.get("focuses", []),
                        "descriptions": result.get("descriptions", []),
                        "reasoning": result.get("reasoning", ""),
                        "answer": result.get("answer", "")
                    }
                }
                
                logger.save_incorrect_case(i, incorrect_data, maric_steps)
                
        except Exception as e:
            print(f"\n[Image {i+1}] ✗ Error: {e}")
            
        # Update progress on same line
        if i < num_samples - 1:  # Don't print progress on last iteration
            print(f"Processing... Current accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions*100:.2f}%)", end='', flush=True)
    
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print(f"\n\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal images processed: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Per-class accuracy
    print("\n" + "-" * 40)
    print("PER-CLASS ACCURACY:")
    print("-" * 40)
    print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 40)
    
    for cls in cifar10_classes:
        if class_total[cls] > 0:
            class_acc = (class_correct[cls] / class_total[cls]) * 100
            print(f"{cls:<15} {class_correct[cls]:<10} {class_total[cls]:<10} {class_acc:<10.2f}%")
        else:
            print(f"{cls:<15} {0:<10} {0:<10} {'N/A':<10}")
    
    print("=" * 60)
    
    # Save experiment logs
    logger.save_experiment()
    print(f"\nLogs saved to: logs/maric_direct_test/")