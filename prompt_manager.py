import os
from typing import Dict, Any, Optional
import json


class PromptManager:
    """Manages prompts from files for different agents and versions"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}
        
    def load_prompt(self, agent_name: str, version: str = "v1") -> str:
        """Load prompt from file for specific agent and version"""
        # Normalize agent_name to handle any unicode or whitespace issues
        agent_name = agent_name.strip()
        cache_key = f"{agent_name}_{version}"
        
        # Check cache first
        if cache_key in self.prompts_cache:
            return self.prompts_cache[cache_key]
        
        # Load from file
        # Check if agent_name contains any path separator
        has_separator = "/" in agent_name or "\\" in agent_name or os.sep in agent_name
        
        if has_separator:
            # Handle paths like "baseline/cot" -> "prompts/baseline/cot_v1.txt"
            # Normalize path separators
            agent_name_normalized = agent_name.replace("\\", "/").replace(os.sep, "/")
            parts = agent_name_normalized.split("/")
            
            if len(parts) == 2:
                # Expected format: "baseline/cot"
                dir_part, name_part = parts
                filename = f"{name_part}_{version}.txt"
                filepath = os.path.join(self.prompts_dir, dir_part, filename)
            else:
                # Unexpected format - just use the last part as name
                name_part = parts[-1]
                dir_part = "/".join(parts[:-1])
                filename = f"{name_part}_{version}.txt"
                filepath = os.path.join(self.prompts_dir, dir_part, filename)
        else:
            # Handle paths like "outliner_agent" -> "prompts/outliner_agent/outliner_agent_v1.txt"
            filename = f"{agent_name}_{version}.txt"
            filepath = os.path.join(self.prompts_dir, agent_name, filename)
        
        if not os.path.exists(filepath):
            # Provide more helpful error with actual vs expected path
            if "/" in agent_name:
                expected = os.path.join(self.prompts_dir, agent_name.split("/")[0], f"{agent_name.split('/')[1]}_{version}.txt")
                raise ValueError(f"Prompt file not found: {filepath}\nExpected location: {expected}\nAgent name received: '{agent_name}'")
            else:
                raise ValueError(f"Prompt file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            prompt_template = f.read().strip()
            
        # Cache the prompt
        self.prompts_cache[cache_key] = prompt_template
        
        return prompt_template
    
    
    def format_reasoning_prompt(self, template: str, descriptions: list, classes: list, 
                               version: str = "v1", focuses: Optional[list] = None) -> str:
        """Special formatting for reasoning prompt to include class numbers"""
        # Create numbered class list
        classes_with_numbers = '\n'.join([f"{i}: {cls}" for i, cls in enumerate(classes)])
        num_classes_minus_1 = len(classes) - 1
        
        return self.format_prompt(
            template,
            focus1=focuses[0] if focuses else "",
            description1=descriptions[0] if descriptions else "",
            focus2=focuses[1] if focuses and len(focuses) > 1 else "",
            description2=descriptions[1] if len(descriptions) > 1 else "",
            focus3=focuses[2] if focuses and len(focuses) > 2 else "",
            description3=descriptions[2] if len(descriptions) > 2 else "",
            classes=", ".join(classes),
            classes_with_numbers=classes_with_numbers,
            num_classes_minus_1=num_classes_minus_1
        )
    
    def format_prompt(self, prompt_template: str, **kwargs) -> str:
        """Format prompt template with provided variables"""
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}")
    
    def get_outliner_prompt(self, classes: list = None, version: str = "v1") -> str:
        """Get outliner agent prompt"""
        template = self.load_prompt("outliner_agent", version)
        
        # Check if template needs classes formatting
        if classes and '{classes}' in template:
            classes_str = ', '.join(classes)
            # print(f"[PromptManager] Formatting outliner {version} with classes: {classes_str}")  # Debug
            try:
                formatted = self.format_prompt(template, classes=classes_str)
                return formatted
            except Exception as e:
                # print(f"ERROR in get_outliner_prompt: {e}")
                # print(f"Template snippet: {template[:200]}...")
                raise
        else:
            # Template doesn't use classes or classes not provided
            return template
    
    def get_aspect_prompt(self, prefix: str, postfix: str, version: str = "v1", classes: list = None) -> str:
        """Get formatted aspect agent prompt"""
        template = self.load_prompt("aspect_agent", version)
        
        # Check if template needs classes formatting
        if classes and '{classes}' in template:
            classes_str = ', '.join(classes)
            # print(f"[PromptManager] Formatting aspect {version} with classes: {classes_str}, prefix: {prefix[:50]}...")  # Debug
            return self.format_prompt(template, prefix=prefix, postfix=postfix, classes=classes_str)
        else:
            # Template doesn't use classes or classes not provided
            # print(f"[PromptManager] Formatting aspect {version} (no classes), prefix: {prefix[:50]}...")  # Debug
            return self.format_prompt(template, prefix=prefix, postfix=postfix)
    
    def get_reasoning_prompt(self, descriptions: list, classes: list, 
                           version: str = "v1", focuses: Optional[list] = None) -> str:
        """Get formatted reasoning agent prompt"""
        template = self.load_prompt("reasoning_agent", version)
        
        # Use format_reasoning_prompt to include numeric formatting
        return self.format_reasoning_prompt(template, descriptions, classes, version, focuses)
    
    def list_available_prompts(self) -> Dict[str, list]:
        """List all available prompts by agent"""
        available = {
            "outliner_agent": [],
            "aspect_agent": [],
            "reasoning_agent": []
        }
        
        if not os.path.exists(self.prompts_dir):
            return available
            
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.txt'):
                parts = filename[:-4].split('_')  # Remove .txt and split
                if len(parts) >= 3:  # agent_name_version format
                    agent_name = '_'.join(parts[:-1])
                    version = parts[-1]
                    if agent_name in available:
                        available[agent_name].append(version)
                        
        return available
    
    def save_prompt(self, agent_name: str, version: str, content: str):
        """Save a new prompt version"""
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        filename = f"{agent_name}_{version}.txt"
        filepath = os.path.join(self.prompts_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        # Clear cache for this prompt
        cache_key = f"{agent_name}_{version}"
        if cache_key in self.prompts_cache:
            del self.prompts_cache[cache_key]


class PromptExperiment:
    """Helper class to run experiments with different prompt versions"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.results = {}
        
    def run_with_prompts(self, outliner_version: str = "v1", 
                        aspect_version: str = "v1", 
                        reasoning_version: str = "v1") -> str:
        """Create experiment identifier for prompt combination"""
        return f"O{outliner_version}_A{aspect_version}_R{reasoning_version}"
    
    def get_prompt_config(self, experiment_id: str) -> Dict[str, str]:
        """Parse experiment ID to get prompt versions"""
        parts = experiment_id.split('_')
        return {
            'outliner': parts[0][1:],  # Remove 'O' prefix
            'aspect': parts[1][1:],   # Remove 'A' prefix
            'reasoning': parts[2][1:] # Remove 'R' prefix
        }