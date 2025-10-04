import os
import json
import time
import logging
import datetime
import base64
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any

# Import API providers from the local file
from .api_providers import (
    anthropic_completion,
    anthropic_text_completion,
    anthropic_multiimage_completion,
    openai_completion,
    openai_text_completion,
    openai_multiimage_completion,
    gemini_completion,
    gemini_text_completion,
    gemini_multiimage_completion,
    together_ai_completion,
    together_ai_text_completion,
    together_ai_multiimage_completion,
    deepseek_text_reasoning_completion,
    xai_grok_text_completion,
    vllm_text_completion,
    vllm_completion,
    vllm_multiimage_completion,
    modal_vllm_text_completion,
    modal_vllm_completion,
    modal_vllm_multiimage_completion,
)

# Import cost calculator utilities from the local file
from .api_cost_calculator import (
    calculate_all_costs_and_tokens,
    count_message_tokens,
    count_string_tokens,
    count_image_tokens,
    calculate_cost_by_tokens,
    calculate_prompt_cost,
    calculate_completion_cost,
    calculate_image_cost,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[]
)
logger = logging.getLogger(__name__)


class APIManager:
    """
    Object-oriented manager for API calls with cost tracking and logging.
    
    This class centralizes all API calls, handles token counting, cost calculation,
    and provides structured logging for inputs, outputs, and costs.
    """
    
    def __init__(
        self, 
        game_name: str, 
        base_cache_dir: str = "cache",
        enable_logging: bool = True,
        info: Optional[Dict[str, Any]] = None,
        session_dir: Optional[str] = None,
        vllm_url: Optional[str] = None,
        modal_url: Optional[str] = None
    ):
        """
        Initialize the API Manager.
        """
        self.game_name = game_name
        self.base_cache_dir = base_cache_dir
        self.enable_logging = enable_logging
        self.info = info or {}
        self.vllm_url = vllm_url
        self.modal_url = modal_url
        self.timestamp = self.info.get('datetime', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        self.total_session_cost = Decimal('0.0')
        self.last_action_end_time = None # To track idle time between calls
        
        if session_dir:
            self._set_session_dir(session_dir)
        else:
            self._setup_directories()
        
        self._setup_logger()
        logger.info(f"Initialized API Manager for {game_name}")
    
    def _setup_directories(self):
        """Set up all necessary cache directories."""
        self.game_dir = os.path.join(self.base_cache_dir, self.game_name)
        os.makedirs(self.game_dir, exist_ok=True)
        self.session_dir = None
    
    def _set_session_dir(self, session_dir: str) -> None:
        """
        Set the session directory directly, bypassing the directory creation logic.
        """
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.game_dir = os.path.dirname(os.path.dirname(os.path.dirname(session_dir)))
        logger.info(f"Using provided session directory: {session_dir}")
    
    def _setup_logger(self):
        """Set up logger with file handler for this session."""
        if not self.enable_logging:
            return
        
        log_file = os.path.join(self.session_dir, f"{self.game_name}_api_manager.log") if self.session_dir else os.path.join(self.game_dir, f"{self.game_name}_api_manager.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _get_model_session_dir(self, model_name: str, session_name: Optional[str] = None, modality: str = "default") -> str:
        """
        Get or create the directory for this model session.
        """
        model_name = self.info.get('model_name', model_name)
        modality = self.info.get('modality', modality)
        clean_model_name = model_name.lower().split('/')[-1] if '/' in model_name else model_name.lower()
        model_dir = os.path.join(self.game_dir, clean_model_name)
        os.makedirs(model_dir, exist_ok=True)
        modality_dir = os.path.join(model_dir, modality)
        os.makedirs(modality_dir, exist_ok=True)
        datetime_dir = os.path.join(modality_dir, self.timestamp)
        if session_name:
            datetime_dir = os.path.join(modality_dir, f"{self.timestamp}_{session_name}")
        os.makedirs(datetime_dir, exist_ok=True)
        self.session_dir = datetime_dir
        return datetime_dir
    
    def _log_api_call(
        self, 
        model_name: str,
        input_data: Dict[str, Any],
        output_data: str,
        costs: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        session_name: Optional[str] = None,
        modality: str = "default",
        slack_time_s: Optional[float] = None # MODIFIED: Accept slack time
    ) -> Dict[str, str]:
        """
        Log API call details to files, including performance metrics and total session cost.
        """
        if not self.enable_logging:
            return {}
        
        session_dir = self.session_dir or self._get_model_session_dir(model_name, session_name, modality)
        
        json_file = os.path.join(session_dir, "api_call.json")
        cost_log_file = os.path.join(session_dir, f"{self.game_name}_api_costs.log")
        
        current_call_cost = Decimal(costs.get('prompt_cost', 0)) + Decimal(costs.get('completion_cost', 0))
        self.total_session_cost += current_call_cost

        # MODIFIED: Add slack time to performance metrics for JSON logging
        if slack_time_s is not None:
            performance_metrics['slack_time_s'] = slack_time_s

        with open(json_file, "w", encoding="utf-8") as f:
            logged_input = {k: v for k, v in input_data.items() if "base64" not in k and "image" not in k}

            json_data = {
                "input": logged_input,
                "output": output_data,
                "costs": {k: str(v) if isinstance(v, Decimal) else v for k, v in costs.items()},
                "performance": performance_metrics,
                "total_session_cost": str(self.total_session_cost),
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "model": model_name,
                "game": self.game_name,
                "modality": modality,
                "conversation": {
                    "system": input_data.get("system_prompt", ""),
                    "user": input_data.get("prompt", ""),
                    "assistant": output_data
                }
            }
            json.dump(json_data, f, indent=2)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cost_log_entry = (
            f"[{timestamp}]\n"
            f"Game: {self.game_name}\n"
            f"Model: {model_name}\n"
            f"Modality: {modality}\n"
            f"Total Input Tokens: {costs.get('prompt_tokens', 0)}\n"
            f"Input Text Tokens: {costs.get('prompt_tokens', 0)}\n"
            f"Input Image Tokens: {costs.get('image_tokens', 0)}\n"
            f"Output Tokens: {costs.get('completion_tokens', 0)}\n"
            f"Total Input Cost: ${costs.get('prompt_cost', Decimal('0')):.6f}\n"
            f"Total Output Cost: ${costs.get('completion_cost', Decimal('0')):.6f}\n"
            f"Call Cost: ${current_call_cost:.6f}\n"
        )
        # MODIFIED: Log slack time here if available, then other performance metrics
        if slack_time_s is not None:
            cost_log_entry += f"Slack Time: {slack_time_s:.3f}s\n"
        
        if performance_metrics:
            ttft = performance_metrics.get('time_to_first_token_s', 0)
            latency = performance_metrics.get('average_inter_token_latency_ms', 0)
            total_time = performance_metrics.get('total_generation_time_s', 0)
            cost_log_entry += f"TTFT: {ttft:.3f}s, TBT: {latency:.2f}ms, Total Time: {total_time:.3f}s\n"
        
        cost_log_entry += f"Total Session Cost: ${self.total_session_cost:.6f}\n"
        cost_log_entry += f"{'-'*50}\n"
        
        with open(cost_log_file, "a", encoding="utf-8") as f:
            f.write(cost_log_entry)

        if 'total_generation_time_s' in performance_metrics:
            time_in_ms = performance_metrics['total_generation_time_s'] * 1000
            action_times_log_path = os.path.join(session_dir, "action_times_ms.log")
            try:
                prepend_comma = os.path.exists(action_times_log_path) and os.path.getsize(action_times_log_path) > 0
                with open(action_times_log_path, 'a', encoding='utf-8') as f:
                    if prepend_comma:
                        f.write(',')
                    f.write(f"{time_in_ms:.2f}")
            except Exception as e:
                logger.error(f"Failed to write to action_times_ms.log: {e}")
        
        logger.info(f"Logged API call ({modality}) to {json_file} and costs to {cost_log_file}")
        return {"json_file": json_file, "cost_log_file": cost_log_file}
    
    def _calculate_costs(
        self, 
        model_name: str, 
        prompt: Union[str, List[Dict]], 
        completion: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate token usage and costs.
        """
        try:
            costs = calculate_all_costs_and_tokens(
                prompt=prompt,
                completion=completion,
                model=model_name,
                image_path=image_path
            )
            logger.info(
                f"Calculated costs for {model_name}: "
                f"input={costs['prompt_tokens']} tokens (${costs['prompt_cost']}), "
                f"output={costs['completion_tokens']} tokens (${costs['completion_cost']})"
            )
            return costs
        except Exception as e:
            logger.error(f"Error calculating costs: {e}")
            return {
                "prompt_tokens": 0, "completion_tokens": 0,
                "prompt_cost": Decimal("0"), "completion_cost": Decimal("0"),
                "image_tokens": 0 if image_path else None,
                "image_cost": Decimal("0") if image_path else None
            }
    
    def _get_base64_from_path(self, image_path: str) -> str:
        """
        Convert image file to base64 string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            raise
    
    def vision_text_completion(
        self, 
        model_name: str, 
        system_prompt: str, 
        prompt: str, 
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
        stream: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a combined vision-text completion API call.
        """
        if not (image_path or base64_image):
            raise ValueError("Either image_path or base64_image must be provided")
        
        if image_path and not base64_image:
            base64_image = self._get_base64_from_path(image_path)
        
        input_data = {
            "system_prompt": system_prompt, "prompt": prompt, "base64_image": base64_image,
            "model_name": model_name, "temperature": temperature, "thinking": thinking,
            "reasoning_effort": reasoning_effort, "token_limit": token_limit, "stream": stream, **kwargs
        }
        
        completion, performance_metrics = "", {}

        try:
            if "gemini" in model_name.lower():
                completion, performance_metrics = gemini_completion(
                    system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, 
                    prompt=prompt, token_limit=token_limit, stream=stream
                )
            elif "claude" in model_name.lower():
                completion, performance_metrics = anthropic_completion(
                    system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, 
                    prompt=prompt, thinking=thinking, token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion, performance_metrics = openai_completion(
                    system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, 
                    prompt=prompt, temperature=temperature, reasoning_effort=reasoning_effort, token_limit=token_limit
                )
            elif model_name.startswith("vllm-"):
                completion, performance_metrics = vllm_completion(
                    system_prompt=system_prompt, vllm_model_name=model_name, prompt=prompt, 
                    base64_image=base64_image, temperature=temperature, token_limit=token_limit, url=self.vllm_url
                )
            elif model_name.startswith("modal-"):
                completion, performance_metrics = modal_vllm_completion(
                    system_prompt=system_prompt, model_name=model_name, prompt=prompt, 
                    base64_image=base64_image, temperature=temperature, token_limit=token_limit, url=self.modal_url
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower() or "deepseek" in model_name.lower() or "qwen" in model_name.lower():
                completion, performance_metrics = together_ai_completion(
                    system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, 
                    prompt=prompt, temperature=temperature, token_limit=token_limit
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            empty_costs = {
                "prompt_tokens": 0, "completion_tokens": 0, "prompt_cost": "0",
                "completion_cost": "0", "image_tokens": 0, "image_cost": "0"
            }
            empty_costs.update(performance_metrics)
            
            return completion, empty_costs
            
        except Exception as e:
            logger.error(f"Error in vision-text completion API call: {e}")
            raise
    
    def vision_only_completion(
        self, 
        model_name: str, 
        system_prompt: str,
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        session_name: Optional[str] = None,
        temperature: float = 0,
        thinking: bool = False,
        stream: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a vision-only completion API call.
        """
        # MODIFIED: Calculate slack time but do not log it here
        slack_time_s = None
        if self.last_action_end_time is not None:
            slack_time_s = time.time() - self.last_action_end_time
        
        if not (image_path or base64_image):
            raise ValueError("Either image_path or base64_image must be provided")
        
        if image_path and not base64_image:
            base64_image = self._get_base64_from_path(image_path)
        
        empty_prompt = "Describe this image in detail."
        
        input_data = {
            "system_prompt": system_prompt, "prompt": empty_prompt, "base64_image": base64_image,
            "model_name": model_name, "temperature": temperature, "thinking": thinking, "stream": stream, **kwargs
        }
        
        completion, performance_metrics = "", {}

        try:
            if "gemini" in model_name.lower():
                completion, performance_metrics = gemini_completion(
                    system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, 
                    prompt=empty_prompt, stream=stream, **kwargs
                )
            elif "claude" in model_name.lower():
                completion, performance_metrics = anthropic_completion(system_prompt=system_prompt, model_name=model_name, base64_image=base64_image, prompt=empty_prompt, thinking=thinking, **kwargs)
            else:
                completion, performance_metrics = self.vision_text_completion(model_name=model_name, system_prompt=system_prompt, prompt=empty_prompt, base64_image=base64_image, temperature=temperature, **kwargs)
            
            costs = self._calculate_costs(model_name=model_name, prompt=empty_prompt, completion=completion, image_path=image_path)
            
            # MODIFIED: Pass slack time to the logger
            self._log_api_call(
                model_name=model_name, input_data=input_data, output_data=completion, 
                costs=costs, performance_metrics=performance_metrics, session_name=session_name, modality="vision_only",
                slack_time_s=slack_time_s
            )
            
            self.last_action_end_time = time.time()
            costs.update(performance_metrics)
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in vision-only completion API call: {e}")
            raise
    
    def text_only_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
        stream: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a text-only completion API call, now universally handling performance metrics.
        """
        # MODIFIED: Calculate slack time but do not log it here
        slack_time_s = None
        if self.last_action_end_time is not None:
            slack_time_s = time.time() - self.last_action_end_time

        input_data = {
            "system_prompt": system_prompt, "prompt": prompt, "model_name": model_name,
            "temperature": temperature, "thinking": thinking, "reasoning_effort": reasoning_effort,
            "token_limit": token_limit, "stream": stream, **kwargs
        }
        
        completion, performance_metrics = "", {}

        try:
            if "gemini" in model_name.lower():
                completion, performance_metrics = gemini_text_completion(
                    system_prompt=system_prompt, model_name=model_name, prompt=prompt, 
                    token_limit=token_limit, stream=stream
                )
            elif "claude" in model_name.lower():
                completion, performance_metrics = anthropic_text_completion(
                    system_prompt=system_prompt, model_name=model_name, prompt=prompt, 
                    thinking=thinking, token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion, performance_metrics = openai_text_completion(
                    system_prompt=system_prompt, model_name=model_name, prompt=prompt, 
                    reasoning_effort=reasoning_effort, token_limit=token_limit
                )
            elif "deepseek" in model_name.lower():
                 completion, performance_metrics = deepseek_text_reasoning_completion(system_prompt, model_name, prompt, token_limit)
            elif "grok" in model_name.lower():
                 completion, performance_metrics = xai_grok_text_completion(system_prompt, model_name, prompt, token_limit=token_limit, temperature=temperature, reasoning_effort=reasoning_effort)
            elif "llama" in model_name.lower() or "meta" in model_name.lower() or "qwen" in model_name.lower():
                 completion, performance_metrics = together_ai_text_completion(system_prompt, model_name, prompt, temperature=temperature, token_limit=token_limit)
            elif model_name.startswith("vllm-"):
                 completion, performance_metrics = vllm_text_completion(system_prompt, model_name, prompt, temperature=temperature, token_limit=token_limit, url=self.vllm_url)
            elif model_name.startswith("modal-"):
                 completion, performance_metrics = modal_vllm_text_completion(system_prompt, model_name, prompt, temperature=temperature, token_limit=token_limit, url=self.modal_url)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            formatted_prompt = [{"role": "user", "content": [{"type": "text", "text": prompt}]}] if "claude" in model_name.lower() else prompt
            
            costs = self._calculate_costs(model_name=model_name, prompt=formatted_prompt, completion=completion)
            
            # MODIFIED: Pass slack time to the logger
            self._log_api_call(
                model_name=model_name, input_data=input_data, output_data=completion, 
                costs=costs, performance_metrics=performance_metrics, session_name=session_name, modality="text_only",
                slack_time_s=slack_time_s
            )
            
            self.last_action_end_time = time.time()
            costs.update(performance_metrics)
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in text-only completion API call: {e}")
            raise
    
    def vision_completion(self, *args, **kwargs) -> Tuple[str, Dict[str, Any]]:
        logger.warning("Using legacy vision_completion - consider using vision_text_completion instead")
        return self.vision_text_completion(*args, **kwargs)
    
    def text_completion(self, *args, **kwargs) -> Tuple[str, Dict[str, Any]]:
        logger.warning("Using legacy text_completion - consider using text_only_completion instead")
        return self.text_only_completion(*args, **kwargs)
    
    def multi_image_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        list_content: List[str],
        list_image_paths: Optional[List[str]] = None,
        list_image_base64: Optional[List[str]] = None,
        session_name: Optional[str] = None,
        temperature: float = 1,
        reasoning_effort: str = "medium",
        stream: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        # MODIFIED: Calculate slack time but do not log it here
        slack_time_s = None
        if self.last_action_end_time is not None:
            slack_time_s = time.time() - self.last_action_end_time

        if not (list_image_paths or list_image_base64):
            raise ValueError("Either list_image_paths or list_image_base64 must be provided")
            
        if list_image_paths and not list_image_base64:
            list_image_base64 = [self._get_base64_from_path(p) for p in list_image_paths]
        
        input_data = {
            "system_prompt": system_prompt, "prompt": prompt, "list_content": list_content,
            "list_image_base64": list_image_base64, "model_name": model_name, 
            "temperature": temperature, "reasoning_effort": reasoning_effort, **kwargs
        }
        
        completion, performance_metrics = "", {}

        try:
            if "gemini" in model_name.lower():
                completion, performance_metrics = gemini_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, **kwargs)
            else:
                completion, performance_metrics = ("Fallback result", {})

            costs = self._calculate_costs(model_name=model_name, prompt=prompt, completion=completion)
            if list_image_paths:
                image_tokens = sum(count_image_tokens(p, model_name) for p in list_image_paths)
                image_cost = calculate_cost_by_tokens(image_tokens, model_name, "input")
                costs["image_tokens"] = image_tokens
                costs["image_cost"] = image_cost
                costs["prompt_tokens"] += image_tokens
                costs["prompt_cost"] += image_cost

            # MODIFIED: Pass slack time to the logger
            self._log_api_call(
                model_name=model_name, input_data=input_data, output_data=completion, 
                costs=costs, performance_metrics=performance_metrics, session_name=session_name, modality="multi_image",
                slack_time_s=slack_time_s
            )
            
            self.last_action_end_time = time.time()
            costs.update(performance_metrics)
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in multi-image completion API call: {e}")
            raise