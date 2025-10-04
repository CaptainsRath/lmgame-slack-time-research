import logging
logging.basicConfig(filename="inputLog",level=logging.DEBUG)    
import os
import time
import functools
import httpx
import random
from typing import Dict, Any, Tuple, List

# To handle potential errors, it's good practice to import them specifically
try:
    from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, BadRequestError
except ImportError:
    # Define dummy exceptions if openai is not installed, so the script doesn't crash on import
    class RateLimitError(Exception): pass
    class APITimeoutError(Exception): pass
    class APIConnectionError(Exception): pass
    class APIStatusError(Exception): pass
    class BadRequestError(Exception): pass

import anthropic
import google.generativeai as genai
from google.generativeai import types
from together import Together
import requests

def _sleep_with_backoff(base_delay: int, attempt: int) -> None:
    """Sleeps for a duration with exponential backoff and jitter."""
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
    print(f"Retrying in {delay:.2f}s … (attempt {attempt + 1})")
    time.sleep(delay)

def retry_on_openai_error(func):
    """
    Retry wrapper for OpenAI SDK calls.
    Retries on: RateLimitError, Timeout, APIConnectionError,
                APIStatusError (5xx), httpx.RemoteProtocolError.
    Immediately raises on: BadRequestError (400).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop("max_retries", 5)
        base_delay  = kwargs.pop("base_delay", 2)

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            # transient issues worth retrying
            except (RateLimitError, APITimeoutError, APIConnectionError,
                    httpx.RemoteProtocolError, BadRequestError) as e:
                if attempt < max_retries - 1:
                    print(f"OpenAI transient error: {e}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise
            # server‑side 5xx response
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < max_retries - 1:
                    print(f"OpenAI server error {e.status_code}: {e.message}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise
    return wrapper

def retry_on_overload(func):
    """
    A decorator to retry a function call on specific transient errors,
    including anthropic overload, httpx errors, or empty responses.
    It uses exponential backoff with jitter.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)

                # Check if result is None or empty string, or a tuple with an empty string
                is_empty = False
                if isinstance(result, tuple):
                    # Handle cases where the function returns (completion, metrics)
                    completion = result[0]
                    if completion is None or (isinstance(completion, str) and not completion.strip()):
                        is_empty = True
                elif result is None or (isinstance(result, str) and not result.strip()):
                    is_empty = True

                if is_empty:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + (random.uniform(0, 1))
                        print(f"API returned None/empty response. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"API still returning None/empty after {max_retries} attempts. Raising an error.")
                        raise RuntimeError("API returned None/empty response after all retry attempts")

                return result
            except anthropic.APIStatusError as e:
                if e.body and e.body.get('error', {}).get('type') == 'overloaded_error':
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + (random.uniform(0, 1))
                        print(f"Anthropic API overloaded. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"Anthropic API still overloaded after {max_retries} attempts. Raising the error.")
                        raise
                else:
                    raise
            except httpx.RemoteProtocolError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (random.uniform(0, 1))
                    print(f"Streaming connection closed unexpectedly. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Streaming connection failed after {max_retries} attempts. Raising the error.")
                    raise
    return wrapper

@retry_on_overload
def anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=False, token_limit=30000):
    print(f"anthropic vision-text activated... thinking: {thinking}")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    if "claude-3-5" in model_name:
        print("claude-3-5 only supports 8192 tokens and no thinking")
        thinking = False
        token_limit = 8192

    if "claude-3-7" in model_name:
        print("claude-3-7 supports 64000 tokens")
        token_limit = 64000

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000

    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000

    if thinking:
        with client.messages.stream(
                max_tokens=token_limit,
                thinking={
                    "type": "enabled",
                    "budget_tokens": token_limit - 1
                },
                messages=messages,
                temperature=1,
                system=system_prompt,
                model=model_name,
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    return "".join(partial_chunks)
    else:
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name,
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    return "".join(partial_chunks)

    generated_code_str = "".join(partial_chunks)
    return generated_code_str

@retry_on_overload
def anthropic_text_completion(system_prompt, model_name, prompt, thinking=False, token_limit=30000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    token_limit = 64000 if "claude-3-7" in model_name and token_limit > 64000 else token_limit
    print(f"model_name: {model_name}, token_limit: {token_limit}, thinking: {thinking}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    if "claude-3-5" in model_name:
        print("claude-3-5 only supports 8192 tokens and no thinking")
        thinking = False
        token_limit = 8192

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000

    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000

    if thinking:
        with client.messages.stream(
                max_tokens=token_limit,
                thinking={
                    "type": "enabled",
                    "budget_tokens": token_limit - 1
                },
                messages=messages,
                temperature=1,
                system=system_prompt,
                model=model_name,
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    return "".join(partial_chunks)
    else:
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name,
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    return "".join(partial_chunks)

    generated_str = "".join(partial_chunks)
    return generated_str

@retry_on_overload
def anthropic_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000

    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000

    content_blocks = []
    for text_item, base64_image in zip(list_content, list_image_base64):
        content_blocks.append({"type": "text", "text": text_item})
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image,
            },
        })

    content_blocks.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content_blocks}]
    print(f"message size: {len(content_blocks)+1}")

    with client.messages.stream(
            max_tokens=token_limit,
            messages=messages,
            temperature=0,
            system=system_prompt,
            model=model_name,
        ) as stream:
            partial_chunks = []
            try:
                for chunk in stream.text_stream:
                    print(chunk)
                    partial_chunks.append(chunk)
            except httpx.RemoteProtocolError as e:
                print(f"Streaming connection closed unexpectedly: {e}")
                return "".join(partial_chunks)

    generated_str = "".join(partial_chunks)
    return generated_str

# Monkey patch for httpx headers
_original_headers_init = httpx.Headers.__init__
def safe_headers_init(self, headers=None, encoding=None):
    if isinstance(headers, dict):
        headers = {k: (v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v) for k, v in headers.items()}
    elif isinstance(headers, list):
        headers = [(k, v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v) for k, v in headers]
    _original_headers_init(self, headers=headers, encoding=encoding)
httpx.Headers.__init__ = safe_headers_init

@retry_on_openai_error
def openai_completion(system_prompt, model_name, base64_image, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI vision-text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name: token_limit = 16384
    elif "gpt-4.1" in model_name: token_limit = 32768
    elif "o3" in model_name: token_limit = 10000

    base64_image = None if "o3-mini" in model_name else base64_image
    if base64_image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}, {"type": "text", "text": prompt}]}]

    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    request_params = {"model": model_name, "messages": messages, token_param: token_limit}
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

@retry_on_openai_error
def openai_text_completion(system_prompt, model_name, prompt, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-only API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name: token_limit = 16384
    elif "gpt-4.1" in model_name: token_limit = 32768
    elif "o3" in model_name: token_limit = 10000

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    request_params = {"model": model_name, "messages": messages, token_param: token_limit}
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = 1

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

@retry_on_openai_error
def openai_text_reasoning_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-reasoning API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name: token_limit = 16384
    elif "gpt-4.1" in model_name: token_limit = 32768
    elif "o3" in model_name: token_limit = 10000

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    request_params = {"model": model_name, "messages": messages, token_param: token_limit}
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

def deepseek_text_reasoning_completion(system_prompt, model_name, prompt, token_limit=30000):
    print(f"DeepSeek text-reasoning API call: model={model_name}")
    if token_limit > 8192:
        token_limit = 8192
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    messages = [{"role": "user", "content": prompt}]
    content = ""
    response = client.chat.completions.create(model=model_name, messages=messages, stream=True, max_tokens=token_limit)
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    return content

def xai_grok_text_completion(system_prompt, model_name, prompt, reasoning_effort="high", token_limit=30000, temperature=1):
    print(f"XAI Grok text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    params = {"model": model_name, "messages": messages, "temperature": temperature, "max_tokens": token_limit}
    if "grok-3-mini" in model_name:
        params["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(**params)
    return completion.choices[0].message.content

@retry_on_openai_error
def openai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI multi-image API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name: token_limit = 16384
    elif "gpt-4.1" in model_name: token_limit = 32768
    elif "o3" in model_name: token_limit = 10000

    content_blocks = []
    joined_steps = "\n\n".join(list_content)
    content_blocks.append({"type": "text", "text": joined_steps})
    for base64_image in list_image_base64:
        content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

    messages = [{"role": "user", "content": content_blocks}]
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    request_params = {"model": model_name, "messages": messages, token_param: token_limit}
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = 1

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

@retry_on_overload
def gemini_text_completion(system_prompt, model_name, prompt, token_limit=30000, stream=False) -> Tuple[str, Dict[str, Any]]:
    """
    Generates text completion using Gemini, with optional performance measurement via streaming.
    Returns the completion text and a dictionary of performance metrics.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    print(f"gemini_text_completion: model_name={model_name}, token_limit={token_limit}, stream={stream}")

    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    logging.info(full_prompt)
    if not stream:
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=types.GenerationConfig(max_output_tokens=token_limit)
            )
            return response.text, {}
        except Exception as e:
            print(f"Error in non-streaming Gemini call: {e}")
            return "", {}

    start_time = time.perf_counter()
    response_stream = model.generate_content(
        full_prompt,
        generation_config=types.GenerationConfig(max_output_tokens=token_limit),
        stream=True
    )
    ttft, first_chunk_received, last_chunk_time = 0.0, False, None
    inter_token_latencies, completion_text = [], ""

    try:
        for chunk in response_stream:
            current_chunk_time = time.perf_counter()
            if not first_chunk_received:
                ttft = current_chunk_time - start_time
                first_chunk_received = True
            elif last_chunk_time:
                inter_token_latencies.append(current_chunk_time - last_chunk_time)
            if hasattr(chunk, 'text') and chunk.text:
                completion_text += chunk.text
            last_chunk_time = current_chunk_time

        total_generation_time = (last_chunk_time - start_time) if last_chunk_time else 0
        avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies) if inter_token_latencies else 0
        performance_metrics = {
            "time_to_first_token_s": ttft,
            "average_inter_token_latency_ms": avg_inter_token_latency * 1000,
            "total_generation_time_s": total_generation_time,
        }
        return completion_text, performance_metrics
    except Exception as e:
        print(f"Error during Gemini stream: {e}")
        return completion_text, {}

@retry_on_overload
def gemini_completion(system_prompt, model_name, base64_image, prompt, token_limit=30000, stream=False) -> Tuple[str, Dict[str, Any]]:
    """
    Generates vision-text completion using Gemini, with optional performance measurement.
    Returns the completion text and a dictionary of performance metrics.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    print(f"gemini_completion: model_name={model_name}, token_limit={token_limit}, stream={stream}")

    messages = [system_prompt if system_prompt else "", {"mime_type": "image/png", "data": base64_image}, prompt]
    logging.info(messages)
    if not stream:
        try:
            response = model.generate_content(
                messages,
                generation_config=types.GenerationConfig(max_output_tokens=token_limit)
            )
            return response.text, {}
        except Exception as e:
            print(f"Error in non-streaming Gemini vision call: {e}")
            return "", {}

    start_time = time.perf_counter()
    response_stream = model.generate_content(
        messages,
        generation_config=types.GenerationConfig(max_output_tokens=token_limit),
        stream=True
    )
    ttft, first_chunk_received, last_chunk_time = 0.0, False, None
    inter_token_latencies, completion_text = [], ""

    try:
        for chunk in response_stream:
            current_chunk_time = time.perf_counter()
            if not first_chunk_received:
                ttft = current_chunk_time - start_time
                first_chunk_received = True
            elif last_chunk_time:
                inter_token_latencies.append(current_chunk_time - last_chunk_time)
            if hasattr(chunk, 'text') and chunk.text:
                completion_text += chunk.text
            last_chunk_time = current_chunk_time

        total_generation_time = (last_chunk_time - start_time) if last_chunk_time else 0
        avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies) if inter_token_latencies else 0
        performance_metrics = {
            "time_to_first_token_s": ttft,
            "average_inter_token_latency_ms": avg_inter_token_latency * 1000,
            "total_generation_time_s": total_generation_time,
        }
        return completion_text, performance_metrics
    except Exception as e:
        print(f"Error during Gemini vision stream: {e}")
        return completion_text, {}

def gemini_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    content_blocks = []
    for base64_image in list_image_base64:
        content_blocks.append({"mime_type": "image/jpeg", "data": base64_image})
    joined_steps = "\n\n".join(list_content)
    content_blocks.append(joined_steps)
    messages = content_blocks
    logging.info(messages)
    try:
        response = model.generate_content(messages, generation_config=types.GenerationConfig(max_output_tokens=token_limit))
        return response.text
    except Exception as e:
        print(f"error: {e}")
        return ""

def together_ai_completion(system_prompt, model_name, prompt, base64_image=None, temperature=1, token_limit=30000):
    try:
        client = Together()
        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")

        if base64_image is not None:
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}, {"type": "text", "text": prompt}]}]
        else:
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        response = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=token_limit)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in together_ai_completion: {e}")
        raise

def together_ai_text_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000):
    print(f"Together AI text-only API call: model={model_name}")
    try:
        client = Together()
        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")

        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=token_limit)
        generated_str = response.choices[0].message.content

        import re
        def extract_move(text):
            think_match = re.search(r"</think>", text)
            after_think = text[think_match.end():] if think_match else text
            return after_think.strip()

        if model_name == "deepseek-ai/DeepSeek-R1" or model_name == "Qwen/Qwen3-235B-A22B-fp8":
            generated_str = extract_move(generated_str)

        return generated_str
    except Exception as e:
        print(f"Error in together_ai_text_completion: {e}")
        raise

def together_ai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, temperature=1, token_limit=30000):
    print(f"Together AI multi-image API call: model={model_name}")
    try:
        client = Together()
        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")

        content_blocks = []
        joined_text = "\n\n".join(list_content)
        content_blocks.append({"type": "text", "text": joined_text})
        for base64_image in list_image_base64:
            content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
        content_blocks.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_blocks})

        response = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=token_limit)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in together_ai_multiimage_completion: {e}")
        raise

def parse_vllm_model_name(model_name: str) -> str:
    return model_name[len("vllm-"):] if model_name.startswith("vllm-") else model_name

def vllm_text_completion(system_prompt, vllm_model_name, prompt, token_limit=30000, temperature=1, port=8000, host="localhost"):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {"model": model_name, "messages": messages, "max_tokens": token_limit, "temperature": temperature, "stream": False}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def vllm_completion(system_prompt, vllm_model_name, prompt, base64_image=None, token_limit=30000, temperature=1.0, port=8000, host="localhost"):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}
    user_content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}, {"type": "text", "text": prompt}] if base64_image else [{"type": "text", "text": prompt}]
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {"model": model_name, "messages": messages, "max_tokens": token_limit, "temperature": temperature, "stream": False}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def vllm_multiimage_completion(system_prompt, vllm_model_name, prompt, list_image_base64, token_limit=30000, temperature=1.0, port=8000, host="localhost"):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}
    user_content = []
    for image_base64 in list_image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    user_content.append({"type": "text", "text": prompt})
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {"model": model_name, "messages": messages, "max_tokens": token_limit, "temperature": temperature, "stream": False}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def parse_modal_model_name(modal_model_name: str) -> str:
    return modal_model_name[len("modal-"):] if modal_model_name.startswith("modal-") else modal_model_name

def modal_vllm_text_completion(system_prompt: str, model_name: str, prompt: str, token_limit: int = 30000, temperature: float = 1.0, api_key: str = "DUMMY_TOKEN", url: str = "https://your-modal-url.modal.run/v1"):
    model_name = parse_modal_model_name(model_name)
    print(f"calling modal_vllm_text_completion...\nmodel_name: {model_name}\nurl: {url}\n")
    client = OpenAI(api_key=api_key or os.getenv("MODAL_API_KEY"), base_url=url)
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=token_limit, temperature=temperature)
    return response.choices[0].message.content

def modal_vllm_completion(system_prompt: str, model_name: str, prompt: str, base64_image: str = None, token_limit: int = 30000, temperature: float = 1.0, api_key: str = "DUMMY_TOKEN", url: str = "https://your-modal-url.modal.run/v1"):
    model_name = parse_modal_model_name(model_name)
    print(f"calling modal_vllm_completion...\nmodel_name: {model_name}\nurl: {url}\n")
    client = OpenAI(api_key=api_key or os.getenv("MODAL_API_KEY"), base_url=url)
    user_content = []
    if base64_image: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    user_content.append({"type": "text", "text": prompt})
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=token_limit, temperature=temperature)
    return response.choices[0].message.content

def modal_vllm_multiimage_completion(system_prompt: str, model_name: str, prompt: str, list_image_base64: list, token_limit: int = 30000, temperature: float = 1.0, api_key: str = "DUMMY_TOKEN", url: str = "https://your-modal-url.modal.run/v1"):
    model_name = parse_modal_model_name(model_name)
    print(f"calling modal_multiimage_vllm_completion...\nmodel_name: {model_name}\nurl: {url}\n")
    client = OpenAI(api_key=api_key or os.getenv("MODAL_API_KEY"), base_url=url)
    user_content = []
    for base64_image in list_image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    user_content.append({"type": "text", "text": prompt})
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=token_limit, temperature=temperature)
    return response.choices[0].message.content