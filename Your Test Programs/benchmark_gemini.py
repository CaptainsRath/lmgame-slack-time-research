import google.generativeai as genai
from google.generativeai import types
import time
import random
import datetime
import os
from tools.serving.api_providers import gemini_text_completion

API_KEY = "AIzaXXXXXXXX" 
MODEL_NAME = "gemini-2.5-flash"
NUMBER_OF_RUNS = 10
DETAILED_LOG_FILE = "performance_log.txt"
TTFT_LOG_FILE = "ttft_log.txt"


def gemini_text_completion_direct(system_prompt, model_name, prompt, api_key, token_limit=30000, stream=False):
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

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

def generate_random_prompt():
    list_length = random.randint(5, 15)
    start_number = random.randint(1, 100)  
    sequential_numbers = list(range(start_number, start_number + list_length))
    
    core_prompt = (
        f"Analyze the following sequence of numbers: {sequential_numbers}. "
        "Based on the pattern, what is the most likely next number in the sequence? "
        "Please provide only the number as your answer."
    )
    
    # Calculate how much padding we need
    target_length = 12741
    current_length = len(core_prompt)
    padding_needed = target_length - current_length
    
    if padding_needed > 0:
        # Add contextual padding that doesn't change the task
        padding_text = (
            " This is a mathematical sequence analysis task. "
            "Please focus on identifying the clear numerical pattern present in the given sequence. "
            "Mathematical sequences often follow predictable rules, and your task is to determine "
            "what comes next based on the established pattern. "
            "The sequence provided above follows a specific mathematical progression. "
            "Examine each number carefully and determine the relationship between consecutive elements. "
            "Once you identify the pattern, apply it to predict the next logical number in the series. "
        )
        
        # Repeat the padding text as needed to reach target length
        full_padding = ""
        while len(full_padding) < padding_needed:
            remaining_space = padding_needed - len(full_padding)
            if remaining_space >= len(padding_text):
                full_padding += padding_text
            else:
                # Add partial padding text to reach exact length
                full_padding += padding_text[:remaining_space]
        
        padded_prompt = core_prompt #+ full_padding
        
        # Ensure exact length (trim if slightly over)
        #if len(padded_prompt) > target_length:
           # padded_prompt = padded_prompt[:target_length]
            
  #  else:
       # padded_prompt = core_prompt
    
    return padded_prompt

def print_performance_log_summary(log_file):
    import pandas as pd
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
    try:
        df = pd.read_csv(log_file)
        print("\nPerformance Log Summary:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not parse log file: {e}")

def get_ttft_and_response(model_name, prompt, api_key):
    
    system_prompt = ""
    response_text, perf = gemini_text_completion_direct(
        system_prompt=system_prompt,
        model_name=model_name,
        prompt=prompt,
        api_key=API_KEY,
        token_limit=30000,
        stream=True
    )
    ttft = perf.get("time_to_first_token_s", None)
    total_time = perf.get("total_generation_time_s", None)
    return response_text, ttft, total_time

def main():
    print(f" Starting benchmark for '{MODEL_NAME}' with {NUMBER_OF_RUNS} runs...")

    # Test with a simple request first, to ensure basic functionality
    try:
        print(" Testing basic functionality...")
        test_response, _ = gemini_text_completion_direct(
            system_prompt="",
            model_name=MODEL_NAME,
            prompt="Hello",
            api_key=API_KEY,
            token_limit=30000,
            stream=False
        )
        print(f"Basic test passed: {test_response[:50]}...")
    except Exception as e:
        print(f"Basic test failed: {type(e).__name__}: {str(e)}")
        return

    ttft_milliseconds_list = []
    last_response_finish_time = time.perf_counter()

    # Create detailed log file
    with open(DETAILED_LOG_FILE, "w") as f:
        f.write(
            "Run,Timestamp,Slack Time (s),Time to First Token (s),"
            "Total Response Time (s),Avg. Inter-Token Time (s),"
            "Input Tokens,Output Tokens\n"
        )

    print(f"\nStarting {NUMBER_OF_RUNS} benchmark runs...")

    for i in range(NUMBER_OF_RUNS):
        run_number = i + 1
        print(f"\n--- Running iteration {run_number}/{NUMBER_OF_RUNS} ---")
        prompt = generate_random_prompt()
        print(f"   - Prompt length: {len(prompt)} characters")
        print(f"   - Prompt preview: {prompt[:100]}...")
        print(f"   - Prompt ending: ...{prompt[-100:]}")

        try:
            request_start_time = time.perf_counter()
            slack_time = request_start_time - last_response_finish_time

            print("   - Sending request...")

            response_text, ttft, total_response_time = get_ttft_and_response(MODEL_NAME, prompt, API_KEY)
            last_response_finish_time = time.perf_counter()

            ttft_ms = int(ttft * 1000) if ttft is not None else -1
            ttft_milliseconds_list.append(ttft_ms)

            print(f"   - Response received in {total_response_time:.3f}s")
            print(f"   - Response: {response_text[:100]}...")

            input_tokens = len(prompt.split())
            output_tokens = len(response_text.split()) if response_text else 0
            print(f"   - Estimated tokens: {input_tokens} → {output_tokens}")

            if output_tokens > 1:
                time_after_first_token = total_response_time - ttft
                avg_inter_token_time = time_after_first_token / (output_tokens - 1)
            else:
                avg_inter_token_time = 0.0

            # Create log entry
            timestamp = datetime.datetime.now().isoformat()
            log_entry = (
                f"{run_number},{timestamp},{slack_time:.4f},{ttft:.4f},"
                f"{total_response_time:.4f},{avg_inter_token_time:.4f},"
                f"{input_tokens},{output_tokens}\n"
            )

            # Write to log file
            with open(DETAILED_LOG_FILE, "a") as f:
                f.write(log_entry)

            print(f" Run {run_number} complete.")
            print(f"   - Estimated TTFT: {ttft_ms:.0f}ms")
            print(f"   - Total time: {total_response_time:.3f}s")

        except Exception as e:
            print(f" Error in run {run_number}: {type(e).__name__}: {str(e)}")
            ttft_milliseconds_list.append(-1)
            
            timestamp = datetime.datetime.now().isoformat()
            error_entry = f"{run_number},{timestamp},-1,-1,-1,-1,0,0\n"
            with open(DETAILED_LOG_FILE, "a") as f:
                f.write(error_entry)
            
            # Brief pause before next attempt
            time.sleep(1)

    with open(TTFT_LOG_FILE, "w") as f:
        formatted_ttft = ", ".join(map(str, ttft_milliseconds_list))
        f.write(f"({formatted_ttft})")

    print(f"\n Benchmark finished!")
    print(f" Detailed results saved to: {DETAILED_LOG_FILE}")
    print(f" TTFT-only results saved to: {TTFT_LOG_FILE}")

    # Summary statistics
    successful_runs = [x for x in ttft_milliseconds_list if x != -1]
    if successful_runs:
        avg_ttft = sum(successful_runs) / len(successful_runs)
        min_ttft = min(successful_runs)
        max_ttft = max(successful_runs)
        
        print(f"\n Summary:")
        print(f"   - Successful runs: {len(successful_runs)}/{NUMBER_OF_RUNS}")
        print(f"   - Average TTFT: {avg_ttft:.0f}ms")
        print(f"   - TTFT range: {min_ttft}ms - {max_ttft}ms")
        
        # Show the TTFT list for verification
        print(f"   - TTFT values: {successful_runs[:5]}{'...' if len(successful_runs) > 5 else ''}")
    else:
        print("\n No successful runs completed")

    # Write slack times to a third log file
    slack_times_ms = []
    with open(DETAILED_LOG_FILE, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 2:
                try:
                    slack_s = float(parts[2])
                    slack_ms = int(slack_s * 1000)
                    slack_times_ms.append(slack_ms)
                except Exception:
                    slack_times_ms.append(-1)
    with open("slack_log.txt", "w") as f:
        formatted_slack = ", ".join(map(str, slack_times_ms))
        f.write(f"({formatted_slack})")
    print(f" Slack times saved to: slack_log.txt")

    try:
        import pandas as pd
        print_performance_log_summary(DETAILED_LOG_FILE)
    except ImportError:
        print("\n(pandas not installed, skipping pretty log summary. Install with 'pip install pandas' for better output.)")

if __name__ == "__main__":
    main()