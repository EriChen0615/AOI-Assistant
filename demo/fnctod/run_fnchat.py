
"""
FnChat an interactive, multi-turn task-oriented dialogue assistant based on the FncTOD approach. 

The approach is inspired by FncTOD https://arxiv.org/abs/2402.10466). At each turn, FnChat performs the following steps:

1. Domain Prediction: predict the relevant domain and thereby retrieving potential userful functions given the user input and dialogue history
2. Dialogue State Tracking via Filling Function Arguments: generate arguments of the function
3. Function Execution: if the generated arguments are sufficient for a call, the function is executed and results returned 
4. Generate Response: generate response based on the dialogue history and the function's returned value.

Jinghong Chen. 2025.6
"""

"-------------HYPER PARAMETERS / PROMPTS-------------"

SUPPORTED_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-0125"]
SUPPORTED_QWEN_MODELS = ["Qwen/Qwen2.5-3B-Instruct"]

EXPERIMENT_DOMAINS = ["[general]", "[wiki]", "[news]"]
DOMAIN_PREDICTION_SYSTEM_MESSAGE = (
    "You are a task-oriented assistant. "
    "Your role is to determine which domain the user is seeking information about or attempting to make a booking in during each turn of the conversation. "
    "Select the most relevant domain from the following options: [wiki], [news]. "
    "Select [wiki] for queries about knowledge."
    "Select [news] for queries on recent events."
    "If the user's inquiry does not align with a specific domain, use: [general]. "
)

domain_prefix = "<domain>"
domain_suffix = "</domain>"
DOMAIN_PREDICTION_EXAMPLES = [
    (
        "\nuser: hi, can you tell me when was CUDA created?"
        "\nassistant: " + domain_prefix + "[wiki]" + domain_suffix
    ),
    (
        "\nuser: what are some recent corporate events by NVIDIA?"
        "\nassistant: " + domain_prefix + "[news]" + domain_suffix
    ),
    (
        "\nuser: okay, thank you . have a good day !"
        "\nassistant: " + domain_prefix + "[general]" + domain_suffix
    ),
]

DOMAIN_TO_FUNCTION_MAPPING =  {
    "[wiki]": ["search_wiki"],
    "[news]": ["search_news"],
}

from executable_functions import do_search_wiki, do_search_news
FUNCTION_SCHEMA = {
    "search_wiki": {
        "type": "function",
        "name": "search_wiki",
        "description": "Search wikipedia for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search wikipedia for"
                },
                "num_results": {
                    "type": "integer",
                    "description": "The number of results to return"
                }
            },
            "required": ["query"]
        },
        "executable_fn": do_search_wiki
    },
    "search_news": {
        "type": "function",
        "name": "search_news", 
        "description": "Search news for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search news for"
                },
                "num_results": {
                    "type": "integer",
                    "description": "The number of results to return"
                }
            },
            "required": ["query"]
        },
        "executable_fn": do_search_news
    }
}

TOD_INSTRUCTION = "You are a task-oriented assistant. You can use the given functions to fetch further data to help the users."

TOD_NOTES = [
    "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    "Use only the argument values explicitly provided or confirmed by the user instead of the assistant. Don't add or guess argument values.",
    "Ensure the accuracy of arguments when calling functions to effectively obtain information of entities requested by the user.",
]

RESPONSE_GENERATION_SYSTEM_MESSAGE = (
    "Please generate a helpful response to the user based on the dialogue history and returned values from the functions if present."
)

USER_INPUT_PREFIX = "> User: "
ASSISTANT_RESPONSE_PREFIX = "> Assistant: "
DEBUG = False

"----------------------------------------------------"


import argparse
import sys
from tqdm import tqdm
import os
from datetime import datetime
sys.path.append(".")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dialogue_save_dir", type=str, default="outputs/fnchat_dialogues")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def log_to_file(func):
    def wrapper(*args, **kwargs):
        # Get the return value from the original function
        return_value = func(*args, **kwargs)
        
        # Print function call details to terminal only if DEBUG is True
        if globals().get('DEBUG', False):
            print("\n[INTERMEDIATE LOG]")
            print(f"    Function: {func.__name__}")
            print(f"    Arguments: {args}, {kwargs}")
            print(f"    Return Value:")
            # Handle multi-line return values with proper indentation
            if isinstance(return_value, str):
                for line in return_value.split('\n'):
                    print(f"        {line}")
            else:
                print(f"        {return_value}")
            print(f"    Time: {datetime.now().strftime('%Y%m%d-%H%M%S')}")
            print("-" * 50 + "\n")
        
        return return_value
    return wrapper


def main(args):
    os.makedirs(args.dialogue_save_dir, exist_ok=True)
    save_path = os.path.join(args.dialogue_save_dir, f"{args.model_name_or_path}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
    print("-"*100)
    print(f"Welcome to FnChat! You are using {args.model_name_or_path} as your LLM.")
    print("You can start the conversation now. Type 'exit' to end the conversation.")
    print("-"*100 + '\n')

    all_turns = []
    while True:
        user_input = input(USER_INPUT_PREFIX)
        if user_input == "exit":
            break
        all_turns.append({"content": user_input, "role": "user"})

        """Step 1: Domain Prediction (DP)"""
        messages_for_dp = make_domain_prediction_messages(
            DOMAIN_PREDICTION_SYSTEM_MESSAGE,
            DOMAIN_PREDICTION_EXAMPLES,
            all_turns
        )

        responses_for_dp = get_model_response(
            messages=messages_for_dp, 
            model_name_or_path=args.model_name_or_path,
            temperature=args.temperature,
            max_tokens=8,
            n_seqs=1,
            continue_final_message=True
        )

        turn_domain = parse_response_to_predicted_domain(responses_for_dp[0]['content'])
        candidate_functions, current_function = get_functions_in_domain(turn_domain, DOMAIN_TO_FUNCTION_MAPPING, FUNCTION_SCHEMA)
        fnc_name, fnc_arguments, fnc_results = None, None, None

        if candidate_functions is not None:
            """Step 2: Dialogue State Tracking (DST)"""
            messages_for_dst = make_dialogue_state_tracking_messages(
                TOD_INSTRUCTION,
                TOD_NOTES,
                all_turns
            )

            responses_for_dst = get_model_response(
                messages=messages_for_dst, 
                model_name_or_path=args.model_name_or_path,
                temperature=args.temperature,
                functions=candidate_functions,
                function_call=current_function,
                max_tokens=128,
                n_seqs=1,
                continue_final_message=True if current_function else False
            )
            fnc_name, fnc_arguments = parse_response_to_predicted_function(
                responses_for_dst[0], 
                current_function, 
                args.model_name_or_path,
            )

            """Step 3: Function Execution"""
            if check_executable(fnc_name, fnc_arguments, FUNCTION_SCHEMA):
                fnc_results = execute_function(fnc_name, fnc_arguments, FUNCTION_SCHEMA)
            
        """Step 4: Generate Response"""
        messages_for_response = make_response_generation_messages(
            fnc_name,
            fnc_arguments,
            fnc_results,
            all_turns,
        )

        response_for_response = get_model_response(
            messages=messages_for_response, 
            model_name_or_path=args.model_name_or_path,
            temperature=args.temperature,
            max_tokens=512,
            n_seqs=1,
        )

        response = response_for_response[0]['content']
        print(f"\n{ASSISTANT_RESPONSE_PREFIX}{response}\n")

        all_turns.append({"content": response, "role": "assistant"})
        with open(save_path, "w") as f:
            json.dump(all_turns, f)

        
def make_domain_prediction_messages(
    dp_system_message, 
    dp_examples,
    all_turns
):
    messages = []
    system_message = dp_system_message
    for dp_example in dp_examples:
        system_message += '\n<EXAMPLE>' + dp_example + '\n</EXAMPLE>\n'

    messages.append({"role": "system", "content": system_message})
    messages.extend(all_turns)
    messages.append({"role": "assistant", "content": domain_prefix + "["})
    return messages

import openai
from helper_functions import transformers_chat_completion
@log_to_file
def get_model_response(
    messages, 
    model_name_or_path, 
    max_tokens, 
    n_seqs, 
    functions=None, 
    function_call=None,
    temperature=0.0,
    continue_final_message=False
):
    params = {
        "model": model_name_or_path,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": n_seqs,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": [],
    }
    if functions:
        params["functions"] = functions
    if function_call:
        params["function_call"] = {"name": function_call["name"]}
    # print("params for generation:", params)
    # breakpoint()
    if model_name_or_path in SUPPORTED_OPENAI_MODELS:
        response = openai.ChatCompletion.create(**params)
        candidates = response["choices"]
        candidates = [candidate["message"] for candidate in candidates]
    elif model_name_or_path in SUPPORTED_QWEN_MODELS: # Serving model locally
        params["stop"] += ['<|im_end|>']
        params["continue_final_message"] = continue_final_message
        candidates = transformers_chat_completion(**params)
    return candidates

def parse_response_to_predicted_domain(response):
    turn_domain = ""
    for d in [
        "wiki",
        "news",
        "general",
    ]:
        if d in response:
            turn_domain = "[" + d + "]"

    return turn_domain

@log_to_file
def get_functions_in_domain(
    turn_domain, 
    domain_to_function_map, 
    function_schema
):
    functions = []
    if turn_domain not in domain_to_function_map:
        return None, None
    for fname in domain_to_function_map[turn_domain]:
        # Create a copy of the function schema without executable_fn
        fn_schema = function_schema[fname].copy()
        fn_schema.pop("executable_fn")
        functions.append(fn_schema)
    
    current_function = functions[0] if functions else None
    return functions, current_function

def make_dialogue_state_tracking_messages(
    dst_system_message,
    dst_notes,
    all_turns
):
    messages = []
    system_message = dst_system_message + "\n" + "\n".join(dst_notes)
    messages.append({"role": "system", "content": system_message})
    messages.extend(all_turns)
    return messages

import json
@log_to_file
def parse_response_to_predicted_function(
    response, 
    current_function, 
    model_name_or_path,
):
    fnc_name, fnc_arguments = None, None
    if model_name_or_path in SUPPORTED_OPENAI_MODELS:
        if "function_call" in response:
            fnc_name = response["function_call"]["name"]
            fnc_arguments = json.loads(response["function_call"]["arguments"])
        else:
            print("Can not parse:", response)
            fnc_name, fnc_arguments = None, None
    else:
        pass
    return fnc_name, fnc_arguments

@log_to_file
def check_executable(fnc_name, fnc_arguments, function_schema):
    if fnc_name is None or fnc_arguments is None or fnc_name not in function_schema:
        return False
    for required_param in function_schema[fnc_name]["parameters"]["required"]:
        if required_param not in fnc_arguments:
            return False
    for predicted_param in fnc_arguments:
        if predicted_param not in function_schema[fnc_name]["parameters"]["properties"]:
            return False
    return True

@log_to_file
def execute_function(fnc_name, fnc_arguments, function_schema):
    call_results = function_schema[fnc_name]["executable_fn"](**fnc_arguments)
    return call_results

def make_response_generation_messages(
    fnc_name,
    fnc_arguments,
    fnc_results,
    all_turns,
):
    messages = []
    system_message = RESPONSE_GENERATION_SYSTEM_MESSAGE
    if fnc_name is not None:
        system_message += f"\nFunction: {fnc_name}\nArguments: {fnc_arguments}\nReturned Results:\n {fnc_results}\n"
    messages.append({"role": "system", "content": system_message})
    messages.extend(all_turns)
    return messages

if __name__ == "__main__":
    args = parse_args()
    DEBUG = args.debug
    main(args)
    
