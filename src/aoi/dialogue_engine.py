"""
==========Project AOI==========

This file contains the implementation of dialogue engines for AOI. A dialogue engine: 
* takes text (& image) as input at every turn.
* produces a text output at every turn.

Internally, a dialogue engine keeps track of:
* the dialogue history an dialogue states of the current session.
* all the previous conversation sessions. i.e., it is responsible for archiving conversations.
* a set of memory items (e.g., in json files).

=====Jinghong Chen, 2025.6.=====
"""

import openai
import os
import json
from datetime import datetime

"""================ HYPER-PARAMETERS / PROMPTS ==============="""

WORKFLOW_SELECTION_PROMPT = ()

SUPPORTED_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-0125", "gpt-4.1-nano"]
SUPPORTED_QWEN_MODELS = ["Qwen/Qwen2.5-3B-Instruct"]

EXPERIMENT_DOMAINS = ["[general]", "[wiki]", "[news]", "[cook]", "[bocha]", "[recall]"]

# Locale-specific prompts
PROMPTS = {
    "EN": {
        "domain_prediction_system_message": (
            "You are a task-oriented assistant named AOI (Assistive Open Intelligence) built by Jinghong Chen. Your response should be oral and brief. "
            "Your role is to determine which domain the user is seeking information about or attempting to make a booking in during each turn of the conversation. "
            "Select the most relevant domain from the following options: [wiki], [news], [cook]. "
            "Select [wiki] for queries about knowledge."
            "Select [news] for queries on recent events."
            "Select [cook] for queries about recipes and cooking."
            "Select [bocha] for queries about other web search."
            "Select [recall] for queries about recalling user memories."
            "If the user's inquiry does not align with a specific domain, use: [general]. "
        ),
        "tod_instruction": "You are a task-oriented assistant. You can use the given functions to fetch further data to help the users.",
        "tod_notes": [
            "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
            "Use only the argument values explicitly provided or confirmed by the user instead of the assistant. Don't add or guess argument values.",
            "Ensure the accuracy of arguments when calling functions to effectively obtain information of entities requested by the user.",
        ],
        "response_generation_system_message": (
            "Please generate a helpful response to the user based on the dialogue history and returned values from the functions if present."
        ),

        "recall_reflection_system_message": (
            "Your task is to recall a piece of memory of the user from the memory store. The user will provide some hints."
            "Think about what useful information you need to locate the memory and also what information you can infer from the user utterance."
            "You should generate a step-by-step plan to retrieve the memory. The plan should contains bullet points in the following structure: \n[Step 1] ...\n, \n[Step 2] ...\n, etc. Each step should be a single line."
            "A list of available functions is provided below. You can use any of them to retrieve the memory in your plan."
        ),
        "recall_act_system_message": (
            "Your task is to recall a piece of memory of the user from the memory store. "
            "You should act according to the current step in the plan."
            "Use only the argument values explicitly provided or confirmed by the user instead of the assistant. Don't add or guess argument values."
            "Ensure the accuracy of arguments when calling functions to effectively obtain information of entities requested by the user."
        ),
        "recall_response_system_message": (
            "You are a task-oriented assistant. Provided with the retrieved memory, address the user's query directly and succinctly."
        )
    },
    "ZH": {
        "domain_prediction_system_message": (
            "你是小蓝，一个由陈镜鸿构建的任务导向助手。你的回答应该口语化且简洁。 "
            "你的角色是确定用户在对话的每个回合中寻求信息或尝试预订的领域。 "
            "从以下选项中选择最相关的领域：[wiki]，[news]，[cook], [bocha]。 "
            "选择[wiki]用于知识查询。"
            "选择[news]用于最近事件的查询。"
            "选择[cook]用于菜谱和烹饪查询。"
            "选择[bocha]用于其他网络搜索。"
            "如果用户的询问与特定领域不符，请使用：[general]。 "
        ),
        "tod_instruction": "你是一个任务导向的助手。你可以使用给定的函数来获取更多数据来帮助用户。",
        "tod_notes": [
            "不要对函数参数值做假设。如果用户请求不明确，请要求澄清。",
            "只使用用户明确提供或确认的参数值，而不是助手的假设。不要添加或猜测参数值。",
            "确保调用函数时参数的准确性，以有效获取用户请求的实体信息。",
        ],
        "response_generation_system_message": (
            "请根据对话历史和函数返回值（如果存在）为用户生成有用的回答。"
        )
    }
}

domain_prefix = "<domain>"
domain_suffix = "</domain>"

# Locale-specific examples
DOMAIN_PREDICTION_EXAMPLES = {
    "EN": [
        (
            "\nuser: hi, can you tell me when was CUDA created?"
            "\nassistant: " + domain_prefix + "[wiki]" + domain_suffix
        ),
        (
            "\nuser: what are some recent corporate events by NVIDIA?"
            "\nassistant: " + domain_prefix + "[news]" + domain_suffix
        ),
        (
            "\nuser: what is the weather in Beijing?"
            "\nassistant: " + domain_prefix + "[bocha]" + domain_suffix
        ),
        (
            "\nuser: what did I browse on the internet yesterday?"
            "\nassistant: " + domain_prefix + "[recall]" + domain_suffix
        ),
        (
            "\nuser: okay, thank you . have a good day !"
            "\nassistant: " + domain_prefix + "[general]" + domain_suffix
        ),
    ],
    "ZH": [
        (
            "\nuser: 你好，能告诉我CUDA是什么时候创建的吗？"
            "\nassistant: " + domain_prefix + "[wiki]" + domain_suffix
        ),
        (
            "\nuser: 英伟达最近有什么企业活动？"
            "\nassistant: " + domain_prefix + "[news]" + domain_suffix
        ),
        (
            "\nuser: 怎么做蛋炒饭？"
            "\nassistant: " + domain_prefix + "[cook]" + domain_suffix
        ),
        (
            "\nuser: 好的，谢谢，祝你有美好的一天！"
            "\nassistant: " + domain_prefix + "[general]" + domain_suffix
        ),
        (
            "\nuser: 最近股市怎么样?"
            "\nassistant: " + domain_prefix + "[bocha]" + domain_suffix
        ),
    ]
}

DOMAIN_TO_FUNCTION_MAPPING =  {
    "[wiki]": ["search_wiki"],
    "[news]": ["search_news"],
    "[cook]": ["search_cookbook"],
    "[bocha]": ["search_bocha"],
    "[recall]": ["recall_memory"],
}

from aoi.executable_functions import do_search_wiki, do_search_news, do_search_cookbook, do_search_bocha, do_recall_memory

# Locale-specific function schemas
FUNCTION_SCHEMAS = {
    "EN": {
        "search_wiki": {
            "type": "function",
            "name": "search_wiki",
            "function": {
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
                }
            },
            "executable_fn": do_search_wiki
        },
        "search_news": {
            "type": "function",
            "name": "search_news", 
            "function": {
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
                }
            },
            "executable_fn": do_search_news
        },
        "search_cookbook": {
            "type": "function",
            "name": "search_cookbook",
            "function": {
                "name": "search_cookbook",
                "description": "Search for recipes and cooking instructions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The recipe or cooking query to search for"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "The number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_cookbook
        },
        "search_bocha": {
            "type": "function",
            "name": "search_bocha",
            "function": {
                "name": "search_bocha",
                "description": "Search the web using Bocha AI search engine",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "The number of results to return (max 50)"
                        },
                        "freshness": {
                            "type": "string",
                            "description": "Time range filter: oneDay, oneWeek, oneMonth, oneYear, noLimit",
                            "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
                        },
                        "include": {
                            "type": "string",
                            "description": "Site restrictions (e.g., 'qq.com|m.163.com')"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_bocha
        },
        "recall_memory": {
            "type": "function",
            "name": "recall_memory",
            "function": {
                "name": "recall_memory",
                "description": "Recall a piece of memory from the memory store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search the memory store for"
                        },
                        "time_range": {
                            "type": "string",
                            "description": "The time range to search the memory store for",
                            "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
                        },
                        "source_type": {
                            "type": "string",
                            "description": "The type of the source of the memory",
                            "enum": ["active", "passive"]
                        },
                        "tags": {
                            "type": "array",
                            "description": "The tags of the memory",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_recall_memory
        }
    },
    "ZH": {
        "search_wiki": {
            "type": "function",
            "name": "search_wiki",
            "function": {
                "name": "search_wiki",
                "description": "搜索维基百科获取信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要搜索维基百科的查询"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "要返回的结果数量"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_wiki
        },
        "search_news": {
            "type": "function",
            "name": "search_news", 
            "function": {
                "name": "search_news", 
                "description": "搜索新闻获取信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要搜索新闻的查询"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "要返回的结果数量"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_news
        },
        "search_cookbook": {
            "type": "function",
            "name": "search_cookbook",
            "function": {
                "name": "search_cookbook",
                "description": "搜索菜谱和烹饪方法",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要搜索的菜谱或烹饪查询"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "要返回的结果数量"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_cookbook
        },
        "search_bocha": {
            "type": "function",
            "name": "search_bocha",
            "function": {
                "name": "search_bocha",
                "description": "使用Bocha AI搜索引擎搜索网络",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "要返回的结果数量（最多50个）"
                        },
                        "freshness": {
                            "type": "string",
                            "description": "时间范围过滤器：oneDay（一天内）, oneWeek（一周内）, oneMonth（一个月内）, oneYear（一年内）, noLimit（不限）",
                            "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]
                        },
                        "include": {
                            "type": "string",
                            "description": "网站限制（例如：'qq.com|m.163.com'）"
                        }
                    },
                    "required": ["query"]
                }
            },
            "executable_fn": do_search_bocha
        }
    }
}

"""======================= Helper Functions ===================="""
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
        params["tools"] = functions
    if function_call:
        params["tool_choice"] = {"type": "function", "function": {"name": function_call["name"]}}
    
    if model_name_or_path in SUPPORTED_OPENAI_MODELS:
        # Initialize client with API key from environment or file
        try:
            client = openai.OpenAI()
        except Exception as e:
            raise Exception(f"Error initializing OpenAI client: {e}")
        
        response = client.chat.completions.create(**params)
        candidates = []
        for choice in response.choices:
            candidate = {
                "content": choice.message.content
            }
            
            # Handle tool calls (function calls) in new API
            if choice.message.tool_calls:
                # Get the first tool call
                tool_call = choice.message.tool_calls[0]
                candidate["function_call"] = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            else:
                candidate["function_call"] = None
            
            candidates.append(candidate)
    else:
        raise NotImplementedError(f"Model {model_name_or_path} not supported")
    return candidates

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

def parse_response_to_predicted_domain(response):
    turn_domain = ""
    for d in [
        "wiki",
        "news",
        "cook",
        "bocha",
        "recall",
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
        print(f"DEBUG: Parsing response: {response}")
        function_call = response.get("function_call")
        
        if function_call is not None:
            # Handle consistent structure from _get_model_response
            print(f"DEBUG: Function call object: {function_call}")
            
            if isinstance(function_call, dict) and "name" in function_call:
                fnc_name = function_call["name"]
                fnc_arguments = json.loads(function_call["arguments"])
                print(f"DEBUG: Parsed function call - name: {fnc_name}, args: {fnc_arguments}")
            else:
                print("Can not parse function call:", function_call)
                fnc_name, fnc_arguments = None, None
        else:
            print("No function call in response (function_call is None)")
            fnc_name, fnc_arguments = None, None
    else:
        pass
    return fnc_name, fnc_arguments

@log_to_file
def check_executable(fnc_name, fnc_arguments, function_schema):
    if fnc_name is None or fnc_arguments is None or fnc_name not in function_schema:
        return False
    for required_param in function_schema[fnc_name]["function"]["parameters"]["required"]:
        if required_param not in fnc_arguments:
            return False
    for predicted_param in fnc_arguments:
        if predicted_param not in function_schema[fnc_name]["function"]["parameters"]["properties"]:
            return False
    return True

@log_to_file
def execute_function(fnc_name, fnc_arguments, function_schema, locale="EN"):
    # Check if the function supports locale parameter
    if fnc_name == "search_cookbook" and "locale" in function_schema[fnc_name]["executable_fn"].__code__.co_varnames:
        fnc_arguments["locale"] = locale
    
    call_results = function_schema[fnc_name]["executable_fn"](**fnc_arguments)
    return call_results

def make_response_generation_messages(
    fnc_name,
    fnc_arguments,
    fnc_results,
    all_turns,
    locale="EN"
):
    messages = []
    system_message = PROMPTS[locale]["response_generation_system_message"]
    if fnc_name is not None:
        system_message += f"\nFunction: {fnc_name}\nArguments: {fnc_arguments}\nReturned Results:\n {fnc_results}\n"
    messages.append({"role": "system", "content": system_message})
    messages.extend(all_turns)
    return messages

def make_recall_reflection_messages(
    reflection_system_message,
    all_turns,
    locale="EN"
):
    recall_fn_names = [fnc_name for fnc_name in DOMAIN_TO_FUNCTION_MAPPING["[recall]"]]
    available_recall_functions = "\n\n".join([json.dumps(fnc_schema["function"]) for fnc_name, fnc_schema in FUNCTION_SCHEMAS[locale].items() if fnc_name in recall_fn_names])
    reflection_system_message += f"\nAvailable functions: {available_recall_functions}\n"
    messages = []
    messages.append({"role": "system", "content": reflection_system_message})
    print(f"[DEBUG] Reflection system message: {reflection_system_message}")
    messages.extend(all_turns)
    return messages

def make_recall_act_messages(
    act_system_message,
    retrieval_step,
    all_turns,
    locale="EN"
):
    sys_msg = act_system_message
    sys_msg += f"\n\nCurrent step: {retrieval_step}"
    messages = []
    messages.append({"role": "system", "content": sys_msg})
    messages.extend(all_turns)
    return messages

def parse_recall_retrieval_plan(response):
    retrieval_plan = []
    for line in response.split("\n"):
        if line.startswith("[Step"):
            retrieval_plan.append(line)
    return retrieval_plan

def make_recall_response_generation_messages(
    recall_response_system_message,
    retrieved_memory,
    all_turns,
    locale="EN"
):
    sys_msg = f"{recall_response_system_message}\n\nRetrieved memories:\n{retrieved_memory}"
    messages = []
    messages.append({"role": "system", "content": sys_msg})
    messages.extend(all_turns)
    return messages

"""======================= AOI Dialogue Engine ===================="""

class AOIDialogueEngine:
    def __init__(self, model_name, save_dir, api_key_file=None, locale="EN"):
        self.model_name = model_name
        self.save_dir = save_dir
        self._session_save_fpath = os.path.join(save_dir, 'all_sessions.jsonl')
        self._turns_save_fpath = os.path.join(save_dir, 'all_turns.jsonl')
        self._memory_save_fpath = os.path.join(save_dir, 'memories.jsonl')
        self._api_key_file = api_key_file
        self.locale = locale.upper() if locale else "EN"
        print(f"DEBUG: Locale: {self.locale}")
        
        # Validate locale
        if self.locale not in PROMPTS:
            raise ValueError(f"Unsupported locale: {locale}. Supported locales: {list(PROMPTS.keys())}")
        
        self._init_session_states()
        self._init_model()
    
    def _init_session_states(self):
        self._all_turns = []
        self._session_history = []

    def _init_model(self):
        """Initialize OpenAI client with API key"""
        try:
            if self._api_key_file:
                with open(self._api_key_file, 'r') as f:
                    api_key = f.read().strip()
                if not api_key:
                    raise ValueError("API key file is empty")
                self.client = openai.OpenAI(api_key=api_key)
            else:
                # Try to use environment variable
                self.client = openai.OpenAI()
        except Exception as e:
            raise Exception(f"Error initializing OpenAI client: {e}")
    
    def _get_model_response(
        self,
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
            params["tools"] = functions
        if function_call:
            params["tool_choice"] = {"type": "function", "function": {"name": function_call["name"]}}
        
        print(f"DEBUG: API params: {params}")
        
        if model_name_or_path in SUPPORTED_OPENAI_MODELS:
            response = self.client.chat.completions.create(**params)
            print(f"DEBUG: Raw API response: {response}")
            print(f"DEBUG: Response choices: {response.choices}")
            
            candidates = []
            for i, choice in enumerate(response.choices):
                print(f"DEBUG: Processing choice {i}: {choice}")
                print(f"DEBUG: Choice message: {choice.message}")
                print(f"DEBUG: Choice tool_calls: {choice.message.tool_calls}")
                
                candidate = {
                    "content": choice.message.content
                }
                
                # Handle tool calls (function calls) in new API
                if choice.message.tool_calls:
                    # Get the first tool call
                    tool_call = choice.message.tool_calls[0]
                    print(f"DEBUG: Tool call: {tool_call}")
                    print(f"DEBUG: Tool call function: {tool_call.function}")
                    
                    candidate["function_call"] = {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                    print(f"DEBUG: Created function_call: {candidate['function_call']}")
                else:
                    candidate["function_call"] = None
                    print(f"DEBUG: No tool calls, setting function_call to None")
                
                candidates.append(candidate)
                print(f"DEBUG: Final candidate: {candidate}")
        else:
            raise NotImplementedError(f"Model {model_name_or_path} not supported")
        
        print(f"DEBUG: Returning candidates: {candidates}")
        return candidates
    
    def start_new_session(self):
        self._init_session_states()

    def _single_function_call_pipeline(self, turn_domain):
        candidate_functions, current_function = get_functions_in_domain(turn_domain, DOMAIN_TO_FUNCTION_MAPPING, FUNCTION_SCHEMAS[self.locale])
        fnc_name, fnc_arguments, fnc_results = None, None, None

        if candidate_functions is not None:
            """Step 2: Dialogue State Tracking (DST)"""
            messages_for_dst = make_dialogue_state_tracking_messages(
                PROMPTS[self.locale]["tod_instruction"],
                PROMPTS[self.locale]["tod_notes"],
                self._all_turns
            )

            print(f"DEBUG: Candidate functions: {candidate_functions}")
            print(f"DEBUG: Current function: {current_function}")
            responses_for_dst = self._get_model_response(
                messages=messages_for_dst, 
                model_name_or_path=self.model_name,
                temperature=0.3,
                functions=candidate_functions,
                function_call=current_function,
                max_tokens=128,
                n_seqs=1,
                continue_final_message=True if current_function else False
            )
            fnc_name, fnc_arguments = parse_response_to_predicted_function(
                responses_for_dst[0], 
                current_function, 
                self.model_name,
            )
            self._session_history.append({"type": "function_argument_filling", "content": {"fnc_name": fnc_name, "fnc_arguments": fnc_arguments}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})

            """Step 3: Function Execution"""
            if check_executable(fnc_name, fnc_arguments, FUNCTION_SCHEMAS[self.locale]):
                fnc_results = execute_function(fnc_name, fnc_arguments, FUNCTION_SCHEMAS[self.locale], locale=self.locale)
                self._session_history.append({"type": "function_execution", "content": {"fnc_name": fnc_name, "fnc_arguments": fnc_arguments, "fnc_results": fnc_results}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})
            
        """Step 4: Generate Response"""
        messages_for_response = make_response_generation_messages(
            fnc_name,
            fnc_arguments,
            fnc_results,
            self._all_turns,
            locale=self.locale
        )
        self._session_history.append({"type": "response_generation", "content": {"fnc_name": fnc_name, "fnc_arguments": fnc_arguments, "fnc_results": fnc_results}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})

        response_for_response = self._get_model_response(
            messages=messages_for_response, 
            model_name_or_path=self.model_name,
            temperature=0.3,
            max_tokens=512,
            n_seqs=1,
        )

        response = response_for_response[0]['content']
        self._all_turns.append({"content": response, "role": "assistant"})
        self._session_history.append({"type": "response_generation", "content": {"response": response}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})
        
        return response
        

    def _recall_pipeline(self, turn_domain):
        # Reflect-Act
        """Step 1: Reflect on retrieval strategy"""
        messages_for_reflection = make_recall_reflection_messages(
            PROMPTS[self.locale]["recall_reflection_system_message"],
            self._all_turns
        )
        responses_for_reflection = self._get_model_response(
            messages=messages_for_reflection, 
            model_name_or_path=self.model_name,
            temperature=0.3,
            max_tokens=512,
            n_seqs=1,
        )

        retrieval_plan = parse_recall_retrieval_plan(responses_for_reflection[0]['content'])
        self._session_history.append({"type": "recall_retrieval_plan", "content": {"retrieval_plan": retrieval_plan}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})

        retrieved_memory = []
        print(f"DEBUG: Retrieval plan: {retrieval_plan}")
        """Step 2: Act on the retrieval plan"""
        for retrieval_step in retrieval_plan:
            candidate_functions, current_function = get_functions_in_domain(turn_domain, DOMAIN_TO_FUNCTION_MAPPING, FUNCTION_SCHEMAS[self.locale])
            print(f"DEBUG: Candidate functions: {candidate_functions}")
            print(f"DEBUG: Current function: {current_function}")
            print(f"DEBUG: Retrieval step: {retrieval_step}")

            messages_for_act = make_recall_act_messages(
                PROMPTS[self.locale]["recall_act_system_message"],
                retrieval_step,
                self._all_turns,
                self.locale
            )
            responses_for_act = self._get_model_response(
                messages=messages_for_act, 
                model_name_or_path=self.model_name,
                temperature=0.3,
                max_tokens=512,
                n_seqs=1,
                functions=candidate_functions,
                function_call=current_function,
            )
            self._session_history.append({"type": "recall_act", "content": {"retrieval_step": retrieval_step, "response": responses_for_act[0]}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})
            fnc_name, fnc_arguments = parse_response_to_predicted_function(
                responses_for_act[0], 
                current_function, 
                self.model_name,
            )
            fnc_results = ""
            if check_executable(fnc_name, fnc_arguments, FUNCTION_SCHEMAS[self.locale]):
                fnc_results = execute_function(fnc_name, fnc_arguments, FUNCTION_SCHEMAS[self.locale], locale=self.locale)
            
            print(f"DEBUG: Fnc results: {fnc_results}")
            self._session_history.append({"type": "recall_function_execution", "content": {"fnc_name": fnc_name, "fnc_arguments": fnc_arguments, "fnc_results": fnc_results}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})
            retrieved_memory.append(fnc_results)
            break #TODO 
        
        print(f"DEBUG: Retrieved memory: {retrieved_memory}")
        """Step 3: Generate Response"""
        messages_for_recall_response = make_recall_response_generation_messages(
            PROMPTS[self.locale]["recall_response_system_message"],
            "\n".join(retrieved_memory),
            self._all_turns,
            locale=self.locale
        )
        response_for_recall_response = self._get_model_response(
            messages=messages_for_recall_response, 
            model_name_or_path=self.model_name,
            temperature=0.3,
            max_tokens=1024,
            n_seqs=1,
        )

        return response_for_recall_response[0]['content']




        
    def run_turn(self, user_input):
        self._all_turns.append({"role": "user", "content": user_input})
        self._session_history.append({"type": "user_input", "content": user_input, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})

        """Step 1: Domain Prediction (DP)"""
        messages_for_dp = make_domain_prediction_messages(
            PROMPTS[self.locale]["domain_prediction_system_message"],
            DOMAIN_PREDICTION_EXAMPLES[self.locale],
            self._all_turns
        )

        responses_for_dp = self._get_model_response(
            messages=messages_for_dp, 
            model_name_or_path=self.model_name,
            temperature=0.3,
            max_tokens=8,
            n_seqs=1,
            continue_final_message=True
        )
        print(f"DEBUG: Responses for DP: {responses_for_dp}")
        turn_domain = parse_response_to_predicted_domain(responses_for_dp[0]['content'])
        self._session_history.append({"type": "domain_prediction", "content": {"turn_domain": turn_domain}, "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S")})

        """Step 2: Run respective pipeline"""
        response = ""
        if turn_domain in ["[general]", "[bocha]", "[news]", "[cook]", "[wiki]"]:
            response = self._single_function_call_pipeline(turn_domain)
        elif turn_domain in ["[recall]"]:
            response = self._recall_pipeline(turn_domain)
        else:
            response = "I'm sorry, I don't know how to answer that."

        """Logging to disk"""
        with open(self._session_save_fpath, "a") as f:
            json.dump(self._session_history, f)
            f.write("\n")
        with open(self._turns_save_fpath, "a") as f:
            json.dump(self._all_turns, f)
            f.write("\n")
        return response

