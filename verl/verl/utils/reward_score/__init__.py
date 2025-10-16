# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

import re
import os
try:
    from .client import ChatClient, OpenAIClient
except:
    from client import ChatClient, OpenAIClient


used_model = os.environ.get('USED_MODEL', None) # 'gpt'
if used_model in ['gpt-4o', 'gpt-4.1']:
    client = OpenAIClient(
        api_key=os.environ.get('OPENAI_API_KEY', None),
        api_base=os.environ.get('OPENAI_API_BASE', None),
        model=used_model,
        max_tokens=12288,
    )
elif used_model == 'no_api':
    print("We do not use API for evaluation")
else:
    client_ip = os.environ.get('CLIENT_IP', 'http://127.0.0.1:8001')
    print(client_ip)
    client = ChatClient(server_url=client_ip, model=used_model)


mc_prompt = '''**Task:**  
Given a **question** and a **model_response** to a multiple-choice problem, extract the exact option (e.g., A, B, C, D) that the model selects as its answer. If the response does not explicitly state an option but implies an answer, infer the most likely choice. If no clear option is provided, return "Unclear."

### Example 1:  
- **Question**: ```What is the capital of France? (A) London (B) Paris (C) Berlin (D) Rome```  
- **Model Response**: ```The correct answer is Paris, which corresponds to option B.```  
- **Extracted Option**: \\boxed{{B}}

### Example 2:  
- **Question**: ```Which element has the chemical symbol 'O'? (A) Gold (B) Oxygen (C) Carbon (D) Iron```  
- **Model Response**: ```Oxygen is the element with symbol 'O'.```  
- **Extracted Option**: \\boxed{{B}} 

### Your Task:  
- **Question**: ```{question}```
- **Model Response**: ```{model_response}```
NOTE: Please do not try to solve the problem or provide an answer! Instead, focus on extracting the option!
Please think step-by-step and put your final extracted option, i.e., A, B, C, etc in \\boxed{{}}.'''


math_prompt = '''**Task:**  
Given a **question** and a **model_response** to a math problem, extract the concise answer from the **model_response**.

Example 1. **Math (Symbolic):**  
- Model Answer: ```x = 2```  
- Ground Truth: ```2```  
- Output: \\boxed{{Y}}  

Example 2. **Math (Unit Conversion):**  
- Problem: ```Convert 300 seconds to minutes.```  
- Model Answer: ```5 minutes.```  
- Ground Truth: ```5```  
- Output: \\boxed{{Y}}


Example 2. **Math:**  
- Model Answer: ```x = \\frac{{1}}{{3}}```  
- Ground Truth: ```x = \\frac{{2}}{{3}}```  
- Output: \\boxed{{N}}

### Your Task:  
- **Question**: ```{question}```
- **Model Response**: ```{model_response}```
- **Ground Truth**: ```{ground_truth}```

NOTE: Please do NOT try to solve the problem or provide an answer by yourself! Instead, focus on EXTRACT the origin answer from the **model_response** and then decide whether this answer is aligned with the ground truth (yes for aligned, no for not-aligned).
Please put your final extracted option, i.e., "Y" or "N" in \\boxed{{}}.'''



general_prompt = '''### User Question: {question}

### Ground Truth Answer: {ground_truth}

### Student Answer: {model_response}

For the above question, please verify if the student's answer is aligned with the ground truth answer.
Please do NOT try to solve the problem or provide an answer by yourself! Instead, just check if the student's answer is aligned with the ground truth answer (yes for aligned, no for not-aligned).
Please think step-by-step and put your final judgement, i.e., "Y" or "N" in \\boxed{{}}.'''



def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def get_raw_question_from_prompt(prompt_str):
    if 'user\n' in prompt_str:
        question = 'user\n'.join(prompt_str.split('user\n')[1:])
    elif 'User: ' in prompt_str:
        question = 'User: '.join(prompt_str.split('User: ')[1:])
    elif '<|im_start|>user\n' in prompt_str:
        question = '<|im_start|>user\n'.join(prompt_str.split('<|im_start|>user\n')[1:])
    else:
        raise ValueError(f"Invalid prompt format: {prompt_str}")
    if 'Question:\n' in question:
        question = question.split('Question:\n')[1]
    
    if '<|im_start|>assistant' in question:
        question = question.split('<|im_start|>assistant')[0]

    question = question.replace('Format your response as follows: "The correct answer is (insert answer here)"', '') # WebInstruct-verified-OC, gpqa_diamond-OC
    question = question.replace(r'Please reason step by step, and put your final answer within \boxed{}.', '') 
    question = question.strip()
    if 'What is the correct answer to this question: ' in question:
        question = question.replace('What is the correct answer to this question: ', '')
    if 'Please reason step by step, and put your final answer within \\boxed{{}}.' in question:
        question = question.replace('Please reason step by step, and put your final answer within \\boxed{{}}.', '')
    if 'Please only provide the letter of the answer in the box, e.g. \\boxed{A}.' in question:
        question = question.replace('Please only provide the letter of the answer in the box, e.g. \\boxed{A}.', '')
    if question.endswith('\nassistant\n'):
        question = question[:-len('\nassistant\n')]
    while '<|im_end|>' in question:
        question = question.replace('<|im_end|>', '')
    while '<|im_start|>' in question:
        question = question.replace('<|im_start|>', '')

    return question.strip()

def extract_option(judge_response):
    # Pattern to match \boxed{<valid option>}, where valid option is A-P
    pattern = r'\\boxed\{([a-z])\}'
    match = re.findall(pattern, judge_response)
    if match:
        return match[-1]  # Return the extracted option (A-P)
    return None  # Return None if no valid option is found


def format_reward(predict_str: str, prompt_str: str, format_mode='R1') -> float:
    def _validate_tags(input_string):
        if format_mode == 'R1':
            # tags = ['<think>', '</think>', '<answer>', '</answer>']
            tags = ['</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0


    if format_mode == 'R1':
        if _validate_tags(predict_str) == 0.0:
            return 0.0
        # pattern = re.compile(r'<think>.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>', re.DOTALL)
        pattern = re.compile(r'.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>', re.DOTALL)

    else:
        pattern = re.compile(r'.*\\boxed\{.*\}.*', re.DOTALL)
        if predict_str.count(r'\boxed{') != 1: # avoid the issue that the model repeats the boxed answer
            return 0.0
    match_result = re.fullmatch(pattern, predict_str)
    
    return 1.0 if match_result else 0.0

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    prompt_str=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    
    if '<answer>' in prompt_str and '</answer>' in prompt_str:
        format_mode = "R1"
    else:
        format_mode = "boxed"
    format_score = format_reward(solution_str, prompt_str, format_mode=format_mode)
    if format_score == 0.0:
        return 0.0
    
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in ["DeepMath-103K"] or any(dataset_name in data_source for dataset_name in ["Math-500", "MATH500", "AMC2023", "OlympiadBench", "Minerva", "TheoremQA", "AIME2024", "AIME2025", "MMLUPro"]):
        from . import math_verify

        res = math_verify.compute_score(solution_str, ground_truth)
        # from . import math

        # res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    prompt_str=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, prompt_str
    )


__all__ = ["default_compute_score"]
