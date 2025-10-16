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

from collections import defaultdict
import os
import re

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

# Import client modules
try:
    from verl.utils.reward_score.client import ChatClient, OpenAIClient
except:
    from client import ChatClient, OpenAIClient

from verl.utils.reward_score import last_boxed_only_string, remove_boxed, format_reward

@register("llm")
class LLMRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", use_batch_api=True) -> None:
        """
        Initialize the LLMRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            use_batch_api: Whether to use batch API calls for efficiency. Defaults to True.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.use_batch_api = use_batch_api
        
        # Initialize API client if using batch API
        if self.use_batch_api:
            self.used_model = os.environ.get('USED_MODEL', None)
            if self.used_model in ['gpt-4o', 'gpt-4.1']:
                self.client = OpenAIClient(
                    api_key=os.environ.get('OPENAI_API_KEY', None),
                    api_base=os.environ.get('OPENAI_API_BASE', None),
                    model=self.used_model,
                    max_tokens=12288,
                )
            elif self.used_model == 'no_api':
                print("We do not use API for evaluation")
                self.client = None
            else:
                client_ip = os.environ.get('CLIENT_IP', 'http://127.0.0.1:8001')
                print(f"Initializing ChatClient with {client_ip}")
                self.client = ChatClient(server_url=client_ip, model=self.used_model)
        else:
            self.client = None
    
    def get_raw_question_from_prompt(self, prompt_str):
        """Extract raw question from prompt string"""
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

        question = question.replace('Format your response as follows: "The correct answer is (insert answer here)"', '')
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

    def extract_option(self, judge_response):
        """Extract option from judge response"""
        pattern = r'\\boxed\{([a-z])\}'
        match = re.findall(pattern, judge_response)
        if match:
            return match[-1]
        return None

    def extract_option_from_general_verifier(self, judge_response):
        """Extract Yes or No from 'Final Decision: Yes' or 'Final Decision: No' in judge response"""
        pattern = r'Final Decision:\s*(Yes|No)'
        match = re.findall(pattern, judge_response, re.IGNORECASE)
        if match:
            return match[-1]
        return None

    def get_prompt_template(self, data_source):
        """Get appropriate prompt template based on data source"""
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

        general_verifier_prompt = '''User: ### Question: {question}

### Ground Truth Answer: {ground_truth}

### Student Answer: {model_response}

For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:'''

        if any(dataset_name in data_source for dataset_name in ["Math-500", "AMC2023", "OlympiadBench", "Minerva", "TheoremQA", "AIME2024", "AIME2025"]):
            return math_prompt
        elif 'gpqa_diamond' in data_source or 'MMLUPro' in data_source or 'SuperGPQA' in data_source:
            # return mc_prompt
            return general_verifier_prompt
        elif 'WebInstruct-verified' in data_source:
            # return general_prompt
            return general_verifier_prompt
        else:
            raise NotImplementedError(f"data_source={data_source} not in the implementation")
    
    def batch_compute_scores(self, batch_data):
        """Compute scores using batch API calls"""
        if not self.client or self.used_model == 'no_api':
            # Fallback to original computation
            return [self.compute_score(
                data_source=item['data_source'],
                solution_str=item['solution_str'],
                ground_truth=item['ground_truth'],
                extra_info=item['extra_info'],
                prompt_str=item['prompt_str']
            ) for item in batch_data]
        
        # Pre-process: check boxed answers and filter out items that should get 0.0
        results = [None] * len(batch_data)
        valid_items = []  # Items that need API processing
        
        for i, item in enumerate(batch_data):
            if '<answer>' in item['prompt_str'] and '</answer>' in item['prompt_str']:
                format_mode = "R1"
            else:
                format_mode = "boxed"
            format_score = format_reward(item['solution_str'], item['prompt_str'], format_mode=format_mode)
            if format_score == 0.0:
                results[i] = 0.0
                continue
            answer_str = remove_boxed(last_boxed_only_string(item['solution_str']))
            if answer_str is None:
                # If boxed format exists but can't extract answer, score is 0.0
                results[i] = 0.0
                continue
            # Store extracted answer for later use
            item['extracted_answer'] = answer_str
            valid_items.append((i, item))

            # # First check if boxed answer exists and is valid (following __init__.py logic)
            # if '\\boxed{' in item['prompt_str']:
            #     answer_str = remove_boxed(last_boxed_only_string(item['solution_str']))
            #     if answer_str is None:
            #         # If boxed format exists but can't extract answer, score is 0.0
            #         results[i] = 0.0
            #         continue
            #     # Store extracted answer for later use
            #     item['extracted_answer'] = answer_str
            # else:
            #     item['extracted_answer'] = None
            
            # # This item needs API processing
            # valid_items.append((i, item))
        
        # If no items need API processing, return early
        if not valid_items:
            return results
        
        # Group valid items by data_source to use the same prompt template
        data_by_source = defaultdict(list)
        for i, item in valid_items:
            data_by_source[item['data_source']].append((i, item))
        
        for data_source, items in data_by_source.items():
            try:
                selected_prompt = self.get_prompt_template(data_source)
                
                # Prepare batch prompts
                batch_prompts = []
                for idx, item in items:
                    question = self.get_raw_question_from_prompt(item['prompt_str'])
                    model_response = item['solution_str']
                    if 'user\n' in model_response:
                        model_response = model_response.split('user\n')[0]
                    
                    if 'WebInstruct-verified' in data_source or "gpqa_diamond" in data_source or "MMLUPro" in data_source:
                        if item['extracted_answer'] is not None:
                            model_response = item['extracted_answer']
                    
                    sent_prompt = selected_prompt.format(
                        question=question, 
                        ground_truth=item['ground_truth'], 
                        model_response=model_response
                    )
                    batch_prompts.append(sent_prompt)
                
                # Make batch API call
                if self.used_model in ['gpt-4o', 'gpt-4.1']:
                    responses = self.client.batch_chat_sync(batch_prompts, model=self.used_model)
                else:
                    try:
                        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                        responses = self.client.batch_chat(batch_messages, max_tokens=1024)
                        responses = [resp["choices"][0]["message"]["content"] if resp else None for resp in responses]
                    except Exception as e:
                        print(f"batch_chat failed: {e}")
                        raise e
                
                # Process responses
                for (idx, item), response in zip(items, responses):
                    if response:
                        judge_response = response
                        # extracted_option = self.extract_option(judge_response.lower())
                        extracted_option = self.extract_option_from_general_verifier(judge_response)
                        if extracted_option:
                            extracted_option = extracted_option.strip().upper()
                        
                        # Determine if answer is correct based on prompt type
                        if any(dataset_name in data_source for dataset_name in ["Math-500", "AMC2023", "OlympiadBench", "Minerva", "TheoremQA", "AIME2024", "AIME2025"]):
                            # Math prompt - Y means correct
                            correct = extracted_option == 'Y'
                        # elif 'gpqa_diamond' in data_source or 'MMLUPro' in data_source or 'SuperGPQA' in data_source:
                        #     # Multiple choice prompt - extracted option should match ground truth
                        #     correct = extracted_option and extracted_option.lower() == str(item['ground_truth']).lower()
                        # elif 'WebInstruct-verified' in data_source:
                        #     # General prompt - Y means correct
                        #     correct = extracted_option == 'Y'
                        elif 'gpqa_diamond' in data_source or 'MMLUPro' in data_source or 'SuperGPQA' in data_source or 'WebInstruct-verified' in data_source:
                            # General verifier prompt - Yes means correct
                            correct = extracted_option == 'YES'
                        else:
                            correct = False
                        results[idx] = 1.0 if correct else 0.0
                    else:
                        results[idx] = 0.0
                        
            except Exception as e:
                print(f"Error processing data_source {data_source}: {e}")
                # Fallback to 0.0 for all items in this group
                for idx, _ in items:
                    results[idx] = 0.0
        
        return results

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        if self.use_batch_api and self.client and self.used_model != 'no_api':
            # Collect all data for batch processing
            batch_data = []
            response_lengths = []
            
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_lengths.append(valid_response_length)

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                prompt_str_raw = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns

                batch_data.append({
                    'data_source': data_source,
                    'solution_str': response_str,
                    'ground_truth': ground_truth,
                    'extra_info': extra_info,
                    'prompt_str': prompt_str_raw,
                    'prompt_str_clean': prompt_str,
                    'response_str': response_str,
                    'index': i
                })

            # Batch compute scores
            scores = self.batch_compute_scores(batch_data)
            
            # Assign scores to tensor and print debug info
            for i, (item, score) in enumerate(zip(batch_data, scores)):
                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[i, response_lengths[i] - 1] = reward

                # Debug printing
                data_source = item['data_source']
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", item['prompt_str_clean'])
                    print("[response]", item['response_str'])
                    print("[ground_truth]", item['ground_truth'])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
        else:
            # Fallback to original sequential processing
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                prompt_str_raw = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    prompt_str=prompt_str_raw,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
