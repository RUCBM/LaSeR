import argparse
import os
import torch
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from math_verify import parse, verify
import copy



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
    
def apply_chat_template(toker, messages, chat_template=None):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])
    return toker(input_prompt.strip(), add_special_tokens=False).input_ids


def apply_chat_template_full_text(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    input_prompt = input_prompt.strip()
    if "<|im_end|>" == input_prompt[-10:]: # fix template issue for Qwen3-4B-Base
        input_prompt = input_prompt[:-10]
    if "<think>\n\n</think>\n\n" in input_prompt: # fix template issue for Qwen3-4B-Base
        input_prompt = input_prompt.replace("<think>\n\n</think>\n\n", "")
    # print(input_prompt)
    return toker(input_prompt, add_special_tokens=False).input_ids

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--self_rewarding_token_id", type=int, required=True)
    parser.add_argument("--self_rewarding_ratio_coef", type=float, default=0.0, required=True)
    parser.add_argument("--self_rewarding_ref_constant", type=float, default=1.0)
    parser.add_argument("--self_rewarding_ref_model_path", type=str, default=None)
    args = parser.parse_args()

    toker = AutoTokenizer.from_pretrained(args.model_path)
    args.model_name = os.path.basename(args.model_path)

    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=torch.cuda.device_count(),
        # enable_prefix_caching=True, swap_space=16,
        # max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, n=args.n, seed=42)


    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]
    if args.begin_idx >= 0 and args.end_idx >= 0:
        input_data = input_data[args.begin_idx: args.end_idx]


    res_data = copy.deepcopy(input_data)
    
    specified_token_id = args.self_rewarding_token_id
    eos_token_id = toker.eos_token_id
    specified_token = toker.decode(specified_token_id)
    prompts = []
    solutions = []
    data_items_idx = []
    print(f"Specified token: {specified_token}")
    if args.self_rewarding_ref_constant < 0.0:
        for item_idx, item in enumerate(res_data):
            problem = item["problem"]
            if 'Qwen3-4B' in args.model_path:
                problem = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            for response_idx, response in enumerate(item["responses"]):
                if "extracted_options_results" in item:
                    extracted_pred_answer = item["extracted_options_results"][response_idx]
                else:
                    extracted_pred_answer = item["pred_answers"][response_idx]
                if extracted_pred_answer is not None:
                    prompts.append(problem)
                    solutions.append(response + toker.eos_token + specified_token)
                    data_items_idx.append([item_idx, response_idx])
        
        # Initialize self_rewarding_scores for all items
        for i in range(len(res_data)):
            self_rewarding_scores = [0.0 for _ in range(len(res_data[i]["responses"]))]
            res_data[i]["self_rewarding_scores"] = self_rewarding_scores
        
        # Process prompts in batches to avoid OOM
        batch_size = args.max_num_seqs
        total_prompts = len(prompts)
        print(f"Processing {total_prompts} prompts in batches of {batch_size}")
        
        sampling_params = SamplingParams(temperature=0, top_p=1.0,
                                max_tokens=1, n=1, seed=42, prompt_logprobs=0, logprobs=0, ignore_eos=True)  
        
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size}: prompts {batch_start} to {batch_end-1}")
            
            # Prepare batch data
            batch_prompts = prompts[batch_start:batch_end]
            batch_solutions = solutions[batch_start:batch_end]
            batch_data_items_idx = data_items_idx[batch_start:batch_end]
            
            # Convert to token ids
            batch_self_rewarding_prompt_token_ids = [
                apply_chat_template_full_text(toker, [{"role": "user", "content": prompt}, {"role": "assistant", "content": solution}])
                for prompt, solution in zip(batch_prompts, batch_solutions)
            ]
            
            # Generate for this batch
            batch_self_rewarding_generations = llm.generate(
                prompt_token_ids=batch_self_rewarding_prompt_token_ids, 
                sampling_params=sampling_params
            )
            
            # Process results for this batch
            for i, generation_output in enumerate(batch_self_rewarding_generations):
                item_idx, response_idx = batch_data_items_idx[i]
                
                # Get the eos token's log prob in order to predict the log prob of the next specified token
                prompt_logprobs = getattr(generation_output, 'prompt_logprobs', None)
                # Get the logprob of the last token in the prompt (i.e., the logprob assigned to the last prompt token)
                specified_token_logprob = prompt_logprobs[-1][specified_token_id].logprob
                self_rewarding_score = (specified_token_logprob - args.self_rewarding_ref_constant) * args.self_rewarding_ratio_coef
                res_data[item_idx]["self_rewarding_scores"][response_idx] = self_rewarding_score
    else:
        assert args.self_rewarding_ref_model_path is not None
        # first forward policy model
        for item_idx, item in enumerate(res_data):
            problem = item["problem"]
            for response_idx, response in enumerate(item["responses"]):
                if item["pred_answers"][response_idx] is not None:
                    prompts.append(problem)
                    solutions.append(response + toker.eos_token + specified_token)
                    data_items_idx.append([item_idx, response_idx])
        
        # Initialize self_rewarding_scores for all items
        for i in range(len(res_data)):
            self_rewarding_scores = [0.0 for _ in range(len(res_data[i]["responses"]))]
            res_data[i]["self_rewarding_scores"] = self_rewarding_scores
        
        for i in range(len(res_data)):
            res_data[i]["policy_model_logprobs"] = copy.deepcopy(res_data[i]["self_rewarding_scores"])
            res_data[i]["ref_model_logprobs"] = copy.deepcopy(res_data[i]["self_rewarding_scores"])
        
        # Process prompts in batches to avoid OOM
        batch_size = args.max_num_seqs
        total_prompts = len(prompts)
        print(f"Processing {total_prompts} prompts in batches of {batch_size}")
        
        sampling_params = SamplingParams(temperature=0, top_p=1.0,
                                max_tokens=1, n=1, seed=42, prompt_logprobs=0, logprobs=0, ignore_eos=True)  
        
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size}: prompts {batch_start} to {batch_end-1}")
            
            # Prepare batch data
            batch_prompts = prompts[batch_start:batch_end]
            batch_solutions = solutions[batch_start:batch_end]
            batch_data_items_idx = data_items_idx[batch_start:batch_end]
            
            # Convert to token ids
            batch_self_rewarding_prompt_token_ids = [
                apply_chat_template_full_text(toker, [{"role": "user", "content": prompt}, {"role": "assistant", "content": solution}])
                for prompt, solution in zip(batch_prompts, batch_solutions)
            ]
            
            # Generate for this batch
            batch_self_rewarding_generations = llm.generate(
                prompt_token_ids=batch_self_rewarding_prompt_token_ids, 
                sampling_params=sampling_params
            )
            
            # Process results for this batch
            for i, generation_output in enumerate(batch_self_rewarding_generations):
                item_idx, response_idx = batch_data_items_idx[i]
                
                # Get the eos token's log prob in order to predict the log prob of the next specified token
                prompt_logprobs = getattr(generation_output, 'prompt_logprobs', None)
                # Get the logprob of the last token in the prompt (i.e., the logprob assigned to the last prompt token)
                specified_token_logprob = prompt_logprobs[-2][specified_token_id].logprob
                # self_rewarding_score = (specified_token_logprob - args.self_rewarding_ref_constant) * args.self_rewarding_ratio_coef
                res_data[item_idx]["policy_model_logprobs"][response_idx] = specified_token_logprob
            
        del llm

        # then forward ref model
        ref_llm = LLM(
            model=args.self_rewarding_ref_model_path, tokenizer=args.self_rewarding_ref_model_path,
            gpu_memory_utilization=0.6,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        
        
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size}: prompts {batch_start} to {batch_end-1}")
            
            # Prepare batch data
            batch_prompts = prompts[batch_start:batch_end]
            batch_solutions = solutions[batch_start:batch_end]
            batch_data_items_idx = data_items_idx[batch_start:batch_end]
            
            # Convert to token ids
            batch_self_rewarding_prompt_token_ids = [
                apply_chat_template_full_text(toker, [{"role": "user", "content": prompt}, {"role": "assistant", "content": solution}])
                for prompt, solution in zip(batch_prompts, batch_solutions)
            ]
            
            # Generate for this batch
            batch_self_rewarding_generations = ref_llm.generate(
                prompt_token_ids=batch_self_rewarding_prompt_token_ids, 
                sampling_params=sampling_params
            )
            
            # Process results for this batch
            for i, generation_output in enumerate(batch_self_rewarding_generations):
                item_idx, response_idx = batch_data_items_idx[i]
                
                # Get the eos token's log prob in order to predict the log prob of the next specified token
                prompt_logprobs = getattr(generation_output, 'prompt_logprobs', None)
                # Get the logprob of the last token in the prompt (i.e., the logprob assigned to the last prompt token)
                specified_token_logprob = prompt_logprobs[-2][specified_token_id].logprob
                # self_rewarding_score = (specified_token_logprob - args.self_rewarding_ref_constant) * args.self_rewarding_ratio_coef
                res_data[item_idx]["ref_model_logprobs"][response_idx] = specified_token_logprob
        for i in range(len(res_data)):
            for j in range(len(res_data[i]["responses"])):
                res_data[i]["self_rewarding_scores"][j] = (res_data[i]["policy_model_logprobs"][j] - res_data[i]["ref_model_logprobs"][j]) * args.self_rewarding_ratio_coef
        
    # Calculate metrics for all items (including those with pred_answers is None)
    y_true_all = []
    y_pred_all = []
    correct_solution_self_rewarding_correct_count_all = 0
    correct_solution_total_all = 0
    incorrect_solution_self_rewarding_correct_count_all = 0
    incorrect_solution_total_all = 0

    # Calculate metrics only for items where pred_answers is not None
    y_true = []
    y_pred = []
    correct_solution_self_rewarding_correct_count = 0
    correct_solution_total = 0
    incorrect_solution_self_rewarding_correct_count = 0
    incorrect_solution_total = 0

    for item in res_data:
        if "verify_accs_results" in item: # for general reasoning tasks
            acc_list = item.get("verify_accs_results", [])
        else:
            acc_list = item.get("acc_list", [])
        self_rewarding_scores = item.get("self_rewarding_scores", [])
        pred_answers = item.get("pred_answers", [])
        for acc, score, p_ans in zip(acc_list, self_rewarding_scores, pred_answers):
            # For all items
            if acc:
                correct_solution_total_all += 1
                if score > 0.5:
                    correct_solution_self_rewarding_correct_count_all += 1
            else:
                incorrect_solution_total_all += 1
                if score <= 0.5:
                    incorrect_solution_self_rewarding_correct_count_all += 1
            y_true_all.append(int(acc))
            y_pred_all.append(int(score > 0.5))

            # For items where pred_answers is not None
            if p_ans is not None:
                if acc:
                    correct_solution_total += 1
                    if score > 0.5:
                        correct_solution_self_rewarding_correct_count += 1
                else:
                    incorrect_solution_total += 1
                    if score <= 0.5:
                        incorrect_solution_self_rewarding_correct_count += 1
                y_true.append(int(acc))
                y_pred.append(int(score > 0.5))

    # Metrics for filtered items (pred_answers is not None)
    true_acc = correct_solution_self_rewarding_correct_count / correct_solution_total if correct_solution_total > 0 else 0.0
    false_acc = incorrect_solution_self_rewarding_correct_count / incorrect_solution_total if incorrect_solution_total > 0 else 0.0
    overall_acc = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true) if y_true else 0.0
    f1 = 2 * true_acc * false_acc / (true_acc + false_acc) if true_acc + false_acc > 0 else 0.0

    print(f"Self-rewarding score accuracy for acc_list==True (filtered): {true_acc:.4f} ({true_acc*100:.2f}%)")
    print(f"Self-rewarding score accuracy for acc_list==False (filtered): {false_acc:.4f} ({false_acc*100:.2f}%)")
    print(f"Self-rewarding score overall accuracy (filtered): {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Self-rewarding score F1 (filtered): {f1:.4f}")

    # Metrics for all items (including those with pred_answers is None)
    true_acc_all = correct_solution_self_rewarding_correct_count_all / correct_solution_total_all if correct_solution_total_all > 0 else 0.0
    false_acc_all = incorrect_solution_self_rewarding_correct_count_all / incorrect_solution_total_all if incorrect_solution_total_all > 0 else 0.0
    overall_acc_all = sum([yt == yp for yt, yp in zip(y_true_all, y_pred_all)]) / len(y_true_all) if y_true_all else 0.0
    f1_all = 2 * true_acc_all * false_acc_all / (true_acc_all + false_acc_all) if true_acc_all + false_acc_all > 0 else 0.0

    print(f"Self-rewarding score accuracy for acc_list==True (all): {true_acc_all:.4f} ({true_acc_all*100:.2f}%)")
    print(f"Self-rewarding score accuracy for acc_list==False (all): {false_acc_all:.4f} ({false_acc_all*100:.2f}%)")
    print(f"Self-rewarding score overall accuracy (all): {overall_acc_all:.4f} ({overall_acc_all*100:.2f}%)")
    print(f"Self-rewarding score F1 (all): {f1_all:.4f}")
    # save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d) + "\n")


if __name__ == '__main__':
    main()
