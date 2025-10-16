MODEL_PATH=/path/to/ORZ-7B-LaSeR
MODEL_NAME=ORZ-7B-LaSeR

# reasoning performance
# math500
python3 src/eval_math.py \
    --input_file ./data/math500/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/math500/$MODEL_NAME.jsonl \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 512 \
    --n 2 


# amc23
python3 src/eval_math.py \
    --input_file ./data/amc23/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/amc23/$MODEL_NAME.jsonl \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 512 \
    --n 32 


# aime24
python3 src/eval_math.py \
    --input_file ./data/aime24/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/aime24/$MODEL_NAME.jsonl \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 512 \
    --n 32 



# aime25
python3 src/eval_math.py \
    --input_file ./data/aime25/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/aime25/$MODEL_NAME.jsonl \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 512 \
    --n 32 



# OlympiadBench
python3 src/eval_math.py \
    --input_file ./data/OlympiadBench/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/OlympiadBench/$MODEL_NAME.jsonl \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 512 \
    --n 2 



# self-rewarding performance
# self_rewarding_token_id:  151652 for Qwen, 128002 for OctoThinker
# self_rewarding_ref_constant: -23.0 for Qwen, -25.0 for OctoThinker
# math500
python3 src/calculate_self_rewarding_scores.py \
    --input_file ./self-rewarding-outputs/math500/$MODEL_NAME.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/math500/${MODEL_NAME}_self_rewarding.jsonl \
    --max_tokens 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_num_seqs 128 \
    --n 1 \
    --self_rewarding_token_id 151652 \
    --self_rewarding_ratio_coef 0.1 \
    --self_rewarding_ref_constant -23.0

# amc23
python3 src/calculate_self_rewarding_scores.py \
    --input_file ./self-rewarding-outputs/amc23/$MODEL_NAME.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/amc23/${MODEL_NAME}_self_rewarding.jsonl \
    --max_tokens 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_num_seqs 128 \
    --n 1 \
    --self_rewarding_token_id 151652 \
    --self_rewarding_ratio_coef 0.1 \
    --self_rewarding_ref_constant -23.0


# aime24
python3 src/calculate_self_rewarding_scores.py \
    --input_file ./self-rewarding-outputs/aime24/$MODEL_NAME.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/aime24/${MODEL_NAME}_self_rewarding.jsonl \
    --max_tokens 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_num_seqs 128 \
    --n 1 \
    --self_rewarding_token_id 151652 \
    --self_rewarding_ratio_coef 0.1 \
    --self_rewarding_ref_constant -23.0



# aime25
python3 src/calculate_self_rewarding_scores.py \
    --input_file ./self-rewarding-outputs/aime25/$MODEL_NAME.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/aime25/${MODEL_NAME}_self_rewarding.jsonl \
    --max_tokens 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_num_seqs 128 \
    --n 1 \
    --self_rewarding_token_id 151652 \
    --self_rewarding_ratio_coef 0.1 \
    --self_rewarding_ref_constant -23.0



# OlympiadBench
python3 src/calculate_self_rewarding_scores.py \
    --input_file ./self-rewarding-outputs/OlympiadBench/$MODEL_NAME.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./self-rewarding-outputs/OlympiadBench/${MODEL_NAME}_self_rewarding.jsonl \
    --max_tokens 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_num_seqs 128 \
    --n 1 \
    --self_rewarding_token_id 151652 \
    --self_rewarding_ratio_coef 0.1 \
    --self_rewarding_ref_constant -23.0

