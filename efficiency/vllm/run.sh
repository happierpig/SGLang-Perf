dir=$(pwd)
output_dir=$dir/results
mkdir -p $output_dir

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name="agentica-org/DeepCoder-14B-Preview"

tp_rank=1
num_requests=16



num_speculateve_tokens=3

(
    port_num=30001
    CUDA_VISIBLE_DEVICES=4 python3 run_ngram_single.py \
        --model-name $model_name \
        --num-requests $num_requests \
        --num-speculative-tokens $num_speculateve_tokens \
        --base-port $port_num \
        --output-dir $output_dir \
        --tp-rank $tp_rank \
        --enable-ngram
)&


# (
#     port_num=30002
#     CUDA_VISIBLE_DEVICES=5 python3 run_ngram_single.py \
#         --model-name $model_name \
#         --num-requests $num_requests \
#         --base-port $port_num \
#         --output-dir $output_dir \
#         --tp-rank $tp_rank
# )&

# num_speculateve_tokens=4

# (
#     port_num=30003
#     CUDA_VISIBLE_DEVICES=6 python3 run_ngram_single.py \
#         --model-name $model_name \
#         --num-requests $num_requests \
#         --num-speculative-tokens $num_speculateve_tokens \
#         --base-port $port_num \
#         --output-dir $output_dir \
#         --tp-rank $tp_rank \
#         --enable-ngram
# )&


# num_speculateve_tokens=5

# (
#     port_num=30004
#     CUDA_VISIBLE_DEVICES=7 python3 run_ngram_single.py \
#         --model-name $model_name \
#         --num-requests $num_requests \
#         --num-speculative-tokens $num_speculateve_tokens \
#         --base-port $port_num \
#         --output-dir $output_dir \
#         --tp-rank $tp_rank \
#         --enable-ngram
# )&

wait
