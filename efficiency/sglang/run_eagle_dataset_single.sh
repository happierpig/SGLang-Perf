# Parameter tuned by:
# https://github.com/sgl-project/sglang/pull/4247
# API call follows:
# https://docs.sglang.ai/backend/speculative_decoding.html

dir=$(pwd)

# check whether the number of argument is 12
# device, rank, and memory fraction, tgt_model_path, spec_model_path, spec_len_stride,
# spec_total_tokens, spec_topk, acc_len, num_request, len_input, len_output

if [ $# -ne 10 ]; then
    echo "Usage: $0 <device> <rank> <mem_frac> <tgt_model_path> <spec_model_path> <spec_len_stride> \
    <spec_total_tokens> <spec_topk> <num_request> <dataset_name>"
    exit 1
fi


DEVICES=$1
TP_RANKS=$2
MEM_FRAC=$3

target_model_path=$4
spec_model_path=$5

# spec args
SPEC_LEN_STRIDE=$6
SPEC_TOTAL_TOKENS=$7
SPEC_TOPK=$8

NUM_REQUEST=$9
DATASET_NAME=${10}


MODEL_NAME=$(basename $target_model_path)

# log path
OUTPUT_DIR=$dir/output/$MODEL_NAME/$DATASET_NAME
mkdir -p $OUTPUT_DIR

LOG_NAME=${MODEL_NAME}_TP-${TP_RANKS}_STRIDE-${SPEC_LEN_STRIDE}_TREETOKENS-${SPEC_TOTAL_TOKENS}_TOPK-${SPEC_TOPK}_APP-${DATASET_NAME}
server_log=$OUTPUT_DIR/${LOG_NAME}_server.log
client_log=$OUTPUT_DIR/${LOG_NAME}_client.log

(
    # we do not use flashinfer attn backend due to the huge CUDA Graph memory consumption
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python3 -m sglang.launch_server --model $target_model_path --speculative-draft $spec_model_path \
    --speculative-algo EAGLE3 --speculative-num-steps $SPEC_LEN_STRIDE \
    --speculative-eagle-topk $SPEC_TOPK --speculative-num-draft-tokens $SPEC_TOTAL_TOKENS \
    --dtype bfloat16 --port 30000 --tp $TP_RANKS --disable-radix --mem-frac $MEM_FRAC --attention-backend triton &> $server_log
) &

sleep 100

(
    python3 bench_eagle.py --backend sglang --dataset-name $DATASET_NAME --model ${target_model_path} --disable-ignore-eos \
    --num-prompts ${NUM_REQUEST} --output-file ${OUTPUT_DIR}/${LOG_NAME}.jsonl &> $client_log
)

# echo all args into client log
echo "--- Arguments ---" >> $client_log
echo "Device: $DEVICES" >> $client_log
echo "Rank: $TP_RANKS" >> $client_log
echo "Memory fraction: $MEM_FRAC" >> $client_log
echo "Target model path: $target_model_path" >> $client_log
echo "Spec model path: $spec_model_path" >> $client_log
echo "Spec len stride: $SPEC_LEN_STRIDE" >> $client_log
echo "Spec total tokens: $SPEC_TOTAL_TOKENS" >> $client_log
echo "Spec topk: $SPEC_TOPK" >> $client_log
echo "Spec simulated acceptance length: $SPEC_SIMULATED_ACC_LEN" >> $client_log
echo "Num request: $NUM_REQUEST" >> $client_log
echo "Len input: $LEN_INPUT" >> $client_log
echo "Len output: $LEN_OUTPUT" >> $client_log
echo "Server log: $server_log" >> $client_log
echo "Client log: $client_log" >> $client_log

pkill -f sglang::scheduler_TP0
pkill -f sglang.launch_server
