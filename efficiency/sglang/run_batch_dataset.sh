dir=$(pwd)

DEVICES=(3)
TP_RANKS=(1)
MEM_FRAC=(0.85)

TARGET_MODEL_PATHS=("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
SPEC_MODEL_PATHS=("yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B")

SPEC_LEN_STRIDE=(4 5)
# SPEC_TOTAL_TOKENS=(4)

SPEC_TOPK=(1)

NUM_REQUEST=(512)
DATA_SET=("aime" "math500"  "gpqa-diamond")

for device in "${DEVICES[@]}"; do
  for rank in "${TP_RANKS[@]}"; do
    for mem in "${MEM_FRAC[@]}"; do
      for tgt_model in "${TARGET_MODEL_PATHS[@]}"; do
        for num_req in "${NUM_REQUEST[@]}"; do
            for dataset in "${DATA_SET[@]}"; do
              echo "Running with: $device $rank $mem $tgt_model $num_req $dataset"
              bash "$dir/run_base_dataset_single.sh" "$device" "$rank" "$mem" "$tgt_model" "$num_req" "$dataset"
            done
        done
      done
    done
  done
done


for device in "${DEVICES[@]}"; do
  for rank in "${TP_RANKS[@]}"; do
    for mem in "${MEM_FRAC[@]}"; do
      for tgt_model in "${TARGET_MODEL_PATHS[@]}"; do
        for spec_model in "${SPEC_MODEL_PATHS[@]}"; do
          for stride in "${SPEC_LEN_STRIDE[@]}"; do
              for topk in "${SPEC_TOPK[@]}"; do
                for num_req in "${NUM_REQUEST[@]}"; do
                  for dataset in "${DATA_SET[@]}"; do
                    echo "Running with: $device $rank $mem $tgt_model $spec_model $stride $stride $topk $num_req $dataset"
                    bash "$dir/run_eagle_dataset_single.sh" "$device" "$rank" "$mem" "$tgt_model" "$spec_model" "$stride" "$stride" "$topk" "$num_req" "$dataset"
                  done
                done
              done
          done
        done
      done
    done
  done
done
