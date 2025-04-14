dir=$(pwd)

# if setup.sh in not in dir, return
if [ ! -f "$dir/setup.sh" ]; then
    echo "Please run this script from /efficiency/vllm"
    exit 1
fi

cd $dir/../../3rdparty/vllm

git checkout v0.8.3
git submodule update --init --recursive

# Better if using a new conda env
# conda create -n vllm-ngram python=3.12 -y
# conda activate vllm-ngram

# Install vllm
git apply ../vllm.patch
ln -s $dir/../loop_detector.py ./vllm/v1/core/sched/loop_detector.py

VLLM_USE_PRECOMPILED=1 pip install --editable .
