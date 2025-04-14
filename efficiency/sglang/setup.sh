dir=$(pwd)

# conda creat -n sglang-eagle python=3.12
# conda activate sglang-eagle

git submodule update --init --recursive

# if setup.sh in not in dir, return
if [ ! -f "$dir/setup.sh" ]; then
    echo "Please run this script from /efficiency/sglang"
    exit 1
fi

cd $dir/../../3rdparty/sglang

git apply ../sglang.patch
ln -s $dir/../loop_detector.py ./python/sglang/srt/managers/loop_detector.py

pip install numba
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
