vllm serve /data/lsb/deep_past/output/checkpoint-2400-merged \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size 1 \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --generation-config vllm


conda deactivate && source /data/lsb/ms-swift/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1,2,3,4 python /data/lsb/deep_past/script/train.py


conda deactivate && source /data/lsb/ms-swift/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python /data/lsb/deep_past/script/train.py --skip-initial-eval


conda deactivate && source /data/lsb/ms-swift/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com 
CUDA_VISIBLE_DEVICES=1 python /data/lsb/deep_past/script/train.py --skip-initial-eval


conda deactivate && source /data/lsb/ms-swift/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 python /data/lsb/deep_past/script/train.py  --skip-initial-eval


conda deactivate && source /data/lsb/ms-swift/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=3 python /data/lsb/deep_past/script/train.py --skip-initial-eval

