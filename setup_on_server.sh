#!/bin/bash
# =============================================================
# Cloud Server Setup Script for Soccer-Twos RL Training
# Tested on: Linux + NVIDIA A40/A100 + CUDA 11.x/12.x
# =============================================================
set -e

echo "=== Step 1: Load CUDA module ==="
module load cuda/12.6 2>/dev/null || echo "CUDA module not found, assuming CUDA is already in PATH"

echo "=== Step 2: Create conda environment ==="

# 1. 使用 -p 来指定绝对路径创建环境（去掉了掩耳盗铃的 2>/dev/null 以便看到真实报错）
conda create -p /storage/ice1/1/4/zguo407/soccertwos python=3.8 -y

# 2. 在 Bash 脚本内部激活 conda 必须先运行 hook
eval "$(conda shell.bash hook)"

# 3. 激活这个特定路径的环境
conda activate /storage/ice1/1/4/zguo407/soccertwos

echo "=== Step 3: Install pip/setuptools/wheel (pinned versions for compatibility) ==="
python -m pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

echo "=== Step 4: Install PyTorch 1.8.1 + CUDA 11.1 (supports A40/A100 sm_86) ==="
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "=== Step 5: Install project dependencies ==="
pip install -r requirements.txt

echo "=== Step 6: Patch Ray 1.4.0 torch_policy.py (fix GPU detection bug) ==="
TORCH_POLICY_FILE=$(python -c "import ray.rllib.policy.torch_policy as m; print(m.__file__)")
if grep -q "if not gpu_ids:" "$TORCH_POLICY_FILE"; then
    echo "Patch already applied, skipping."
else
    echo "Applying patch to $TORCH_POLICY_FILE ..."
    python -c "
import re

filepath = '$TORCH_POLICY_FILE'
with open(filepath, 'r') as f:
    content = f.read()

old_block = '''            gpu_ids = ray.get_gpu_ids()
            self.devices = [
                torch.device(\"cuda:{}\".format(i))
                for i, id_ in enumerate(gpu_ids) if i < config[\"num_gpus\"]
            ]
            self.device = self.devices[0]'''

new_block = '''            gpu_ids = ray.get_gpu_ids()
            if not gpu_ids:
                # Fallback to CPU if no GPUs were allocated to this worker.
                logger.info(
                    \"TorchPolicy (worker={}) no GPU IDs allocated by Ray, \"
                    \"falling back to CPU.\".format(
                        worker_idx if worker_idx > 0 else \"local\"))
                self.device = torch.device(\"cpu\")
                self.devices = [self.device]
                self.model_gpu_towers = [model]
                self.model = model
            else:
                self.devices = [
                    torch.device(\"cuda:{}\".format(i))
                    for i, id_ in enumerate(gpu_ids)
                    if i < config[\"num_gpus\"]
                ]
                self.device = self.devices[0]'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(filepath, 'w') as f:
        f.write(content)
    print('Patch applied successfully!')
else:
    print('Could not find exact block to patch. Check torch_policy.py manually.')
"
fi

echo "=== Step 7: Verify GPU ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== Setup Complete! ==="
echo "To start training, run:"
echo "  conda activate soccertwos"
echo "  export MLAGENTS_FORCE_NO_GRAPHICS=1"
echo "  export RAY_DISABLE_MEMORY_MONITOR=1"
echo "  python example_ray_ppo_sp_still.py"
