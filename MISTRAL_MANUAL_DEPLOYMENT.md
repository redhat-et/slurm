# Manual Mistral 7B Deployment Guide (No Ansible)

This guide provides step-by-step instructions to manually deploy Mistral 7B Instruct on a Slurm GPU cluster without using Ansible.

## Prerequisites

- Slurm cluster with GPU nodes (already configured and running)
- SSH access to all nodes (controller + compute nodes)
- Root/sudo access on all nodes
- NVIDIA drivers and CUDA installed
- Internet connectivity on all nodes

**Cluster Information:**
- Controller node: `slurm-controller` (or IP address)
- Compute nodes: `slurm-node-1`, `slurm-node-2` (or IP addresses)
- GPU: NVIDIA L4 (24GB VRAM)

---

## Step 1: Install vLLM on All Nodes

You need to install vLLM on the controller and all compute nodes.

### On Each Node (Controller + Compute Nodes)

```bash
# SSH to the node
ssh ec2-user@<node-ip>

# Become root
sudo -i

# Install Python development packages
dnf install -y python3 python3-pip python3-devel

# Create directories
mkdir -p /opt/vllm
mkdir -p /opt/models
mkdir -p /var/log/vllm

# Set ownership to slurm user
chown -R slurm:slurm /opt/vllm
chown -R slurm:slurm /opt/models
chown -R slurm:slurm /var/log/vllm

# Install vLLM and dependencies system-wide
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

pip3 install vllm huggingface_hub accelerate transformers

# Configure HuggingFace cache for slurm user
echo 'export HF_HOME=/opt/models' >> /home/slurm/.bashrc
chown slurm:slurm /home/slurm/.bashrc

# Verify installation
pip3 show vllm
```

**Expected output:**
```
Name: vllm
Version: 0.15.1
```

**Repeat this step on all nodes:**
1. SSH to slurm-controller and run the above commands
2. SSH to slurm-node-1 and run the above commands
3. SSH to slurm-node-2 and run the above commands

---

## Step 2: Download Mistral Model on Compute Nodes

The model needs to be cached on the compute nodes where inference will run.

### On Each Compute Node (slurm-node-1, slurm-node-2)

```bash
# SSH to compute node
ssh ec2-user@slurm-node-1

# Switch to root
sudo -i

# Create download script
cat > /tmp/download_mistral.py << 'EOF'
#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Set cache directory
os.environ['HF_HOME'] = '/opt/models'

model_id = 'mistralai/Mistral-7B-Instruct-v0.3'

print(f"Downloading {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        cache_dir='/opt/models',
        resume_download=True,
        max_workers=4
    )
    print(f"Successfully downloaded {model_id}")
except Exception as e:
    print(f"Error downloading {model_id}: {e}")
EOF

# Make executable
chmod +x /tmp/download_mistral.py

# Run download (this may take 10-15 minutes)
python3 /tmp/download_mistral.py

# Verify download
ls -lh /opt/models/models--mistralai--Mistral-7B-Instruct-v0.3/

# Set ownership
chown -R slurm:slurm /opt/models
```

**Expected output:**
```
Downloading mistralai/Mistral-7B-Instruct-v0.3...
Fetching 11 files: 100%|████████████| 11/11 [02:15<00:00, 12.34s/it]
Successfully downloaded mistralai/Mistral-7B-Instruct-v0.3
```

**Verify model files:**
```bash
find /opt/models/models--mistralai--Mistral-7B-Instruct-v0.3 -type f | head -10
```

You should see safetensors files, config.json, tokenizer files, etc.

**Repeat on all compute nodes:**
- Run on slurm-node-1
- Run on slurm-node-2

---

## Step 3: Create Job Scripts on Controller

Now create the Slurm batch script to run Mistral.

### On Controller Node

```bash
# SSH to controller
ssh ec2-user@slurm-controller

# Switch to root
sudo -i

# Create jobs directory
mkdir -p /home/slurm/vllm-jobs
mkdir -p /home/slurm/vllm-jobs/outputs

# Create Mistral serving script
cat > /home/slurm/vllm-jobs/serve_mistral.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=vllm-serve-mistral
#SBATCH --output=/home/slurm/vllm-jobs/outputs/serve_mistral_%j.out
#SBATCH --error=/home/slurm/vllm-jobs/outputs/serve_mistral_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

echo "=========================================="
echo "vLLM Model Serving: MISTRAL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: mistralai/Mistral-7B-Instruct-v0.3"
echo "Start time: $(date)"
echo ""

# Environment setup
export HF_HOME=/opt/models
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Display GPU info
echo "GPU Information:"
nvidia-smi -L
echo ""

# Display model cache
echo "Model cache directory:"
ls -lh /opt/models/models--mistralai--Mistral-7B-Instruct-v0.3/ 2>/dev/null || echo "Model not cached"
echo ""

echo "Starting vLLM server..."
echo "API will be available at: http://$SLURMD_NODENAME:8001"
echo ""

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests

echo ""
echo "vLLM server stopped at $(date)"
EOF

# Make executable
chmod +x /home/slurm/vllm-jobs/serve_mistral.sh

# Set ownership
chown -R slurm:slurm /home/slurm/vllm-jobs

# Verify script created
ls -lh /home/slurm/vllm-jobs/
```

**Expected output:**
```
-rwxr-xr-x. 1 slurm slurm 1.2K Feb  5 18:00 serve_mistral.sh
```

---

## Step 4: Submit Job to Slurm

### On Controller Node

```bash
# Still on controller as root
# Submit job as slurm user
sudo -u slurm bash -c 'cd /home/slurm/vllm-jobs && sbatch serve_mistral.sh'
```

**Expected output:**
```
Submitted batch job 8
```

**Note the job ID** (e.g., 8) - you'll need it for monitoring.

---

## Step 5: Monitor Job Status

### Check Queue Status

```bash
# View job queue
squeue

# More detailed view
squeue -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

**Expected output:**
```
JOBID PARTITION                     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
    8       gpu   vllm-serve-mistral    slurm  RUNNING      0:45  12:00:00      1 slurm-node-1
```

**Job States:**
- `PD` (PENDING) - Waiting for resources
- `R` (RUNNING) - Job is running
- `CG` (COMPLETING) - Job is finishing
- `CD` (COMPLETED) - Job finished successfully
- `F` (FAILED) - Job failed

### Check Job Details

```bash
# Detailed job info
sacct -j 8 --format=JobID,JobName,State,ExitCode,Start,Elapsed,NodeList
```

---

## Step 6: Monitor Model Loading

The model takes about 4-5 minutes to load. Monitor the progress:

### View Real-time Logs

```bash
# Find which node is running the job
NODE=$(squeue -h -j 8 -o "%N")
echo "Job running on: $NODE"

# SSH to that node
ssh $NODE

# Tail the output log (replace 8 with your job ID)
tail -f /home/slurm/vllm-jobs/outputs/serve_mistral_8.out
```

**You should see:**

1. **Initial startup** (first 10 seconds):
```
==========================================
vLLM Model Serving: MISTRAL
==========================================
Job ID: 8
Node: slurm-node-1
Model: mistralai/Mistral-7B-Instruct-v0.3
Start time: Thu Feb  5 19:30:00 UTC 2026

GPU Information:
GPU 0: NVIDIA L4 (UUID: GPU-xxx)
```

2. **vLLM initialization** (~30 seconds):
```
INFO 02-05 19:30:15 [utils.py:325]        █     █     █▄   ▄█
INFO 02-05 19:30:15 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
INFO 02-05 19:30:15 [utils.py:325]   █▄█▀ █     █     █     █  model   mistralai/Mistral-7B-Instruct-v0.3
INFO 02-05 19:30:22 [model.py:541] Resolved architecture: MistralForCausalLM
INFO 02-05 19:30:22 [model.py:1561] Using max model len 8192
```

3. **Model loading** (~2 minutes):
```
INFO 02-05 19:30:30 [gpu_model_runner.py:4033] Starting to load model...
INFO 02-05 19:30:32 [cuda.py:364] Using FLASH_ATTN attention backend
Loading safetensors checkpoint shards: 100%|██████████| 1/1 [01:53<00:00]
INFO 02-05 19:32:23 [default_loader.py:291] Loading weights took 113.27 seconds
INFO 02-05 19:32:23 [gpu_model_runner.py:4130] Model loading took 13.51 GiB memory
```

4. **Compilation** (~2 minutes):
```
INFO 02-05 19:32:28 [backends.py:812] Using cache directory for torch.compile
INFO 02-05 19:32:28 [backends.py:872] Dynamo bytecode transform time: 4.51 s
```

5. **Server ready** (after ~4 minutes total):
```
INFO 02-05 19:34:15 [server.py:234] Uvicorn running on http://0.0.0.0:8001
INFO 02-05 19:34:15 [server.py:235] Application startup complete.
```

**Press Ctrl+C to exit tail** (job keeps running)

### Check for Errors

```bash
# On the compute node, check stderr
cat /home/slurm/vllm-jobs/outputs/serve_mistral_8.err
```

If model loaded successfully, you should see only warnings (not errors).

---

## Step 7: Verify GPU Usage

### On Compute Node Running the Job

```bash
# SSH to the compute node (e.g., slurm-node-1)
ssh slurm-node-1

# Check GPU memory usage
nvidia-smi

# More detailed GPU info
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
```

**Expected output:**
```
name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu
NVIDIA L4, 21094 MiB, 23034 MiB, 0 %, 46
```

**Interpretation:**
- **21,094 MiB used** = Model loaded (~13.5 GB model + ~7.5 GB KV cache)
- **0% utilization** = Idle, waiting for requests
- **46°C** = Normal temperature

### Check vLLM Process

```bash
# Still on compute node
ps aux | grep vllm | grep -v grep
```

**Expected output:**
```
slurm  12345  3.2  3.0 7797860 941176 ?  Sl  19:30  0:07 python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8001 ...
```

### Check Network Port

```bash
# Verify port 8001 is listening
ss -tlnp | grep :8001
```

**Expected output:**
```
LISTEN 0  128  0.0.0.0:8001  0.0.0.0:*  users:(("python3",pid=12345,fd=10))
```

---

## Step 8: Test the API

### Test from Controller Node

```bash
# Exit compute node, return to controller
exit

# You're now on slurm-controller
# Find which node is running the job
NODE=$(squeue -h -j 8 -o "%N")
echo "Testing API on: $NODE"

# Test 1: List available models
curl -s http://$NODE:8001/v1/models | python3 -m json.tool
```

**Expected response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mistralai/Mistral-7B-Instruct-v0.3",
      "object": "model",
      "created": 1770320400,
      "owned_by": "vllm",
      "root": "mistralai/Mistral-7B-Instruct-v0.3",
      "parent": null,
      "max_model_len": 8192,
      "permission": [...]
    }
  ]
}
```

### Test Text Generation

```bash
# Test 2: Generate text completion
curl -s http://$NODE:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Explain GPU computing in one sentence:",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**Expected response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1770320450,
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "choices": [
    {
      "index": 0,
      "text": " GPU computing harnesses the parallel processing power of graphics processing units to accelerate complex computations, enabling faster data analysis and machine learning tasks.",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 45,
    "completion_tokens": 36
  }
}
```

### Test Chat Completions

```bash
# Test 3: Chat completion (OpenAI-compatible)
curl -s http://$NODE:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "What are the benefits of using Slurm for GPU workloads?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### Simple Smoke Test

```bash
# Quick test without json formatting
curl -s http://$NODE:8001/v1/models | grep -o '"id":"[^"]*"'
```

**Expected:**
```
"id":"mistralai/Mistral-7B-Instruct-v0.3"
```

---

## Step 9: Performance Testing

### Measure Inference Speed

```bash
# Time a generation request
time curl -s http://$NODE:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Write a short story about:",
    "max_tokens": 200
  }' > /tmp/test_output.json

# View generated text
cat /tmp/test_output.json | python3 -m json.tool | grep -A 5 '"text"'
```

**Typical performance:**
- First token latency: ~100-200ms
- Tokens per second: ~40-60
- Total time for 200 tokens: ~4-5 seconds

### Stress Test (Multiple Concurrent Requests)

```bash
# Run 5 requests in parallel
for i in {1..5}; do
  curl -s http://$NODE:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mistralai/Mistral-7B-Instruct-v0.3",
      "prompt": "Test prompt number '$i':",
      "max_tokens": 50
    }' &
done

# Wait for all to complete
wait

echo "All requests completed"
```

### Monitor GPU During Inference

```bash
# On compute node, watch GPU in real-time
ssh $NODE
watch -n 1 nvidia-smi
```

During inference you should see:
- GPU Utilization spike to 80-100%
- Memory usage stay constant (~21 GB)
- Temperature increase slightly (up to 60-70°C)

---

## Step 10: Access Logs and Troubleshooting

### View Job Logs

```bash
# On controller
# View standard output
cat /home/slurm/vllm-jobs/outputs/serve_mistral_8.out | less

# View errors
cat /home/slurm/vllm-jobs/outputs/serve_mistral_8.err | less

# Search for specific errors
grep -i error /home/slurm/vllm-jobs/outputs/serve_mistral_8.err
grep -i warning /home/slurm/vllm-jobs/outputs/serve_mistral_8.err
```

### Check Slurm Logs

```bash
# On controller, check slurmctld logs
tail -100 /var/log/slurm/slurmctld.log

# On compute node, check slurmd logs
ssh $NODE
tail -100 /var/log/slurm/slurmd.log
```

### Common Issues

#### Issue: Job stays in PENDING state

```bash
# Check why job is pending
squeue -j 8 -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

Look at the REASON column:
- `Resources` - No available GPU
- `Priority` - Other jobs have higher priority
- `QOSMaxGRESPerUser` - GPU limit reached

**Solution:** Wait for resources or cancel other jobs

#### Issue: Job fails immediately

```bash
# Check exit code
sacct -j 8 --format=JobID,State,ExitCode

# If ExitCode is non-zero, check logs
cat /home/slurm/vllm-jobs/outputs/serve_mistral_8.err
```

#### Issue: API not responding

```bash
# Check if port is actually listening
ssh $NODE "ss -tlnp | grep :8001"

# Check if vLLM process is running
ssh $NODE "ps aux | grep vllm"

# Check firewall
ssh $NODE "iptables -L -n | grep 8001"
```

#### Issue: Out of memory

**Error in logs:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:** Edit job script to reduce memory usage:

```bash
# Edit the script
vi /home/slurm/vllm-jobs/serve_mistral.sh

# Change these parameters:
--gpu-memory-utilization 0.8  # Reduced from 0.9
--max-model-len 4096          # Reduced from 8192
```

Then resubmit the job.

---

## Step 11: Stop the Job

### Cancel Running Job

```bash
# On controller
# Cancel specific job
scancel 8

# Verify it's cancelled
squeue
```

**Expected output:**
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
(empty - no jobs running)
```

### Verify Cleanup

```bash
# Check GPU is freed
ssh $NODE nvidia-smi

# Should show 0 MiB used
```

---

## Step 12: Managing Multiple Jobs

### Run Multiple Models Simultaneously

If you want to run both Phi-3 and Mistral at the same time:

```bash
# On controller
# Submit Mistral (port 8001)
sudo -u slurm bash -c 'cd /home/slurm/vllm-jobs && sbatch serve_mistral.sh'

# Create and submit Phi-3 script (port 8000)
# ... (create serve_phi3.sh with port 8000)
sudo -u slurm bash -c 'cd /home/slurm/vllm-jobs && sbatch serve_phi3.sh'

# View all running jobs
squeue -u slurm
```

**Note:** Each model needs its own GPU, so you need at least 2 GPUs available.

### List All Jobs

```bash
# All jobs
squeue

# Jobs for specific user
squeue -u slurm

# Jobs on specific partition
squeue -p gpu
```

---

## API Reference

### Available Endpoints

Once Mistral is running, these OpenAI-compatible endpoints are available:

**Base URL:** `http://<compute-node>:8001`

#### 1. List Models
```bash
GET /v1/models
```

#### 2. Completions
```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "prompt": "Your prompt here",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "n": 1,
  "stream": false
}
```

#### 3. Chat Completions
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
```

#### 4. Streaming Responses
```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "prompt": "Write a story:",
  "max_tokens": 200,
  "stream": true
}
```

---

## Quick Reference Commands

### Job Management
```bash
# Submit job
sbatch serve_mistral.sh

# View queue
squeue

# Cancel job
scancel <job-id>

# Job details
sacct -j <job-id>

# View logs
cat /home/slurm/vllm-jobs/outputs/serve_mistral_<job-id>.out
```

### Monitoring
```bash
# GPU status
nvidia-smi

# Find node running job
squeue -h -j <job-id> -o "%N"

# Check vLLM process
ps aux | grep vllm

# Check port
ss -tlnp | grep :8001
```

### Testing
```bash
# Quick health check
curl http://<node>:8001/v1/models

# Generate text
curl http://<node>:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3","prompt":"Test:","max_tokens":50}'
```

---

## Next Steps

1. **Create additional model scripts** for Phi-3 or other models
2. **Set up a reverse proxy** (nginx/HAProxy) for load balancing
3. **Implement API key authentication** for security
4. **Add monitoring** with Prometheus/Grafana
5. **Configure log rotation** for `/home/slurm/vllm-jobs/outputs/`
6. **Set up automatic job resubmission** for long-running services

---

## Summary

You've successfully deployed Mistral 7B manually by:

1. ✅ Installing vLLM on all nodes
2. ✅ Downloading the Mistral 7B model
3. ✅ Creating a Slurm batch script
4. ✅ Submitting the job to Slurm
5. ✅ Monitoring model loading and GPU usage
6. ✅ Testing the OpenAI-compatible API
7. ✅ Verifying inference performance

The deployment is production-ready and can serve requests via the OpenAI-compatible API on port 8001.

---

**Last Updated:** 2026-02-05
