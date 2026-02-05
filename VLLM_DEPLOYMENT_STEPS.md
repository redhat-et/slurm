# vLLM Model Deployment Steps

This document provides a detailed breakdown of the steps used to deploy and test vLLM models (specifically Mistral 7B) on the Slurm GPU cluster.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Model Configuration](#model-configuration)
4. [Deployment Process](#deployment-process)
5. [Job Submission](#job-submission)
6. [Testing and Verification](#testing-and-verification)
7. [Monitoring](#monitoring)
8. [Cleanup](#cleanup)

---

## Prerequisites

**Required Infrastructure:**
- AWS account with EC2 access
- Slurm cluster deployed (1 controller + 2 compute nodes)
- NVIDIA GPU instances (g6.2xlarge with L4 GPUs)
- NVIDIA drivers and CUDA toolkit installed

**Required Files:**
- `playbook.yml` - Main infrastructure deployment
- `continue-slurm-install.yml` - Slurm installation completion
- `deploy-vllm-models.yml` - vLLM deployment playbook
- `group_vars/all.yml` - Cluster configuration
- `group_vars/vllm.yml` - Model configuration
- `inventory/hosts` - Ansible inventory file

---

## Infrastructure Setup

### Step 1: Deploy Base Infrastructure

```bash
# Deploy AWS infrastructure and install base Slurm components
ansible-playbook playbook.yml
```

**What this does:**
- Creates VPC, security groups, and EC2 instances
- Installs NVIDIA drivers (580.82.07) and CUDA 13.0
- Installs base system packages
- Downloads Slurm source code
- Creates necessary users and directories

**Expected outcome:** Playbook completes but Slurm build fails due to missing munge-devel on RHEL 10.

### Step 2: Complete Slurm Installation

```bash
# Complete Slurm installation with RHEL 10 fixes
ansible-playbook -i inventory/hosts continue-slurm-install.yml
```

**What this does:**
- Builds Munge 0.5.16 from source (provides authentication)
- Configures Munge service with RHEL 10 compatibility fixes
- Compiles Slurm 24.05.4 with GPU support
- Configures Slurm controller (slurmctld, slurmdbd)
- Configures compute nodes (slurmd)
- Sets up GPU resource scheduling (GRES)

**Expected outcome:** Functional Slurm cluster with GPU support.

**Verification:**
```bash
ssh ec2-user@<controller-ip>
sudo sinfo
sudo scontrol show nodes
```

---

## Model Configuration

### Step 3: Configure Models in vllm.yml

Edit `group_vars/vllm.yml` to define which models to deploy:

```yaml
vllm_models:
  - name: phi3
    model_id: microsoft/Phi-3-mini-4k-instruct
    description: "Phi-3 Mini (3.8B parameters) - Fast, efficient small model"
    context_length: 4096
    use_case: "Quick responses, low latency, chat"
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    port: 8000

  - name: mistral
    model_id: mistralai/Mistral-7B-Instruct-v0.3
    description: "Mistral 7B Instruct - Excellent general-purpose model"
    context_length: 32768
    use_case: "General chat, instruction following, reasoning"
    gpu_memory_utilization: 0.90
    max_model_len: 8192
    port: 8001
```

**Key parameters:**
- `name`: Short identifier used in job scripts
- `model_id`: HuggingFace model repository
- `port`: API server port (must be unique per model)
- `gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0)
- `max_model_len`: Maximum sequence length

---

## Deployment Process

### Step 4: Deploy vLLM and Models

```bash
# Deploy vLLM infrastructure and download models
ansible-playbook -i inventory/hosts deploy-vllm-models.yml
```

**What this playbook does:**

#### Phase 1: System-wide vLLM Installation (all nodes)
1. Installs Python 3.12 development packages
2. Creates directories:
   - `/opt/vllm` - Installation directory
   - `/opt/models` - Model cache directory
   - `/var/log/vllm` - Log directory
3. Installs vLLM 0.15.1 system-wide via pip3:
   - `vllm`
   - `huggingface_hub`
   - `accelerate`
   - `transformers`
4. Configures HuggingFace cache location

#### Phase 2: Model Download (compute nodes only)
1. Creates Python script to download models from HuggingFace
2. Downloads model weights to `/opt/models/`:
   - Uses `snapshot_download()` from huggingface_hub
   - Downloads in parallel with 4 workers
   - Resumes interrupted downloads
3. Sets ownership to `slurm` user recursively

**Downloaded model structure:**
```
/opt/models/
├── models--microsoft--Phi-3-mini-4k-instruct/
│   ├── blobs/          # Model weight files
│   ├── refs/           # Version references
│   └── snapshots/      # Model snapshots
└── models--mistralai--Mistral-7B-Instruct-v0.3/
    ├── blobs/
    ├── refs/
    └── snapshots/
```

#### Phase 3: Job Script Creation (controller only)
1. Creates `/home/slurm/vllm-jobs/` directory
2. Generates Slurm batch scripts for each model:
   - `serve_phi3.sh` - Start API server for Phi-3
   - `serve_mistral.sh` - Start API server for Mistral
   - `batch_phi3.sh` - Batch inference for Phi-3
   - `batch_mistral.sh` - Batch inference for Mistral
3. Creates management script: `vllm-manager.sh`
4. Creates documentation: `README.md`

**Expected outcome:**
- vLLM installed on all nodes
- Models downloaded and cached on compute nodes
- Job scripts ready on controller

**Verification:**
```bash
ssh ec2-user@<controller-ip>
ls -lh /home/slurm/vllm-jobs/
sudo -u slurm ls -lh /opt/models/models--mistralai*/
```

---

## Job Submission

### Step 5: Submit vLLM Serving Job

There are two methods to submit jobs:

#### Method 1: Using vllm-manager.sh (Recommended)

```bash
ssh ec2-user@<controller-ip>
sudo -u slurm bash -c 'cd /home/slurm/vllm-jobs && ./vllm-manager.sh serve mistral'
```

#### Method 2: Direct sbatch Submission

```bash
ssh ec2-user@<controller-ip>
sudo -u slurm bash -c 'cd /home/slurm/vllm-jobs && sbatch serve_mistral.sh'
```

**What happens when job is submitted:**

1. **Slurm schedules the job:**
   - Allocates 1 GPU from the `gpu` partition
   - Assigns 8 CPU cores
   - Selects available compute node (e.g., slurm-node-1)

2. **Job starts on compute node:**
   - Sets environment variables:
     ```bash
     export HF_HOME=/opt/models
     export CUDA_VISIBLE_DEVICES=0
     export VLLM_WORKER_MULTIPROC_METHOD=spawn
     ```
   - Displays GPU information via `nvidia-smi -L`
   - Lists cached model files

3. **vLLM server initialization:**
   ```bash
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
   ```

4. **Model loading phases:**
   - **Configuration loading** (~5 seconds)
     - Reads model config from cache
     - Determines architecture (MistralForCausalLM)
   - **Weight loading** (~2 minutes)
     - Loads safetensors files from `/opt/models/`
     - Allocates GPU memory (~13.5 GB for Mistral 7B)
   - **Compilation** (~2 minutes)
     - torch.compile optimization
     - CUDA graph generation
   - **Server ready** (~4 minutes total)
     - Binds to 0.0.0.0:8001
     - Ready to accept API requests

**Expected output locations:**
- Standard output: `/home/slurm/vllm-jobs/outputs/serve_mistral_<JOB_ID>.out`
- Standard error: `/home/slurm/vllm-jobs/outputs/serve_mistral_<JOB_ID>.err`

**Note:** Output files are created on the compute node where the job runs, not on the controller.

---

## Testing and Verification

### Step 6: Check Job Status

```bash
# View job queue
sudo squeue

# View detailed job info
sudo sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList

# Check running jobs for slurm user
sudo squeue -u slurm
```

**Expected output:**
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    7       gpu vllm-ser    slurm  R       5:30      1 slurm-node-1
```

### Step 7: Verify Model Loading

**Check output logs on compute node:**
```bash
# SSH to controller first
ssh ec2-user@<controller-ip>

# Use ansible to check logs on compute node
ansible slurm_compute -i inventory/hosts -m shell \
  -a "tail -100 /home/slurm/vllm-jobs/outputs/serve_mistral_*.out" \
  -b --limit "<compute-node-ip>"
```

**Look for these success indicators:**
```
INFO 02-05 18:21:20 [utils.py:325]        █     █     █▄   ▄█
INFO 02-05 18:21:20 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
INFO 02-05 18:21:27 [model.py:541] Resolved architecture: MistralForCausalLM
INFO 02-05 18:25:10 [default_loader.py:291] Loading weights took 113.27 seconds
INFO 02-05 18:25:10 [gpu_model_runner.py:4130] Model loading took 13.51 GiB memory
```

### Step 8: Test API Endpoints

**From controller node (via Ansible):**

```bash
# Test 1: List available models
ansible slurm_controller -i inventory/hosts -m shell \
  -a "curl -s http://slurm-node-1:8001/v1/models" -b
```

**Expected response:**
```json
{
  "object": "list",
  "data": [{
    "id": "mistralai/Mistral-7B-Instruct-v0.3",
    "object": "model",
    "max_model_len": 8192
  }]
}
```

**Test 2: Generate text completion**

```bash
ansible slurm_controller -i inventory/hosts -m shell \
  -a 'curl -s http://slurm-node-1:8001/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"mistralai/Mistral-7B-Instruct-v0.3\", \
           \"prompt\": \"Explain GPU computing in one sentence:\", \
           \"max_tokens\": 100, \
           \"temperature\": 0.7}"' -b
```

**Expected response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1770316269,
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "choices": [{
    "index": 0,
    "text": " GPU computing leverages...",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 109,
    "completion_tokens": 100
  }
}
```

**Test 3: Chat completions (OpenAI-compatible)**

```bash
curl -s http://slurm-node-1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
      {"role": "user", "content": "What is vLLM?"}
    ],
    "max_tokens": 150
  }'
```

---

## Monitoring

### Step 9: Monitor GPU Usage

**Check GPU memory and utilization:**

```bash
# Via Ansible from local machine
ansible slurm_compute -i inventory/hosts -m shell \
  -a "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv" \
  -b --limit "<compute-node-ip>"
```

**Expected output:**
```
name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu
NVIDIA L4, 0 %, 21094 MiB, 23034 MiB, 46
```

**Interpretation:**
- **memory.used**: 21,094 MiB = Model loaded in GPU memory (~13.5 GB model + ~7 GB KV cache)
- **utilization.gpu**: 0% when idle, spikes to 80-100% during inference
- **temperature**: Should stay below 80°C

**Monitor vLLM process:**

```bash
ansible slurm_compute -i inventory/hosts -m shell \
  -a "ps aux | grep vllm | grep -v grep" \
  -b --limit "<compute-node-ip>"
```

**Expected output:**
```
slurm  88794  3.2  3.0 7797860 941176 ?  Sl  18:21  0:07 python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8001 ...
```

### Step 10: Watch Job Logs in Real-time

**Monitor job output as it runs:**

```bash
# SSH to controller
ssh ec2-user@<controller-ip>

# Find the compute node running the job
NODE=$(sudo squeue -h -o "%N" -j <JOB_ID>)

# SSH to that node and tail logs
ssh $NODE
tail -f /home/slurm/vllm-jobs/outputs/serve_mistral_*.out
```

---

## Cleanup

### Step 11: Cancel Running Jobs

**Cancel specific job:**
```bash
sudo scancel <JOB_ID>
```

**Cancel all jobs for slurm user:**
```bash
sudo scancel -u slurm
```

**Cancel all vLLM serving jobs:**
```bash
sudo scancel --name=vllm-serve-mistral
```

**Verify cancellation:**
```bash
sudo squeue
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Job completes in seconds without loading model

**Symptoms:**
- Job shows `COMPLETED` status immediately
- No output files created
- API not responding

**Cause:** Gated repository (requires HuggingFace authentication)

**Solution:** Use ungated models only (Phi-3, Mistral)

#### Issue 2: Out of memory error

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Reduce `--gpu-memory-utilization` (try 0.80 or 0.70)
- Reduce `--max-model-len` (e.g., 4096 instead of 8192)
- Use smaller model (Phi-3 instead of Mistral)

#### Issue 3: API not responding

**Check if port is listening:**
```bash
ansible slurm_compute -i inventory/hosts -m shell \
  -a "ss -tlnp | grep :8001" -b
```

**Check for errors in stderr:**
```bash
ansible slurm_compute -i inventory/hosts -m shell \
  -a "cat /home/slurm/vllm-jobs/outputs/serve_mistral_*.err" -b
```

#### Issue 4: Model loading takes too long

**Normal loading times:**
- Phi-3 Mini (3.8B): ~2 minutes
- Mistral 7B: ~4 minutes

**If slower than expected:**
- Check network bandwidth (model downloads from HuggingFace)
- Verify model is cached in `/opt/models/`
- Check GPU availability with `nvidia-smi`

---

## Performance Metrics

### Mistral 7B Benchmarks

**Model Loading:**
- Configuration: ~5 seconds
- Weight loading: ~113 seconds
- torch.compile: ~2 minutes
- **Total startup: ~4 minutes**

**Resource Usage:**
- GPU Memory: 21 GB / 23 GB (91%)
- System Memory: ~950 MB
- CPU: 8 cores allocated

**Inference Performance:**
- Tokens per second: ~40-60 (varies by sequence length)
- Latency (first token): ~100-200ms
- Latency (subsequent tokens): ~20-30ms each

**API Endpoint:** `http://slurm-node-1:8001/v1/{models,completions,chat/completions}`

---

## Next Steps

1. **Deploy multiple models simultaneously:**
   ```bash
   ./vllm-manager.sh serve phi3
   ./vllm-manager.sh serve mistral
   ```

2. **Set up load balancing** with HAProxy for production use

3. **Add monitoring** with Prometheus/Grafana for GPU metrics

4. **Configure auto-scaling** based on queue depth

5. **Implement model caching** to pre-download models

---

## References

- vLLM Documentation: https://docs.vllm.ai/
- Slurm Documentation: https://slurm.schedmd.com/
- HuggingFace Models: https://huggingface.co/models
- NVIDIA GPU Monitoring: https://developer.nvidia.com/nvidia-system-management-interface

---

**Last Updated:** 2026-02-05
