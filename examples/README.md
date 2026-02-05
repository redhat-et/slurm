# Slurm Job Examples

This directory contains example job scripts to test your Slurm GPU cluster.

## Table of Contents

- [Basic GPU Tests](#basic-gpu-tests)
- [Container GPU Tests](#container-gpu-tests)
- [Array Jobs](#array-jobs)
- [PyTorch Tests](#pytorch-tests)
- [vLLM Model Examples](#vllm-model-examples) (requires vLLM deployment)

---

## Basic GPU Test

Test GPU availability and driver:

```bash
sbatch test_gpu.sh
```

This will:
- Display NVIDIA driver information
- List GPU devices
- Show CUDA version
- Display GPU utilization

## Container GPU Test

Test GPU access from containers:

```bash
sbatch container_gpu.sh
```

This will:
- Run nvidia-smi in a CUDA container using Podman
- Verify GPU is accessible from containerized workloads

## Array Job Test

Test parallel job submission with array jobs:

```bash
sbatch multi_job.sh
```

This will:
- Submit 5 parallel jobs
- Each job runs independently on available GPUs
- Useful for parameter sweeps and batch processing

## PyTorch GPU Test

Test PyTorch with GPU support:

```bash
sbatch pytorch_test.sh
```

This will:
- Run PyTorch in a container
- Verify CUDA availability
- Test tensor operations on GPU
- Display GPU device information

---

## vLLM Model Examples

**Note**: These examples require the optional vLLM deployment:
```bash
ansible-playbook deploy-vllm-models.yml
```

### Compare All Models

Run all three models with the same prompt and compare performance:

```bash
sbatch vllm_compare_models.sh
```

This will test Phi-3, Mistral-7B, and Llama-3.1-8B sequentially and compare:
- Response quality
- Load time
- Generation speed
- Throughput (tokens/second)

### Start All Models Simultaneously

Launch all three models at once (one per compute node):

```bash
./vllm_multi_model_serve.sh
```

This will submit serving jobs for all models and display their endpoints.

### API Gateway

Run a unified API gateway that routes requests to all models:

```bash
# First, ensure models are running
./vllm_multi_model_serve.sh

# Wait for models to start, then launch gateway
python3 vllm_api_gateway.py
```

Query the gateway:

```bash
# List available models
curl http://localhost:5000/models

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3",
    "prompt": "Explain quantum computing:",
    "max_tokens": 200
  }'

# Chat completion
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

See [VLLM_DEPLOYMENT.md](../VLLM_DEPLOYMENT.md) for complete vLLM documentation.

## Checking Job Status

```bash
# View all jobs
squeue

# View your jobs
squeue -u $USER

# View specific job
scontrol show job <job-id>

# View job output
tail -f <output-file>
```

## Viewing Results

Job output files are created in the current directory:
- `<jobname>_<jobid>.out` - Standard output
- `<jobname>_<jobid>.err` - Standard error

```bash
# View job output
cat test_gpu_12345.out

# Follow job output in real-time
tail -f test_gpu_12345.out
```

## Interactive Testing

For development and debugging, use interactive jobs:

```bash
# Request interactive session with GPU
srun --gres=gpu:1 --pty bash

# Once in the job, test GPU
nvidia-smi

# Run containers interactively
podman run -it --device nvidia.com/gpu=all --security-opt=label=disable nvidia/cuda:13.0.0-base-ubi9 bash

# Exit when done
exit
```

## Canceling Jobs

```bash
# Cancel a specific job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER

# Cancel array job
scancel <array-job-id>
```

## Customizing Jobs

### Adjust Resources

Edit the `#SBATCH` directives in the scripts:

```bash
#SBATCH --cpus-per-task=8      # Request 8 CPUs
#SBATCH --mem=16G              # Request 16GB memory
#SBATCH --time=01:00:00        # Set 1 hour time limit
```

### Change Output Location

```bash
#SBATCH --output=logs/%j.out   # Save to logs directory
#SBATCH --error=logs/%j.err
```

### Email Notifications

```bash
#SBATCH --mail-type=END,FAIL   # Email on completion/failure
#SBATCH --mail-user=you@example.com
```

## Best Practices

1. **Test interactively first** before submitting batch jobs
2. **Start small** with short time limits, then increase
3. **Monitor resources** to ensure efficient GPU utilization
4. **Clean up** old output files regularly
5. **Use array jobs** for parameter sweeps instead of submitting many individual jobs

## Troubleshooting

### Job Pending

```bash
# Check why job is pending
squeue -j <job-id> --start

# View job priority
sprio
```

### Job Failed

```bash
# Check error output
cat <jobname>_<jobid>.err

# View job accounting
sacct -j <job-id> --format=JobID,State,ExitCode,DerivedExitCode

# Check node logs
ssh <node> sudo tail /var/log/slurm/slurmd.log
```

### GPU Not Available in Job

```bash
# Verify GPU allocation
scontrol show job <job-id> | grep Gres

# Check node GPU status
sinfo -o "%N %G %C"

# On compute node, regenerate CDI
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
