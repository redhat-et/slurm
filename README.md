# Slurm GPU Cluster on AWS with RHEL 10

This Ansible playbook automates the deployment of a Slurm cluster on AWS with 3 g6.2xlarge RHEL 10 instances, complete with NVIDIA GPU drivers and container support.

## Features

- **AWS Infrastructure**: Automated VPC, subnet, security group, and EC2 instance creation
- **NVIDIA GPU Support**: Full NVIDIA driver installation following Red Hat best practices
- **Container Runtime**: Podman with NVIDIA Container Toolkit for GPU-accelerated containers
- **Slurm Cluster**: Production-ready Slurm workload manager configured for GPU scheduling
- **High Availability**: One controller node and two compute nodes with GPU resources
- **Optional vLLM Models**: Deploy 3 production LLMs (Phi-3, Mistral-7B, Llama-3.1-8B) with one command

## Prerequisites

### Local Requirements

1. **Ansible** (version 2.15 or higher)
   ```bash
   pip install ansible
   ```

2. **Ansible Collections**
   ```bash
   ansible-galaxy collection install -r requirements.yml
   ```

3. **AWS CLI** configured with appropriate credentials
   ```bash
   aws configure
   ```

4. **Python boto3 library**
   ```bash
   pip install boto3 botocore
   ```

## Configuration

### Update Variables

Edit `group_vars/all.yml` to customize your deployment:

```yaml
# AWS Configuration
aws_region: us-east-1  # Change to your preferred region
key_name: "slurm-cluster-key"  # Your SSH key pair name. Expects key to be on your local system as ~/.ssh/id_rsa

# Instance Configuration
instance_type: g6.2xlarge  # GPU instance type
instance_count: 3  # Number of instances

# NVIDIA Configuration
nvidia_driver_version: "580.82.07"  # Update if needed

# Slurm Configuration
slurm_version: "24.05.4"  # Update to latest stable version
slurm_cluster_name: "gpu-cluster"
```

## Usage

### Deploy the Cluster

Run the main playbook to create infrastructure and configure the cluster:

```bash
ansible-playbook playbook.yml
```

This will:
1. Create AWS infrastructure (VPC, subnet, security group, instances)
2. Install and configure NVIDIA drivers on all nodes
3. Install NVIDIA Container Toolkit
4. Build and install Slurm from source
5. Configure Munge authentication
6. Set up Slurm controller and compute nodes
7. Configure GPU resource scheduling

**Note**: The full deployment takes approximately 30-45 minutes.

### Verify the Cluster

After deployment completes, SSH to the controller node:

```bash
ssh -i ~/.ssh/slurm-cluster-key.rsa ec2-user@<controller-public-ip>
```

Check cluster status:

```bash
# View cluster nodes
sinfo

# View detailed node information
scontrol show nodes

# Check GPU resources
sinfo -o "%N %G"

# Submit a test GPU job
srun --gres=gpu:1 nvidia-smi
```

### Test GPU Container Support

On any compute node:

```bash
# Test NVIDIA driver
nvidia-smi

# Test GPU container support
sudo podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable nvidia/cuda:13.0.0-base-ubi9 nvidia-smi
```

### Deploy vLLM Models (Optional)

After the cluster is running, optionally deploy three LLM models with vLLM:

```bash
ansible-playbook deploy-vllm-models.yml
```

This installs and configures:
- **Phi-3 Mini (3.8B)**: Fast, efficient small model
- **Mistral 7B Instruct**: General-purpose instruction model

See [VLLM_DEPLOYMENT.md](VLLM_DEPLOYMENT.md) for complete documentation.

Quick test:
```bash
ssh -i ~/.ssh/slurm-cluster-key.pem ec2-user@<controller-ip>
cd ~/vllm-jobs
./vllm-manager.sh serve phi3  # Start serving Phi-3
```

### Destroy the Cluster

When you're done, clean up all AWS resources:

```bash
ansible-playbook destroy.yml
```

## Cluster Architecture

### Node Configuration

- **Controller Node** (`slurm-node-0`):
  - Runs slurmctld (Slurm controller daemon)
  - Runs slurmdbd (Slurm database daemon)
  - Hosts MariaDB for accounting
  - Manages job scheduling and cluster state

- **Compute Nodes** (`slurm-node-1`, `slurm-node-2`):
  - Run slurmd (Slurm compute daemon)
  - Each has 1 NVIDIA L4 GPU (g6.2xlarge)
  - 8 vCPUs and 63GB RAM per node
  - Configured for GPU job execution

### GPU Scheduling

The cluster is configured with:
- Generic Resource Scheduling (GRES) for GPUs
- Automatic GPU detection using NVML
- GPU partition (`gpu`) for GPU jobs
- Consumable resources tracking (cores, memory, GPUs)

### Submit GPU Jobs

```bash
# Interactive GPU job
srun --gres=gpu:1 --pty bash

# Batch GPU job
sbatch --gres=gpu:1 my_gpu_script.sh

# Multiple GPUs (if using larger instances)
sbatch --gres=gpu:2 my_multi_gpu_script.sh
```

## Security Considerations

**WARNING**: This playbook creates a "wide open" security group (0.0.0.0/0) for simplicity. This is **NOT RECOMMENDED FOR PRODUCTION**.

For production use:
1. Restrict security group rules to specific IP ranges
2. Use VPN or bastion host for access
3. Enable AWS Systems Manager Session Manager
4. Implement proper IAM roles and policies
5. Enable CloudWatch logging
6. Use encrypted EBS volumes
7. Configure firewall rules on instances

## Troubleshooting

### CDI Configuration Issues

If GPU containers stop working after a reboot:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### Slurm Node Not Responding

```bash
# On controller
scontrol update nodename=slurm-node-1 state=resume

# Check node state
sinfo -Nel
```

### Munge Authentication Errors

```bash
# Restart munge on all nodes
sudo systemctl restart munge

# Verify munge is working
munge -n | unmunge
```

### Database Connection Issues

```bash
# On controller node
sudo systemctl status mariadb
sudo systemctl status slurmdbd

# Check database
mysql -u slurm -pslurmdbpass slurm_acct_db
```

## Costs

Running this cluster will incur AWS costs:
- **g6.2xlarge instances**: ~$0.75/hour each (3 instances = ~$2.25/hour)
- **EBS storage**: 100GB per instance
- **Data transfer**: Minimal for cluster communication

**Estimated cost**: ~$54/day if running 24/7

Remember to destroy the cluster when not in use!

## File Structure

```
.
├── ansible.cfg                      # Ansible configuration
├── playbook.yml                     # Main cluster deployment
├── deploy-vllm-models.yml          # Optional: Deploy LLM models
├── destroy.yml                      # Cleanup playbook
├── requirements.yml                 # Ansible collection requirements
├── setup.sh                         # Quick setup script
├── group_vars/
│   ├── all.yml                     # Cluster configuration
│   └── vllm.yml                    # vLLM model configuration
├── templates/
│   ├── slurm.conf.j2               # Slurm main configuration
│   ├── slurmdbd.conf.j2            # Slurm database configuration
│   ├── cgroup.conf.j2              # Cgroup configuration
│   ├── gres.conf.j2                # GPU resource configuration
│   ├── vllm_serve_model.sh.j2      # vLLM serving job template
│   ├── vllm_batch_inference.sh.j2  # vLLM batch job template
│   └── vllm_test_inference.py.j2   # vLLM test script
├── examples/                        # Example Slurm job scripts
│   ├── test_gpu.sh
│   ├── container_gpu.sh
│   ├── multi_job.sh
│   └── pytorch_test.sh
├── inventory/
│   └── hosts                        # Generated inventory file
└── Documentation/
    ├── README.md                    # This file
    ├── QUICKSTART.md                # Quick reference guide
    ├── VLLM_DEPLOYMENT.md          # vLLM deployment guide
    ├── PRE_DEPLOYMENT_CHECKLIST.md # Pre-flight checklist
    └── CONTRIBUTING.md              # Customization guide
```

## References

- [NVIDIA RHEL 10 Driver Installation Guide](https://developer.nvidia.com/datacenter-driver-downloads)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Slurm GPU Scheduling](https://slurm.schedmd.com/gres.html)
- [Red Hat Enterprise Linux 10](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/10)

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!
