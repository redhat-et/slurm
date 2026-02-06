# Slurm Deployment on OpenShift

## Prerequisites

- OpenShift cluster with GPU-enabled node(s)
- GPU operator configured and functional

## Installation Steps

### 1. Enable GPU Node

Ensure your GPU node is properly labeled and the NVIDIA GPU Operator is installed and configured.

### 2. Install Slinky Operator

Install the **Slinky** operator from the OpenShift OperatorHub using default configuration. The operator is running in the `slinky` namespace.

### 3. Set up a new project and SCC for Slurm Instance

Let's set up `slurm` namespace and a `privileged` SCC for the `default` SA in this namespace.

```bash
oc adm new-project slurm
oc adm policy add-scc-to-user privileged -n slurm -z default
```

### 4. Create a shared home RWX PVC

You need to have RWX capabilities in your cluster. In our case we solved it by adding EFS provisioner.

```bash
oc apply -n slurm -f pvc.yaml
```

### 3. Deploy Slurm Instance

Create a `values.yaml` with custom image references and GPU configuration, see [values.yaml](./values.yaml) for reference.

Deploy using Helm:

```bash
helm upgrade -i slurm oci://ghcr.io/slinkyproject/charts/slurm \
  --namespace=slurm \
  --version 1.0.1 \
  -f values.yaml \
  --set-literal loginsets.slinky.rootSshAuthorizedKeys="<your-ssh-public-key>"
```

## Accessing Slurm

Connect to the Slurm login node via SSH tunneled through OpenShift:

```bash
ssh -o ProxyCommand='oc exec -i -n slurm svc/%h -- socat STDIO TCP:localhost:22' root@slurm-login-slinky
```

This uses `oc exec` as a proxy to forward SSH traffic to the login service without requiring a Route or NodePort.

## Verify GPU access

Once in Slurm login node, you can verify the worker nodes have access to GPU:

```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
slinky       up   infinite      1   idle slinky-0
all*         up   infinite      1   idle slinky-0


$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)

$ srun nvidia-smi
Fri Feb  6 11:07:25 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
| N/A   30C    P8             17W /   72W |       0MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## Key Configuration Notes

- **GPU Auto-Detection**: `gres.conf: AutoDetect=nvidia` enables automatic GPU discovery
- **Custom Images**: Using `quay.io/slinky-on-openshift/*` images built for OpenShift compatibility
- **Privileged Login**: Login pods require privileged security context for SSH access
- **GPU Resources**: Worker nodes request and limit GPU allocation via `nvidia.com/gpu` resource
