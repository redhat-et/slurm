#!/bin/bash
set -e

echo "========================================"
echo "Slurm GPU Cluster Setup Script"
echo "========================================"
echo ""

# Check if ansible is installed
if ! command -v ansible &> /dev/null; then
    echo "ERROR: Ansible is not installed."
    echo "Please install Ansible: pip install ansible"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI is not installed."
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS credentials not configured."
    echo "Please run: aws configure"
    exit 1
fi

echo "✓ Ansible installed"
echo "✓ AWS CLI installed and configured"
echo ""

# Install Ansible collections
echo "Installing required Ansible collections..."
ansible-galaxy collection install -r requirements.yml

echo ""
echo "========================================"
echo "Ready to deploy Slurm cluster!"
echo "========================================"
echo ""
echo "IMPORTANT: Before running the playbook, make sure you have:"
echo "1. Created an SSH key pair in AWS (or update key_name in group_vars/all.yml)"
echo "2. Reviewed and updated configuration in group_vars/all.yml"
echo "3. Ensured you have quota for 3x g6.2xlarge instances in your target region"
echo ""
echo "To deploy the cluster, run:"
echo "  ansible-playbook playbook.yml"
echo ""
echo "To destroy the cluster later, run:"
echo "  ansible-playbook destroy.yml"
echo ""
