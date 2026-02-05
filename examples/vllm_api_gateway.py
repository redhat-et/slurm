#!/usr/bin/env python3
"""
Simple API Gateway for vLLM Models on Slurm

This script provides a unified interface to query all three deployed models.
It automatically detects which nodes the models are running on.

Usage:
    python3 vllm_api_gateway.py

Then query via curl:
    curl -X POST http://localhost:5000/generate \
      -H "Content-Type: application/json" \
      -d '{"model": "phi3", "prompt": "Hello!"}'
"""

from flask import Flask, request, jsonify
import subprocess
import requests
import json
import re

app = Flask(__name__)

def get_model_endpoints():
    """Query Slurm to find where models are running"""
    try:
        result = subprocess.run(
            ['squeue', '-u', subprocess.getenv('USER'), '-h', '-o', '%i %j %N'],
            capture_output=True,
            text=True
        )

        endpoints = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            job_id, job_name, node = line.split()

            if 'vllm-serve-phi3' in job_name:
                endpoints['phi3'] = f"http://{node}:8000"
            elif 'vllm-serve-mistral' in job_name:
                endpoints['mistral'] = f"http://{node}:8001"
            elif 'vllm-serve-llama3' in job_name:
                endpoints['llama3'] = f"http://{node}:8002"

        return endpoints
    except Exception as e:
        print(f"Error querying Slurm: {e}")
        return {}

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their endpoints"""
    endpoints = get_model_endpoints()

    models = []
    for name, endpoint in endpoints.items():
        try:
            response = requests.get(f"{endpoint}/v1/models", timeout=2)
            status = "online" if response.status_code == 200 else "error"
        except:
            status = "offline"

        models.append({
            "name": name,
            "endpoint": endpoint,
            "status": status
        })

    return jsonify({
        "models": models,
        "count": len(models)
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text using specified model"""
    data = request.json

    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    model = data.get('model', 'phi3')
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 200)
    temperature = data.get('temperature', 0.7)

    endpoints = get_model_endpoints()

    if model not in endpoints:
        return jsonify({
            "error": f"Model '{model}' not found or not running",
            "available_models": list(endpoints.keys())
        }), 404

    try:
        response = requests.post(
            f"{endpoints[model]}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return jsonify({
            "model": model,
            "prompt": prompt,
            "response": result['choices'][0]['text'],
            "usage": result.get('usage', {})
        })

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": f"Error querying model: {str(e)}",
            "model": model,
            "endpoint": endpoints[model]
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat completion using specified model"""
    data = request.json

    if not data or 'messages' not in data:
        return jsonify({"error": "Missing 'messages' in request"}), 400

    model = data.get('model', 'phi3')
    messages = data['messages']
    max_tokens = data.get('max_tokens', 200)
    temperature = data.get('temperature', 0.7)

    endpoints = get_model_endpoints()

    if model not in endpoints:
        return jsonify({
            "error": f"Model '{model}' not found or not running",
            "available_models": list(endpoints.keys())
        }), 404

    try:
        response = requests.post(
            f"{endpoints[model]}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return jsonify({
            "model": model,
            "response": result['choices'][0]['message']['content'],
            "usage": result.get('usage', {})
        })

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": f"Error querying model: {str(e)}",
            "model": model,
            "endpoint": endpoints[model]
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    endpoints = get_model_endpoints()

    health_status = {}
    for name, endpoint in endpoints.items():
        try:
            response = requests.get(f"{endpoint}/health", timeout=2)
            health_status[name] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            health_status[name] = "unreachable"

    overall = "healthy" if any(s == "healthy" for s in health_status.values()) else "unhealthy"

    return jsonify({
        "status": overall,
        "models": health_status,
        "active_models": len([s for s in health_status.values() if s == "healthy"])
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "name": "vLLM Multi-Model API Gateway",
        "version": "1.0",
        "endpoints": {
            "/models": "GET - List available models",
            "/generate": "POST - Generate text (prompt, model, max_tokens, temperature)",
            "/chat": "POST - Chat completion (messages, model, max_tokens, temperature)",
            "/health": "GET - Health check"
        },
        "example": {
            "generate": {
                "method": "POST",
                "url": "/generate",
                "body": {
                    "model": "phi3",
                    "prompt": "Explain AI in simple terms:",
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            },
            "chat": {
                "method": "POST",
                "url": "/chat",
                "body": {
                    "model": "mistral",
                    "messages": [
                        {"role": "user", "content": "Hello!"}
                    ],
                    "max_tokens": 200
                }
            }
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("vLLM Multi-Model API Gateway")
    print("=" * 60)
    print()
    print("Starting server on http://0.0.0.0:5000")
    print()
    print("Available endpoints:")
    print("  GET  /           - API documentation")
    print("  GET  /models     - List available models")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Generate text")
    print("  POST /chat       - Chat completion")
    print()
    print("Example usage:")
    print("  curl -X POST http://localhost:5000/generate \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"model\": \"phi3\", \"prompt\": \"Hello!\"}'")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False)
