---
name: Detect Config
description: Analyze and collect comprehensive system information across Windows, Mac, Linux, Google Colab, and cloud environments with focus on AI/ML capabilities.
version: 1.0.0
---

# Detect Config - System Detection


> [!NOTE]
> This skill enables the LLM to inspect its runtime environment.
> The LLM executes this tool to discover system capabilities, available resources, and constraints.

This skill provides detailed system analysis and hardware/software inventory optimized for AI/ML workloads.

### AI/LLM & ML Capabilities
- **LLM Environment**: Claude/Anthropic specific env vars, context window, token budgets
- **Local LLMs**: Ollama (models, running servers), Llama.cpp (binaries, GGUF models)
- **ML Frameworks**: PyTorch (CUDA/MPS support), TensorFlow, JAX, HuggingFace
- **Accelerators**: NVIDIA (VRAM, Power, Temp, MIG), AMD ROCm, Apple Silicon (Neural Engine), Google TPU
- **Performance**: Floating point capabilities (AVX512/AMX), RAM bandwidth estimation

### Security & Audit (Local Only)
- **Credential Scanning**: Detects exposed secrets in .env, config files, and history
- **SSH & Keys**: Audits SSH private key permissions and agent status
- **Browser Data**: Detects unencrypted browser session tokens/cookies risk
- **Network Security**: Open ports, active listeners, promiscuous mode detection
- **Permissions**: Sudo access, container privileges, file system write access

### Infrastructure & Services
- **MCP Servers**: Detects installed and running Model Context Protocol servers
- **Containers**: Docker (daemon, running containers), Kubernetes (pods, namespaces), Podman
- **Cloud**: AWS/GCP/Azure instance metadata retrieval and region detection
- **Databases**: MySQL/PostgreSQL client configs and connection profiles
- **Remote Access**: SSH, VNC, RDP, TeamViewer active sessions

### Hardware & System
- **Compute**: CPU deep dive (cores, cache, micro-arch), detailed Memory hierarchy
- **Storage**: Disk I/O benchmarks, partition usage, file system types
- **Network**: DNS resolution speed, public/private IPs, geolocation, Wi-Fi signal
- **Environment**: OS details, Kernel, Virtualization (VMware/KVM/Hyper-V), Sandbox detection

## Usage

```bash
# Sortie JSON formatée (par défaut)
python3 detectConfig.py

# Sortie JSON compacte
python3 detectConfig.py --json

# Sortie lisible avec séparations par section
python3 detectConfig.py --readable

# Installer psutil automatiquement si manquant
python3 detectConfig.py --install-optional
```

### Options
- `--json`: Output compact JSON (default is pretty-printed)
- `--human`: Reorganize output into a human-friendly hierarchy
- `--readable`: Print with visual separators between sections
- `--output <file>`: Save the JSON report to a file
- `--install-optional`: Attempt to install `psutil` if missing
- `DETECTOS_AUTO_INSTALL=1`: Environment variable to auto-install dependencies

## Output

The script returns a detailed JSON object with the following top-level keys:

### Core System
- `os`, `environment`, `datetime` (local/UTC/epoch)
- `health` (temperatures, fans, uptime), `usage` (cpu load, memory)

### Hardware Resources
- `cpu`, `cpu_advanced` (flags, cache), `memory` (RAM/Swap)
- `storage`, `storage_performance` (I/O), `peripherals` (USB)
- `gpus_tpus` (NVIDIA/AMD/Apple/TPU specifics)
- `accelerators` (compute capabilities)

### AI & Machine Learning
- `ai_llm_metrics` (Anthropic environment, context window)
- `ml_capabilities` (Torch/TensorFlow availability)
- `ollama` (models, server status), `llamacpp`, `pytorch`

### Network & Connectivity
- `network` (interfaces, IPs), `internet` (connectivity check)
- `dns` (resolvers, DoH/DoT), `speedtest` (bandwidth)
- `open_ports`, `geo` (location based on IP)

### Security & Access
- `security_audit` (credentials, SSH keys scan)
- `user_permissions` (sudo, filesystem), `remote_access`
- `servers_and_passwords` (MCP/FTP servers)

### Software & Services
- `installed_software` (pip/brew/apt counts), `languages_tools`
- `containers` (Docker/K8s/Podman), `services` (active listeners)
- `summary_sizes` (total RAM/DiskGB)

## Use Cases

- **ML Setup Validation**: Verify environment readiness before training
- **Hardware Inventory**: Comprehensive system capabilities audit
- **Performance Analysis**: Identify bottlenecks and optimization opportunities
- **Deployment Planning**: Assess cloud/edge deployment feasibility
- **Troubleshooting**: Diagnose hardware/software configuration issues
