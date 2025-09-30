# SwarmMap Docker Build Instructions for Windows Lambda Machine

## Prerequisites
- Docker Desktop with WSL2 backend
- NVIDIA Container Toolkit configured
- VcXsrv running for X11 forwarding (already detected)

## Step 1: Test Docker GPU Access and Build Base Image

### 1.1 First, test if Docker can access your GPUs:

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-devel-ubuntu20.04 nvidia-smi
```

**Expected output**: Should show your 2x RTX A6000 GPUs

### 1.2 Navigate to the SwarmMap directory:

```powershell
cd "C:\path\to\SwarmMap\docker"
```

### 1.3 Build the base image:

```powershell
docker build -f Dockerfile.base -t swarmmap:base .
```

### 1.4 Test the base image and start interactive container:

```powershell
docker run --rm --gpus all -it swarmmap:base
```

Inside the container, verify CUDA installation:
```bash
nvcc --version
nvidia-smi
```

**Report back**: Does `nvidia-smi` show your GPUs inside the container?

---

## Step 2: Install Build Dependencies (Execute this after Step 1 succeeds)

Once you confirm Step 1 works, I'll provide the next Dockerfile for installing CMake, build tools, and other dependencies.

---

## Notes:
- We're using CUDA 11.8 (compatible with your CUDA 12.6 driver and RTX A6000 Ampere architecture)
- Each step will be tested before moving to the next
- The container will be interactive so you can verify each build step
