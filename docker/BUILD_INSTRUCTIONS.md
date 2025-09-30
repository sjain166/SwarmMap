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

## Step 2: Build OpenCV 4.2.0 (NO CUDA) ✓ COMPLETED

SwarmMap has its own CUDA implementation in `.cu` files, so OpenCV doesn't need CUDA support.

```powershell
cd "C:\Users\sj99\Desktop\SwarmMap\docker"
docker build -f Dockerfile.opencv_nocuda -t swarmmap:opencv .
```

✓ **Build completed successfully** (10-15 minutes)

---

## Step 3: Build Pangolin v0.5 for Visualization

Pangolin provides the map viewer and GUI for SwarmMap.

### 3.1 Build Pangolin:

```powershell
docker build -f Dockerfile.pangolin -t swarmmap:pangolin .
```

**What's happening**:
- Installing OpenGL and visualization dependencies
- Cloning Pangolin v0.5 (specific version required by SwarmMap)
- Building with 8 parallel jobs

**Expected time**: 3-5 minutes

**Report back**: Did the build complete successfully?

---

## Step 4: Install Remaining Dependencies

Install Eigen3, Boost (≥1.70.0), and spdlog.

### 4.1 Build dependencies image:

```powershell
docker build -f Dockerfile.dependencies -t swarmmap:deps .
```

**What's happening**:
- Installing Eigen3 3.3.7 (linear algebra)
- Installing Boost 1.71.0 (serialization & networking)
- Installing spdlog (logging)

**Expected time**: 1-2 minutes

**Report back**: Did the build complete successfully?

---

## Notes:
- We're using CUDA 11.8 (compatible with your CUDA 12.6 driver and RTX A6000 Ampere architecture)
- Each step will be tested before moving to the next
- The container will be interactive so you can verify each build step
- OpenCV is being built with GTK for GUI support and OpenGL enabled
