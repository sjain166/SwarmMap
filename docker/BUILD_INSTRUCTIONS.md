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

## Step 2: Build OpenCV 4.2.0 with CUDA Support ✓ (GPU test passed)

This will take **20-40 minutes** depending on your CPU. OpenCV is being compiled with CUDA support for your RTX A6000 GPUs (compute capability 8.6).

### 2.1 Build the OpenCV image:

```powershell
cd "C:\Users\sj99\Desktop\SwarmMap\docker"
docker build -f Dockerfile.opencv -t swarmmap:opencv .
```

**What's happening**:
- Installing all OpenCV dependencies
- Downloading OpenCV 4.2.0 and opencv_contrib
- Compiling with CUDA support (compute capability 8.6 for RTX A6000)
- Using 8 parallel jobs for faster compilation

### 2.2 Test OpenCV CUDA support:

Once the build completes, start a container:

```powershell
docker run --rm --gpus all -it -v "${PWD}:/workspace" swarmmap:opencv
```

Inside the container, compile and run the test program:

```bash
cd /workspace
g++ -o test_opencv_cuda test_opencv_cuda.cpp `pkg-config --cflags --libs opencv4` -std=c++11
./test_opencv_cuda
```

**Expected output**:
```
OpenCV Version: 4.2.0
OpenCV CUDA Device Count: 2
CUDA is available!

GPU 0 Information:
  Name: NVIDIA RTX A6000
  Compute Capability: 8.6
  Total Memory: 49140 MB

GPU 1 Information:
  Name: NVIDIA RTX A6000
  Compute Capability: 8.6
  Total Memory: 49140 MB

✓ OpenCV CUDA support is working correctly!
```

**Report back**:
1. Did the Docker build complete successfully?
2. Does the test program show CUDA is available with both GPUs?

If yes, we'll move to Step 3: Building Pangolin for visualization.

---

## Notes:
- We're using CUDA 11.8 (compatible with your CUDA 12.6 driver and RTX A6000 Ampere architecture)
- Each step will be tested before moving to the next
- The container will be interactive so you can verify each build step
- OpenCV is being built with GTK for GUI support and OpenGL enabled
