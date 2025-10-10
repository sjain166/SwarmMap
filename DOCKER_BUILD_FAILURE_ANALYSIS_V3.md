# SwarmMap Docker Build and Runtime Analysis: RTX A6000 Incompatibility

**Date:** October 2-3, 2025
**Target Hardware:** 2x NVIDIA RTX A6000 GPUs (Ampere, Compute Capability 8.6)
**Build Environment:** Docker Desktop on Windows with WSL2
**Final Status:** ❌ **BUILD SUCCEEDS, RUNTIME FAILS - CUDA Constant Memory Incompatibility**

---

## Executive Summary

**Build Status:** ✅ Docker image builds successfully with CUDA 10.2 + OpenCV 3.4.6 + SwarmMap

**Runtime Status:** ❌ SwarmMap crashes at runtime on RTX A6000 with `cudaErrorInvalidSymbol (code=13)`

**Root Cause (Empirically Confirmed via Extensive Testing):**

CUDA **constant memory** operations (`cudaMemcpyToSymbol`) require exact GPU architecture match. SwarmMap's CUDA kernels use constant memory extensively in `Fast_gpu.cu` and `Orb_gpu.cu`. CUDA 10.2 supports maximum compute capability 7.5 (Turing), but RTX A6000 requires 8.6 (Ampere). Constant memory symbol tables are architecture-specific and cannot be remapped at runtime, even with NVIDIA driver backward compatibility.

**Issues Identified and Resolved:**
1. ✅ Configuration parsing (`stoi()` error) - Fixed by adding quotes to PORT values in YAML
2. ✅ Image loading failure - Fixed by converting Windows line endings (CRLF→LF) in timestamp files
3. ❌ **CUDA constant memory incompatibility - CANNOT BE FIXED without CUDA 11.1+ or older GPU**

**Compilation Attempts (All Failed at Runtime):**
- Default build (implicit sm_30): `cudaErrorInvalidSymbol`
- Explicit sm_75/sm_70 flags: `cudaErrorInvalidSymbol`
- Attempted sm_86: Build fails (CUDA 10.2 doesn't support Ampere)

**Conclusion:** RTX A6000 fundamentally incompatible with CUDA 10.2 + SwarmMap due to constant memory requirements.

---

## Quick Navigation

- [Empirical Testing Results](#empirical-runtime-investigation-october-3-2025) - What we actually found through hands-on testing
- [Root Cause: Constant Memory](#root-cause-analysis-constant-memory-and-gpu-architecture) - Why it fails
- [Solutions](#viable-solutions) - What can be done about it

---

