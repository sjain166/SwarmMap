#!/bin/bash
################################################################################
# Build OpenCV 4.5.5 with CUDA for Jetson Xavier
# JetPack 5.1.2 (L4T R35.4.1), Ubuntu 20.04, CUDA 11.4
#
# This script builds OpenCV with:
# - CUDA 11.4 support (GPU-accelerated)
# - opencv_contrib modules (includes cudafilters, cudafeatures2d, etc.)
# - cuDNN support
# - Python bindings (optional)
# - Optimized for Jetson Xavier (SM 7.2)
#
# Estimated time: 1.5-2 hours
# Disk space required: ~10GB during build
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Configuration
OPENCV_VERSION="4.5.5"
BUILD_DIR="/mnt/nvme/downloads/opencv_build"
INSTALL_PREFIX="/usr/local"
NUM_CORES=$(nproc)

# Jetson Xavier specific
CUDA_ARCH_BIN="7.2"
CUDA_ARCH_PTX="7.2"

log_info "OpenCV Version: ${OPENCV_VERSION}"
log_info "Build Directory: ${BUILD_DIR}"
log_info "Install Prefix: ${INSTALL_PREFIX}"
log_info "CPU Cores: ${NUM_CORES}"
log_info "CUDA Architecture: ${CUDA_ARCH_BIN}"

# Set temp directory to SSD
export TMPDIR="${BUILD_DIR}/tmp"
mkdir -p "${TMPDIR}"
log_info "Using temp directory: ${TMPDIR}"

################################################################################
# Step 1: Check Prerequisites
################################################################################
log_step "Step 1: Checking Prerequisites"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    log_error "CUDA not found! Please install JetPack first."
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
log_success "CUDA ${CUDA_VERSION} found"

# Check disk space
AVAILABLE_SPACE=$(df -BG "${BUILD_DIR%/*}" | tail -1 | awk '{print $4}' | sed 's/G//')
log_info "Available disk space: ${AVAILABLE_SPACE}GB"
if [ "$AVAILABLE_SPACE" -lt 15 ]; then
    log_error "Need at least 15GB free space. Available: ${AVAILABLE_SPACE}GB"
    exit 1
fi

################################################################################
# Step 2: Install Build Dependencies
################################################################################
log_step "Step 2: Installing Build Dependencies"

log_info "Installing build tools and libraries..."
sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    yasm \
    checkinstall \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev \
    x264 \
    libx264-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libfaac-dev \
    libmp3lame-dev \
    libvorbis-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libgtk-3-dev \
    libtbb-dev \
    libatlas-base-dev \
    gfortran \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    python3-dev \
    python3-pip \
    python3-numpy

log_success "Dependencies installed"

################################################################################
# Step 3: Create Build Directory
################################################################################
log_step "Step 3: Setting Up Build Environment"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

log_info "Cleaning previous builds..."
rm -rf opencv opencv_contrib

################################################################################
# Step 4: Download OpenCV and opencv_contrib
################################################################################
log_step "Step 4: Downloading OpenCV ${OPENCV_VERSION}"

log_info "Downloading OpenCV ${OPENCV_VERSION}..."
wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
log_success "OpenCV downloaded"

log_info "Downloading opencv_contrib ${OPENCV_VERSION}..."
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
log_success "opencv_contrib downloaded"

log_info "Extracting archives..."
unzip -q opencv.zip
unzip -q opencv_contrib.zip

mv opencv-${OPENCV_VERSION} opencv
mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

log_success "Source code extracted"

# Clean up zip files
rm opencv.zip opencv_contrib.zip

################################################################################
# Step 5: Configure CMake
################################################################################
log_step "Step 5: Configuring CMake"

cd opencv
mkdir -p build
cd build

log_info "Running CMake configuration (this may take a few minutes)..."

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -D OPENCV_EXTRA_MODULES_PATH=${BUILD_DIR}/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    -D CUDA_ARCH_PTX=${CUDA_ARCH_PTX} \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr" \
    -D WITH_TBB=ON \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
    -D PYTHON3_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.8.so \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    ..

log_success "CMake configuration complete"

################################################################################
# Step 6: Check Configuration
################################################################################
log_step "Step 6: Verifying Configuration"

log_info "Checking CUDA configuration..."
if grep -q "WITH_CUDA:BOOL=ON" CMakeCache.txt; then
    log_success "CUDA support enabled ✅"

    # Check CUDA architecture
    CUDA_ARCH=$(grep "CUDA_ARCH_BIN:STRING" CMakeCache.txt | cut -d'=' -f2)
    log_info "CUDA Architecture: ${CUDA_ARCH}"

    # Check cuDNN
    if grep -q "WITH_CUDNN:BOOL=ON" CMakeCache.txt; then
        log_success "cuDNN support enabled ✅"
    fi
else
    log_error "CUDA support NOT enabled!"
    log_error "Check CMake configuration above for errors"
    exit 1
fi

log_info "Checking opencv_contrib modules..."
if grep -q "opencv_contrib" CMakeCache.txt; then
    log_success "opencv_contrib modules found ✅"
else
    log_warning "opencv_contrib may not be properly configured"
fi

log_info "Checking for CUDA modules..."
if grep -q "BUILD_opencv_cudafilters:BOOL=ON" CMakeCache.txt && \
   grep -q "BUILD_opencv_cudafeatures2d:BOOL=ON" CMakeCache.txt; then
    log_success "CUDA modules (cudafilters, cudafeatures2d, etc.) will be built ✅"

    # Count how many CUDA modules are enabled
    CUDA_MODULES=$(grep "BUILD_opencv_cuda.*:BOOL=ON" CMakeCache.txt | wc -l)
    log_info "Total CUDA modules enabled: ${CUDA_MODULES}"
else
    log_error "CUDA modules not properly configured!"
    log_error "Check opencv_contrib path"
    exit 1
fi

################################################################################
# Step 7: Build OpenCV
################################################################################
log_step "Step 7: Building OpenCV (This will take 1-2 hours)"

log_info "Building with ${NUM_CORES} cores..."
log_warning "Go get some coffee ☕ - this will take a while..."

START_TIME=$(date +%s)

make -j${NUM_CORES}

END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))
BUILD_MINUTES=$((BUILD_TIME / 60))

log_success "Build completed in ${BUILD_MINUTES} minutes"

################################################################################
# Step 8: Install OpenCV
################################################################################
log_step "Step 8: Installing OpenCV"

log_info "Installing OpenCV to ${INSTALL_PREFIX}..."
sudo make install
sudo ldconfig

log_success "OpenCV installed"

################################################################################
# Step 9: Verify Installation
################################################################################
log_step "Step 9: Verifying Installation"

log_info "Checking installed files..."
if [ -f "${INSTALL_PREFIX}/lib/libopencv_core.so" ]; then
    log_success "Core library found ✅"
else
    log_error "Core library NOT found!"
    exit 1
fi

if [ -f "${INSTALL_PREFIX}/lib/libopencv_cudafilters.so" ]; then
    log_success "CUDA filters library found ✅"
else
    log_error "CUDA filters library NOT found!"
    log_error "Check cmake configuration"
    exit 1
fi

log_info "Testing Python bindings..."
if python3 -c "import cv2; print(f'OpenCV {cv2.__version__}'); print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')" 2>/dev/null; then
    log_success "Python bindings work ✅"
else
    log_warning "Python bindings may have issues"
fi

################################################################################
# Step 10: Create Environment Script
################################################################################
log_step "Step 10: Creating Environment Script"

ENV_FILE="${BUILD_DIR}/opencv_env.sh"
cat > "${ENV_FILE}" << 'EOF'
#!/bin/bash
# OpenCV Environment Setup

export OpenCV_DIR=/usr/local/lib/cmake/opencv4
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

echo "OpenCV environment configured"
echo "OpenCV_DIR=${OpenCV_DIR}"
EOF

chmod +x "${ENV_FILE}"
log_success "Environment script created: ${ENV_FILE}"

################################################################################
# Step 11: Cleanup (Optional)
################################################################################
log_step "Step 11: Cleanup"

log_warning "Build directory at ${BUILD_DIR} uses ~10GB"
read -p "Remove build directory to save space? (y/N): " cleanup
if [[ "$cleanup" =~ ^[Yy]$ ]]; then
    log_info "Removing build directory..."
    cd /
    rm -rf "${BUILD_DIR}/opencv"
    log_success "Build directory removed (kept source downloads)"
else
    log_info "Build directory kept at ${BUILD_DIR}"
fi

################################################################################
# Final Summary
################################################################################
log_step "Build Complete! 🎉"

echo ""
log_success "✅ OpenCV ${OPENCV_VERSION} with CUDA successfully built!"
echo ""
log_info "Installation Summary:"
echo "  - OpenCV version: ${OPENCV_VERSION}"
echo "  - Install location: ${INSTALL_PREFIX}"
echo "  - CUDA support: YES (SM ${CUDA_ARCH_BIN})"
echo "  - CUDA modules: cudafilters, cudafeatures2d, etc."
echo "  - Build time: ${BUILD_MINUTES} minutes"
echo ""
log_info "To use OpenCV in your projects:"
echo "  source ${ENV_FILE}"
echo ""
log_info "To verify installation:"
echo "  pkg-config --modversion opencv4"
echo "  python3 -c 'import cv2; print(cv2.__version__)'"
echo ""
log_success "Ready to build SwarmMap! 🚀"
