#!/bin/bash
################################################################################
# SwarmMap Native Build Script for Jetson Xavier
# JetPack 5.1.2 (L4T R35.4.1), Ubuntu 20.04, CUDA 11.4
#
# Dependencies (from README.md and Dependencies.md):
# - Pangolin v0.5 (using v0.6 for GCC 9+ compatibility)
# - OpenCV 3.4.6/4.2.0 with CUDA (using system OpenCV from JetPack)
# - Eigen3
# - CUDA (using CUDA 11.4 from JetPack, README tested with 10.2)
# - Boost >= 1.70.0 (Ubuntu 20.04 has 1.71)
# - spdlog v1.10.0
# - DBoW2 (bundled, will be patched for OpenCV 4.x)
# - g2o (bundled)
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

# Configuration
SWARMMAP_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SWARMMAP_ROOT}/build_native"
INSTALL_PREFIX="/usr/local"
NUM_CORES=$(nproc)

log_info "SwarmMap Root: ${SWARMMAP_ROOT}"
log_info "Build Directory: ${BUILD_DIR}"
log_info "CPU Cores: ${NUM_CORES}"

# Create build directory
mkdir -p "${BUILD_DIR}"

################################################################################
# Step 1: Check System Information
################################################################################
log_step "Step 1: System Information Check"

log_info "Checking JetPack version..."
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    log_warning "Cannot find /etc/nv_tegra_release"
fi

log_info "Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    log_success "CUDA ${CUDA_VERSION} found"
else
    log_error "nvcc not found! Please install CUDA/JetPack."
    exit 1
fi

log_info "Checking Ubuntu version..."
lsb_release -d

log_info "Checking architecture..."
ARCH=$(uname -m)
log_info "Architecture: ${ARCH}"
if [ "${ARCH}" != "aarch64" ]; then
    log_error "This script is for ARM64 (aarch64) architecture only!"
    exit 1
fi

################################################################################
# Step 2: Check Existing Dependencies
################################################################################
log_step "Step 2: Checking Existing Dependencies"

# Check OpenCV (SwarmMap requires OpenCV with CUDA support)
log_info "Checking OpenCV..."
OPENCV_FOUND=false
OPENCV_CUDA_ENABLED=false

if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    log_success "OpenCV ${OPENCV_VERSION} found"
    OPENCV_FOUND=true

    # Check CUDA support
    if python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null | grep -q "^[1-9]"; then
        log_success "OpenCV has CUDA support enabled ✅"
        OPENCV_CUDA_ENABLED=true
    else
        log_warning "OpenCV found but CUDA support is NOT enabled ⚠️"
        log_warning "SwarmMap requires OpenCV with CUDA for best performance"
        log_warning "You may need to build OpenCV from source with CUDA enabled"
    fi
elif pkg-config --exists opencv; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    log_warning "Found OpenCV ${OPENCV_VERSION} (v3.x), but SwarmMap prefers OpenCV 4.x"
    OPENCV_FOUND=true
else
    log_error "OpenCV not found!"
    log_error "JetPack should include OpenCV. Please check your installation."
    log_error "You may need to install: sudo apt install libopencv-dev python3-opencv"
fi

# Check if OpenCV is good enough to proceed
if [ "$OPENCV_FOUND" = false ]; then
    log_error "Cannot proceed without OpenCV. Exiting."
    exit 1
fi

if [ "$OPENCV_CUDA_ENABLED" = false ]; then
    log_warning "====================================================================="
    log_warning "OpenCV without CUDA will result in SIGNIFICANTLY slower performance"
    log_warning "SwarmMap will fall back to CPU-only ORB extraction"
    log_warning "====================================================================="
    read -p "Continue anyway? (y/N): " continue_without_cuda
    if [[ ! "$continue_without_cuda" =~ ^[Yy]$ ]]; then
        log_info "Exiting. Please install OpenCV with CUDA support first."
        exit 1
    fi
fi

# Check Eigen3
if pkg-config --exists eigen3; then
    EIGEN_VERSION=$(pkg-config --modversion eigen3)
    log_success "Eigen3 ${EIGEN_VERSION} found"
else
    log_warning "Eigen3 not found"
fi

# Check Boost (SwarmMap requires >= 1.70.0 for Beast/serialization)
if dpkg -l | grep -q libboost-dev; then
    BOOST_VERSION=$(dpkg -l | grep "libboost-dev" | awk '{print $3}' | cut -d'.' -f1-2)
    log_success "Boost libraries found (version ${BOOST_VERSION})"

    # Check if version is >= 1.70
    BOOST_MAJOR=$(echo $BOOST_VERSION | cut -d'.' -f1)
    BOOST_MINOR=$(echo $BOOST_VERSION | cut -d'.' -f2)

    if [ "$BOOST_MAJOR" -eq 1 ] && [ "$BOOST_MINOR" -lt 70 ]; then
        log_warning "Boost version ${BOOST_VERSION} is less than required 1.70.0"
        log_warning "You may encounter issues with Boost Beast"
    else
        log_success "Boost version ${BOOST_VERSION} meets requirement (>= 1.70.0)"
    fi
else
    log_warning "Boost not found - will be installed via apt"
fi

################################################################################
# Step 3: Install System Dependencies
################################################################################
log_step "Step 3: Installing System Dependencies"

log_info "Updating package lists..."
sudo apt-get update

log_info "Installing build tools and libraries..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libhdf5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libglew-dev \
    qtdeclarative5-dev \
    libqglviewer-dev-qt5 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk-3-dev \
    python3-dev \
    python3-numpy

log_success "System dependencies installed"

################################################################################
# Step 4: Build Pangolin v0.6
################################################################################
log_step "Step 4: Building Pangolin v0.6"

PANGOLIN_DIR="${BUILD_DIR}/Pangolin"

if [ -f "${INSTALL_PREFIX}/lib/libpangolin.so" ]; then
    log_warning "Pangolin already installed at ${INSTALL_PREFIX}/lib/libpangolin.so"
    read -p "Rebuild Pangolin? (y/N): " rebuild_pangolin
    if [[ ! "$rebuild_pangolin" =~ ^[Yy]$ ]]; then
        log_info "Skipping Pangolin build"
    else
        rebuild_pangolin="y"
    fi
else
    rebuild_pangolin="y"
fi

if [[ "$rebuild_pangolin" == "y" ]]; then
    log_info "Cloning Pangolin v0.6..."
    cd "${BUILD_DIR}"
    if [ -d "Pangolin" ]; then
        rm -rf Pangolin
    fi
    git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git

    cd Pangolin
    mkdir -p build && cd build

    log_info "Configuring Pangolin..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-declarations"

    log_info "Building Pangolin with ${NUM_CORES} cores..."
    make -j${NUM_CORES}

    log_info "Installing Pangolin..."
    sudo make install
    sudo ldconfig

    log_success "Pangolin v0.6 built and installed"
else
    log_info "Using existing Pangolin installation"
fi

################################################################################
# Step 5: Build spdlog (Logging Library)
################################################################################
log_step "Step 5: Building spdlog v1.10.0"

SPDLOG_DIR="${BUILD_DIR}/spdlog"

if [ -f "${INSTALL_PREFIX}/lib/libspdlog.so" ]; then
    log_warning "spdlog already installed"
    read -p "Rebuild spdlog? (y/N): " rebuild_spdlog
    if [[ ! "$rebuild_spdlog" =~ ^[Yy]$ ]]; then
        log_info "Skipping spdlog build"
    else
        rebuild_spdlog="y"
    fi
else
    rebuild_spdlog="y"
fi

if [[ "$rebuild_spdlog" == "y" ]]; then
    log_info "Cloning spdlog v1.10.0..."
    cd "${BUILD_DIR}"
    if [ -d "spdlog" ]; then
        rm -rf spdlog
    fi
    git clone --depth 1 --branch v1.10.0 https://github.com/gabime/spdlog.git

    cd spdlog
    mkdir -p build && cd build

    log_info "Configuring spdlog..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DSPDLOG_BUILD_SHARED=ON

    log_info "Building spdlog..."
    make -j${NUM_CORES}

    log_info "Installing spdlog..."
    sudo make install
    sudo ldconfig

    log_success "spdlog v1.10.0 built and installed"
fi

################################################################################
# Step 6: Build DBoW2 (with OpenCV 4 compatibility patches)
################################################################################
log_step "Step 6: Building DBoW2"

DBOW2_DIR="${SWARMMAP_ROOT}/code/Thirdparty/DBoW2"

if [ ! -d "${DBOW2_DIR}" ]; then
    log_error "DBoW2 directory not found at ${DBOW2_DIR}"
    log_error "Make sure you're running this from SwarmMap root directory"
    exit 1
fi

log_info "Patching DBoW2 for OpenCV 4 compatibility..."
cd "${DBOW2_DIR}"

# Patch OpenCV 3 headers to OpenCV 4 style
sed -i 's|opencv2/core/core.hpp|opencv2/core.hpp|g' DBoW2/FORB.h
sed -i 's|opencv2/core/core.hpp|opencv2/core.hpp|g' DBoW2/TemplatedVocabulary.h
find . -name "*.h" -o -name "*.cpp" | xargs sed -i 's|opencv2/highgui/highgui.hpp|opencv2/highgui.hpp|g'
find . -name "*.h" -o -name "*.cpp" | xargs sed -i 's|opencv2/features2d/features2d.hpp|opencv2/features2d.hpp|g'

log_success "DBoW2 patched for OpenCV 4"

log_info "Building DBoW2..."
rm -rf build
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4

make -j${NUM_CORES}

log_success "DBoW2 built successfully"

################################################################################
# Step 7: Build g2o
################################################################################
log_step "Step 7: Building g2o"

G2O_DIR="${SWARMMAP_ROOT}/code/Thirdparty/g2o"

if [ ! -d "${G2O_DIR}" ]; then
    log_error "g2o directory not found at ${G2O_DIR}"
    exit 1
fi

log_info "Building g2o..."
cd "${G2O_DIR}"
rm -rf build
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release

make -j${NUM_CORES}

log_success "g2o built successfully"

################################################################################
# Step 8: Build SwarmMap Core
################################################################################
log_step "Step 8: Building SwarmMap Core"

log_info "Building SwarmMap with CUDA support..."
cd "${SWARMMAP_ROOT}"
rm -rf build
mkdir build && cd build

log_info "Configuring SwarmMap..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
    -DCUDA_ARCH_BIN="7.2" \
    -DCMAKE_CUDA_ARCHITECTURES="72" \
    -DCUDA_NVCC_FLAGS="-gencode arch=compute_72,code=sm_72 -gencode arch=compute_72,code=compute_72"

log_info "Building SwarmMap with ${NUM_CORES} cores..."
make -j${NUM_CORES}

################################################################################
# Step 9: Verify Build
################################################################################
log_step "Step 9: Build Verification"

log_info "Checking for executables..."
if [ -f "${SWARMMAP_ROOT}/bin/swarm_server" ]; then
    log_success "swarm_server: FOUND"
else
    log_error "swarm_server: MISSING"
fi

if [ -f "${SWARMMAP_ROOT}/bin/swarm_client" ]; then
    log_success "swarm_client: FOUND"
else
    log_error "swarm_client: MISSING"
fi

if [ -f "${SWARMMAP_ROOT}/bin/swarm_map" ]; then
    log_success "swarm_map: FOUND"
else
    log_error "swarm_map: MISSING"
fi

log_info "Checking for libraries..."
if [ -f "${SWARMMAP_ROOT}/lib/libslam_core.so" ]; then
    log_success "libslam_core.so: FOUND"
else
    log_error "libslam_core.so: MISSING"
fi

################################################################################
# Step 10: Setup Environment
################################################################################
log_step "Step 10: Environment Setup"

log_info "Setting up environment variables..."

ENV_FILE="${SWARMMAP_ROOT}/setup_env.sh"
cat > "${ENV_FILE}" << 'EOF'
#!/bin/bash
# SwarmMap Environment Setup

export SWARMMAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SWARMMAP_ROOT}/lib:${LD_LIBRARY_PATH}"
export PATH="${SWARMMAP_ROOT}/bin:${PATH}"

echo "SwarmMap environment configured"
echo "SWARMMAP_ROOT=${SWARMMAP_ROOT}"
EOF

chmod +x "${ENV_FILE}"

log_success "Environment setup script created: ${ENV_FILE}"
log_info "To use SwarmMap, run: source ${ENV_FILE}"

################################################################################
# Final Summary
################################################################################
log_step "Build Complete!"

echo ""
log_success "✅ SwarmMap built successfully on Jetson Xavier!"
echo ""
log_info "Next steps:"
echo "  1. Source environment: source ${SWARMMAP_ROOT}/setup_env.sh"
echo "  2. Download EuRoC dataset"
echo "  3. Run test: ./bin/swarm_client -v code/Vocabulary/ORBvoc.bin -d config/test_mh01.yaml"
echo ""
log_info "Executables location: ${SWARMMAP_ROOT}/bin/"
log_info "  - swarm_server  (server mode)"
log_info "  - swarm_client  (client mode)"
log_info "  - swarm_map     (combined mode)"
echo ""
log_success "Happy SLAMming! 🚀"
