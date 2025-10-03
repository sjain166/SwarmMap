# SwarmMap Complete Build - CUDA 10.2 for Compilation Test
# WARNING: Will compile successfully but WILL FAIL at runtime on RTX A6000
# Purpose: Demonstrate the runtime incompatibility empirically
# Based on working configuration from reference Dockerfile
FROM fangruo/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install system dependencies (no ROS)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    gfortran \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    libqglviewer-dev-qt5 \
    libglew-dev \
    ffmpeg \
    libgtk-3-dev \
    python3 \
    python3-pip \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /opt

# Build Pangolin v0.5 (exact version from SwarmMap docs)
RUN git clone --depth 1 --branch v0.5 https://github.com/stevenlovegrove/Pangolin.git \
    && cd Pangolin \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Install additional dependencies for SwarmMap
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV 3.4.6 WITH CUDA 10.2 (works because CUDA 10.2 has nppicom library)
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.6.zip \
    && unzip opencv.zip \
    && cd opencv-3.4.6 \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D BUILD_opencv_python2=OFF \
             -D BUILD_opencv_python3=OFF \
             -D BUILD_PYTHON_SUPPORT=OFF \
             -D OPENCV_ENABLE_NONFREE=ON \
             -D WITH_CUDA=ON \
             -D WITH_CUDNN=OFF \
             -D CUDA_ARCH_BIN="6.0,6.1,7.0,7.5" \
             -D CUDA_ARCH_PTX="" \
             -D WITH_CUBLAS=ON \
             -D ENABLE_FAST_MATH=ON \
             -D CUDA_FAST_MATH=ON \
             -D BUILD_EXAMPLES=OFF \
             -D BUILD_TESTS=OFF \
             -D BUILD_PERF_TESTS=OFF \
             -D WITH_QT=OFF \
             -D WITH_GTK=ON \
             -D WITH_OPENGL=OFF \
             -D WITH_OPENCL=OFF .. \
    && make -j4 \
    && make install \
    && ldconfig \
    && cd /opt && rm -rf opencv*

# Build Boost 1.75.0 with Beast for WebSocket support (exact configuration from working build)
RUN wget https://sourceforge.net/projects/boost/files/boost/1.75.0/boost_1_75_0.tar.bz2/download -O boost_1_75_0.tar.bz2 \
    && tar -xjf boost_1_75_0.tar.bz2 \
    && cd boost_1_75_0 \
    && ./bootstrap.sh --prefix=/usr/local \
    && ./b2 --with-system --with-filesystem --with-chrono --with-thread --with-date_time --with-regex --with-serialization --with-program_options headers variant=release link=shared threading=multi -j4 install \
    && cd /opt && rm -rf boost_1_75_0*

# Build spdlog with position independent code (exact configuration from working build)
RUN git clone --depth 1 --branch v1.10.0 https://github.com/gabime/spdlog.git \
    && cd spdlog \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS="-fPIC" \
        -DCMAKE_C_FLAGS="-fPIC" \
        -DSPDLOG_BUILD_SHARED=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /opt && rm -rf spdlog

# Install CUDA helper headers for compatibility
RUN git clone --depth 1 --branch v11.8 https://github.com/NVIDIA/cuda-samples.git \
    && mkdir -p /usr/local/cuda/samples/common/inc \
    && cp cuda-samples/Common/*.h /usr/local/cuda/samples/common/inc/ \
    && cp cuda-samples/Common/*.cuh /usr/local/cuda/samples/common/inc/ 2>/dev/null || true \
    && rm -rf cuda-samples

# Copy SwarmMap source code
COPY . /opt/SwarmMap

# Build SwarmMap third-party dependencies
WORKDIR /opt/SwarmMap/code

# Build DBoW2
RUN cd Thirdparty/DBoW2 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Build g2o
RUN cd Thirdparty/g2o \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Verify CUDA samples installation
RUN echo "=== Verifying CUDA samples installation ===" \
    && ls -la /usr/local/cuda/samples/common/inc/ \
    && ls -la /usr/local/cuda/Common/ 2>/dev/null || echo "CUDA Common directory not found (expected)"

# Build SwarmMap core library (native executables only)
WORKDIR /opt/SwarmMap

RUN mkdir build && cd build \
    && echo "=== CMake Configuration ===" \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    && echo "=== Building SwarmMap Core ===" \
    && make -j$(nproc) \
    && echo "=== SwarmMap Build Complete ==="

# Create runtime directories
RUN mkdir -p /opt/SwarmMap/runtime/logs \
    && mkdir -p /opt/SwarmMap/runtime/maps \
    && mkdir -p /opt/SwarmMap/runtime/trajectories \
    && mkdir -p /opt/SwarmMap/runtime/data \
    && mkdir -p /opt/SwarmMap/scripts

# Verify SwarmMap build results (native executables)
RUN echo "=== FINAL SWARMMAP VERIFICATION ===" \
    && echo "Checking for SwarmMap executables..." \
    && ls -la /opt/SwarmMap/bin/ \
    && echo "Checking for core libraries..." \
    && ls -la /opt/SwarmMap/lib/ \
    && echo "=== SwarmMap build complete ===" \
    && echo "swarm_server: $(test -f /opt/SwarmMap/bin/swarm_server && echo 'FOUND' || echo 'MISSING')" \
    && echo "swarm_client: $(test -f /opt/SwarmMap/bin/swarm_client && echo 'FOUND' || echo 'MISSING')" \
    && echo "swarm_map: $(test -f /opt/SwarmMap/bin/swarm_map && echo 'FOUND' || echo 'MISSING')" \
    && echo "libslam_core.so: $(test -f /opt/SwarmMap/lib/libslam_core.so && echo 'FOUND' || echo 'MISSING')"

# Set environment variables
ENV SWARMMAP_ROOT=/opt/SwarmMap
ENV LD_LIBRARY_PATH=/opt/SwarmMap/lib:$LD_LIBRARY_PATH

WORKDIR /opt/SwarmMap

# Default command shows available executables
CMD ["ls", "-la", "/opt/SwarmMap/bin/"]
