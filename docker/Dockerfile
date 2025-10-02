# SwarmMap Complete Build - Ubuntu 18.04 + CUDA 11.8 + OpenCV 4.2.0
# Optimized for NVIDIA RTX A6000 (Ampere, Compute Capability 8.6)
FROM nvidia/cuda:11.8.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_ARCH_BIN="8.6"
ENV CUDA_ARCH_PTX="8.6"

# Install system dependencies
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
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    openexr \
    libopenexr-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    libqglviewer-dev-qt5 \
    libglew-dev \
    ffmpeg \
    python3-pip \
    netcat-openbsd \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Install CUDA samples headers (required by SwarmMap CUDA code)
RUN echo "Installing CUDA samples headers..." && \
    git clone --depth 1 --branch v11.8 https://github.com/NVIDIA/cuda-samples.git && \
    mkdir -p /usr/local/cuda/samples/common/inc && \
    cp cuda-samples/Common/*.h /usr/local/cuda/samples/common/inc/ && \
    cp cuda-samples/Common/*.cuh /usr/local/cuda/samples/common/inc/ 2>/dev/null || true && \
    rm -rf cuda-samples

# Build OpenCV 4.2.0 with CUDA 11.8 support
RUN echo "Building OpenCV 4.2.0 with CUDA..." && \
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip && \
    unzip -q opencv.zip && \
    rm opencv.zip && \
    cd opencv-4.2.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=OFF \
        -D CUDA_ARCH_BIN="8.6" \
        -D CUDA_ARCH_PTX="" \
        -D WITH_CUBLAS=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_TBB=ON \
        -D WITH_OPENGL=ON \
        -D WITH_QT=OFF \
        -D WITH_GTK=ON \
        -D WITH_OPENCL=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        .. && \
    make -j8 && \
    make install && \
    ldconfig && \
    cd /opt && rm -rf opencv-4.2.0

# Build Pangolin v0.5 (Ubuntu 18.04 has compatible ffmpeg 3.x)
RUN echo "Building Pangolin v0.5..." && \
    git clone --depth 1 --branch v0.5 https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j8 && \
    make install && \
    ldconfig && \
    cd /opt && rm -rf Pangolin

# Build Boost 1.75.0 with serialization support
RUN echo "Building Boost 1.75.0..." && \
    wget -q https://sourceforge.net/projects/boost/files/boost/1.75.0/boost_1_75_0.tar.bz2/download -O boost_1_75_0.tar.bz2 && \
    tar -xjf boost_1_75_0.tar.bz2 && \
    cd boost_1_75_0 && \
    ./bootstrap.sh --prefix=/usr/local && \
    ./b2 --with-system \
        --with-filesystem \
        --with-chrono \
        --with-thread \
        --with-date_time \
        --with-regex \
        --with-serialization \
        --with-program_options \
        headers \
        variant=release \
        link=shared \
        threading=multi \
        -j8 install && \
    ldconfig && \
    cd /opt && rm -rf boost_1_75_0*

# Build spdlog v1.10.0 with shared libraries
RUN echo "Building spdlog v1.10.0..." && \
    git clone --depth 1 --branch v1.10.0 https://github.com/gabime/spdlog.git && \
    cd spdlog && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS="-fPIC" \
        -DCMAKE_C_FLAGS="-fPIC" \
        -DSPDLOG_BUILD_SHARED=ON && \
    make -j8 && \
    make install && \
    ldconfig && \
    cd /opt && rm -rf spdlog

# Copy SwarmMap source code
COPY . /opt/SwarmMap

WORKDIR /opt/SwarmMap/code

# Build DBoW2 third-party library
RUN echo "Building DBoW2..." && \
    cd Thirdparty/DBoW2 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j8

# Build g2o third-party library
RUN echo "Building g2o..." && \
    cd Thirdparty/g2o && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j8

# Build SwarmMap main project
WORKDIR /opt/SwarmMap

RUN echo "Building SwarmMap..." && \
    mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 && \
    make -j8

# Create runtime directories
RUN mkdir -p /opt/SwarmMap/runtime/logs \
    && mkdir -p /opt/SwarmMap/runtime/maps \
    && mkdir -p /opt/SwarmMap/runtime/trajectories \
    && mkdir -p /opt/SwarmMap/runtime/data

# Verify build results
RUN echo "=== SwarmMap Build Verification ===" && \
    echo "Executables:" && \
    ls -lh /opt/SwarmMap/bin/ && \
    echo "" && \
    echo "Libraries:" && \
    ls -lh /opt/SwarmMap/lib/ && \
    echo "" && \
    echo "swarm_server: $(test -f /opt/SwarmMap/bin/swarm_server && echo 'FOUND' || echo 'MISSING')" && \
    echo "swarm_client: $(test -f /opt/SwarmMap/bin/swarm_client && echo 'FOUND' || echo 'MISSING')" && \
    echo "swarm_map: $(test -f /opt/SwarmMap/bin/swarm_map && echo 'FOUND' || echo 'MISSING')" && \
    echo "libslam_core.so: $(test -f /opt/SwarmMap/lib/libslam_core.so && echo 'FOUND' || echo 'MISSING')"

# Set environment variables
ENV SWARMMAP_ROOT=/opt/SwarmMap
ENV LD_LIBRARY_PATH=/opt/SwarmMap/lib:$LD_LIBRARY_PATH

WORKDIR /opt/SwarmMap

CMD ["/bin/bash"]
