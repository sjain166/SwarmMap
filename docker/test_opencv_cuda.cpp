// Test program to verify OpenCV CUDA support
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "OpenCV CUDA Device Count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "CUDA is available!" << std::endl;

        for (int i = 0; i < cv::cuda::getCudaEnabledDeviceCount(); i++) {
            cv::cuda::setDevice(i);
            cv::cuda::DeviceInfo info(i);
            std::cout << "\nGPU " << i << " Information:" << std::endl;
            std::cout << "  Name: " << info.name() << std::endl;
            std::cout << "  Compute Capability: " << info.majorVersion() << "." << info.minorVersion() << std::endl;
            std::cout << "  Total Memory: " << info.totalMemory() / (1024*1024) << " MB" << std::endl;
        }

        std::cout << "\n✓ OpenCV CUDA support is working correctly!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ CUDA is NOT available in OpenCV!" << std::endl;
        return 1;
    }
}
