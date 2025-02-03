#include "cuda_kernels.cuh"
#include <opencv2/opencv.hpp>
#include <H5Cpp.h>
#include <stdexcept>
#include <memory>

class VideoProcessor {
private:
    cv::VideoCapture cap;
    std::unique_ptr<H5::H5File> file; // Use std::unique_ptr
    std::unique_ptr<CUDAResources> cuda;
    size_t width, height;

    void processFrame(const cv::Mat& frame) {
        // Upload frame to GPU
        CHECK_CUDA(cudaMemcpyAsync(cuda->d_input, frame.data, 
                                 width * height * sizeof(unsigned char),
                                 cudaMemcpyHostToDevice, cuda->stream));

        // Setup execution parameters
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);

        // Convert to float
        cudaError_t err = launchConvertToFloat(
            cuda->d_input, cuda->d_temp, width, height, 
            gridSize, blockSize, cuda->stream
        );
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to launch convertToFloat kernel");
        }

        // Horizontal scan
        dim3 hBlockSize(256, 1);
        dim3 hGridSize(1, height);
        size_t sharedMemSize = width * sizeof(float);
        err = launchHorizontalScan(
            cuda->d_temp, cuda->d_integral, width, height,
            hGridSize, hBlockSize, sharedMemSize, cuda->stream
        );
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to launch horizontalScan kernel");
        }

        // Vertical scan
        dim3 vBlockSize(1, 256);
        dim3 vGridSize(width, 1);
        sharedMemSize = height * sizeof(float);
        err = launchVerticalScan(
            cuda->d_integral, cuda->d_temp, width, height,
            vGridSize, vBlockSize, sharedMemSize, cuda->stream
        );
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to launch verticalScan kernel");
        }

        // Synchronize stream
        CHECK_CUDA(cudaStreamSynchronize(cuda->stream));
    }

public:
    VideoProcessor(const std::string& videoPath, const std::string& h5Path) {
        try {
            // Open video
            cap.open(videoPath);
            if (!cap.isOpened()) {
                throw std::runtime_error("Failed to open video file: " + videoPath);
            }

            // Get video properties
            width = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            height = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

            // Initialize CUDA resources
            cuda = std::make_unique<CUDAResources>(width, height);

            // Create HDF5 file
            file = std::make_unique<H5::H5File>(h5Path, H5F_ACC_TRUNC); // Correct assignment
        } catch (const std::exception& e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            throw;
        }
    }

    void process() {
        // Create dataset
        hsize_t dims[3] = {0, height, width};  // Extensible dataset
        hsize_t maxdims[3] = {H5S_UNLIMITED, height, width};
        H5::DataSpace dataspace(3, dims, maxdims);

        // Enable chunking and compression
        H5::DSetCreatPropList plist;
        hsize_t chunk_dims[3] = {1, height, width};
        plist.setChunk(3, chunk_dims);
        plist.setDeflate(6);  // GZIP compression level 6

        H5::DataSet dataset = file->createDataSet(
            "integral_images", H5::PredType::NATIVE_FLOAT, dataspace, plist);

        // Process frames
        cv::Mat frame, gray;
        std::vector<float> hostBuffer(width * height);
        hsize_t frameCount = 0;

        while (cap.read(frame)) {
            // Convert to grayscale
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Process frame
            processFrame(gray);

            // Copy result back to host
            CHECK_CUDA(cudaMemcpyAsync(hostBuffer.data(), cuda->d_temp,
                                     width * height * sizeof(float),
                                     cudaMemcpyDeviceToHost, cuda->stream));

            // Extend dataset
            dims[0] = frameCount + 1;
            dataset.extend(dims);

            // Write frame to HDF5
            H5::DataSpace filespace = dataset.getSpace();
            hsize_t start[3] = {frameCount, 0, 0};
            hsize_t count[3] = {1, height, width};
            filespace.selectHyperslab(H5S_SELECT_SET, count, start);
            H5::DataSpace memspace(3, count);
            dataset.write(hostBuffer.data(), H5::PredType::NATIVE_FLOAT,
                         memspace, filespace);

            frameCount++;
            if (frameCount % 100 == 0) {
                std::cout << "Processed " << frameCount << " frames" << std::endl;
            }
        }
    }

    ~VideoProcessor() {
        cap.release();
        // No need to manually delete file; std::unique_ptr handles it
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <output_h5_path>" << std::endl;
        return 1;
    }

    try {
        VideoProcessor processor(argv[1], argv[2]);
        processor.process();
        std::cout << "Processing completed successfully" << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return 1;
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getCDetailMsg() << std::endl; // Use getCDetailMsg()
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}