#include <opencv2/opencv.hpp>
#include <H5Cpp.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// CUDA kernel for integral image computation
__global__ void computeIntegralImage(const unsigned char* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Shared memory for block-wise computation
        extern __shared__ float sharedMem[];
        
        int idx = y * width + x;
        sharedMem[threadIdx.y * blockDim.x + threadIdx.x] = input[idx];
        __syncthreads();
        
        // Horizontal scan
        float sum = 0;
        for (int i = 0; i <= threadIdx.x; i++) {
            sum += sharedMem[threadIdx.y * blockDim.x + i];
        }
        
        // Store intermediate result
        sharedMem[threadIdx.y * blockDim.x + threadIdx.x] = sum;
        __syncthreads();
        
        // Vertical scan
        sum = 0;
        for (int i = 0; i <= threadIdx.y; i++) {
            sum += sharedMem[i * blockDim.x + threadIdx.x];
        }
        
        // Write final result
        output[idx] = sum;
    }
}

class VideoProcessor {
private:
    cv::VideoCapture cap;
    H5::H5File* file;
    dim_t dims[3];  // [frames, height, width]
    unsigned char* d_input;
    float* d_output;
    
public:
    VideoProcessor(const std::string& videoPath, const std::string& h5Path) {
        // Open video
        cap.open(videoPath);
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video file");
        }
        
        // Get video properties
        dims[0] = static_cast<dim_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        dims[1] = static_cast<dim_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        dims[2] = static_cast<dim_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        
        // Create HDF5 file
        file = new H5::H5File(h5Path, H5F_ACC_TRUNC);
        
        // Allocate CUDA memory
        size_t frameSize = dims[1] * dims[2];
        cudaMalloc(&d_input, frameSize * sizeof(unsigned char));
        cudaMalloc(&d_output, frameSize * sizeof(float));
    }
    
    ~VideoProcessor() {
        cap.release();
        delete file;
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    void process() {
        // Create dataset in HDF5 file
        H5::DataSpace dataspace(3, dims);
        H5::DataSet dataset = file->createDataSet("integral_images", 
                                                H5::PredType::NATIVE_FLOAT, 
                                                dataspace);
        
        // Setup CUDA execution parameters
        dim3 blockSize(16, 16);
        dim3 gridSize((dims[2] + blockSize.x - 1) / blockSize.x,
                     (dims[1] + blockSize.y - 1) / blockSize.y);
        size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);
        
        // Process frames
        cv::Mat frame, gray;
        std::vector<float> hostBuffer(dims[1] * dims[2]);
        
        for (dim_t i = 0; i < dims[0]; i++) {
            cap >> frame;
            if (frame.empty()) break;
            
            // Convert to grayscale
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            
            // Copy to GPU
            cudaMemcpy(d_input, gray.data, dims[1] * dims[2] * sizeof(unsigned char),
                      cudaMemcpyHostToDevice);
            
            // Compute integral image
            computeIntegralImage<<<gridSize, blockSize, sharedMemSize>>>
                (d_input, d_output, dims[2], dims[1]);
            
            // Copy result back
            cudaMemcpy(hostBuffer.data(), d_output, 
                      dims[1] * dims[2] * sizeof(float),
                      cudaMemcpyDeviceToHost);
            
            // Write to HDF5
            hsize_t start[3] = {i, 0, 0};
            hsize_t count[3] = {1, dims[1], dims[2]};
            H5::DataSpace memspace(3, count);
            dataspace.selectHyperslab(H5S_SELECT_SET, count, start);
            dataset.write(hostBuffer.data(), H5::PredType::NATIVE_FLOAT, 
                         memspace, dataspace);
            
            if (i % 100 == 0) {
                std::cout << "Processed " << i << " frames" << std::endl;
            }
        }
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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
