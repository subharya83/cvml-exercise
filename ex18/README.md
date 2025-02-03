# CUDA Vector Addition Example

This project demonstrates a simple vector addition implementation using CUDA, including compilation to PTX intermediate representation.

## Overview

The program performs element-wise addition of two vectors using CUDA parallel processing capabilities. It includes:
- Host memory allocation and initialization
- Device memory management
- Parallel vector addition kernel
- Result verification
- PTX code generation

## Requirements

- NVIDIA CUDA Toolkit (minimum version 11.0)
- NVIDIA GPU with compute capability 5.2 or higher
- Compatible NVIDIA driver
- gcc/g++ compiler
- Linux/Unix environment (for provided commands)

## File Structure

```
cuda-vector-addition/
├── src/
│   └── vector_add.cu
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuda-vector-addition.git
cd cuda-vector-addition
```

2. Verify CUDA installation:
```bash
nvcc --version
```

## Compilation

### Compile to Executable
```bash
nvcc src/vector_add.cu -o vector_add
```

### Generate PTX Code
```bash
nvcc -ptx src/vector_add.cu -o vector_add.ptx
```

## Usage

1. Run the executable:
```bash
./vector_add
```

2. View the generated PTX code:
```bash
cat vector_add.ptx
```

## Code Explanation

### CUDA Kernel
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n)
```
- `__global__`: Indicates this is a CUDA kernel function
- Performs parallel addition: `c[i] = a[i] + b[i]`
- Each thread handles one element of the vectors

### Memory Management
The program demonstrates proper CUDA memory handling:
- Host memory allocation using `malloc()`
- Device memory allocation using `cudaMalloc()`
- Memory transfers using `cudaMemcpy()`
- Proper cleanup using `cudaFree()` and `free()`

### Execution Configuration
```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```
- Uses 256 threads per block
- Grid size calculated to cover all elements
- Kernel launch with calculated dimensions

## Performance Considerations

- The example uses a fixed block size of 256 threads
- For optimal performance, consider:
  - Vector size alignment with thread block size
  - Memory coalescing
  - Shared memory usage for larger computations
  - Stream processing for concurrent operations

## Error Handling

The current implementation includes basic verification:
- Checks results against CPU computation
- Prints error message if verification fails
- Consider adding CUDA error checking macros for production use

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA CUDA Programming Guide
- CUDA By Example (Sanders, Kandrot)
