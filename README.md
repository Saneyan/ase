# ASE Coding

Parallelize stream-based lossless data compression using GPU with Adaptive Stream-based Entropy Coding (ASE Coding) algorithm.
This project is the reference implementation of ASE Coding. Sample programs run on CPU and CUDA-based GPU.

### Requirements 

* Linux
* CUDA: `>= 3`
* CMake: `>= 3.10`
* C++ Compiler support for C++14

### Build

Create `build` directory and compile with CMake and make.
Execute a binary file named `app` in the build directory after compiling.

```bash
$ mkdir build/; cd build/
$ cmake ..
$ make
$ ./app
```

### Project Structure

* `app/`: Programs that compress/decompress files with ASE Coding library
* `lib/`: ASE Coding library
* `old/`: Unused files

### Version

#### v0.2.0
* Created the compressor and decompressor running on GPU

#### v0.1.0
* Introduced build system CMake

#### v0.0.1
* Create the compressor and decompressor running on CPU
