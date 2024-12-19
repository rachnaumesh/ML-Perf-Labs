# ML Performance Optimization Labs

## Overview
This repository contains the code and lab work from a semester-long project focused on performance analysis and optimization techniques for machine learning workloads, all implemented in **C** and run on CPUs.

The project involves building a foundational machine learning library and systematically optimizing it through several techniques, culminating in a GPT-2 implementation for CPU.

## Project Highlights
1. **Initial Implementation**  
   - Developed a foundational machine learning library in **C** for the forward pass of a small CNN.

2. **Performance Analysis**  
   - Conducted bottleneck analysis and profiled code performance.

3. **Optimizations Applied**  
   - **Tiling and Blocking**  
   - **Sparse Matrix Multiplication**  
   - **Multithreading** using **pthreads** and **OpenMP**

4. **Final Model**  
   - Implemented GPT-2 optimized to run efficiently on CPU.

## Repository Structure
- **Branches**  
  Each branch corresponds to a specific stage of the project:  
  - `base`: Initial CNN forward-pass implementation  
  - `tiling-blocking`: Optimized with tiling and blocking  
  - `sparse-matrix-mul`: Sparse matrix multiplication  
  - `multithreading`: Multithreaded implementation  
  - `gpt2-cpu`: GPT-2 final optimized implementation

## Usage Instructions

### Cloning the Repo
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

## Switching Branches
To check out a specific stage:
```bash
git checkout <branch-name>
```

## Building and Running
All projects include a `Makefile`.  
To build:
```bash
make
```
To run:
```bash
./<executable-name>
```

## Tools and Techniques

### Profiling Tools
- gprof
- perf
- toplev from pmu-tools

### Programming Techniques
- Blocking
- Tiling
- Multithreading

### Libraries/Frameworks
- pthreads
- OpenMP
