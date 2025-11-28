# CUDA 2D Convolution Implementation (Solution.cu)

This repository contains a high-performance implementation of a 2D convolution operation using CUDA C++. The solution is optimized for GPU architecture using specific techniques, as required by the assignment, to achieve efficient parallelism.

## Problem Statement & Data Format

The assignment requires implementing a 2D convolution of an input image with dimensions $H \times W \times C$ (Height, Width, Channels) with a filter set of $K$ filters, each having dimensions $R \times S \times C$.

### Data Transformation 

To simplify the process, the input image and filter set are supplied as stacked 2D matrices:

* **Input Image:** The $H \times W \times C$ image is stacked along the channel axis, resulting in a single 2D matrix of size $(H \times C) \times W$.
* **Filter Set:** The full set of $K$ filters (each $R \times S \times C$) is concatenated into a single 1D array of size $K \times R \times S \times C$.

### Convolution Output

The convolution must apply the filter's channels to the corresponding image channels and **sum the results over all channels** to produce one output value.

* The output for each filter is an $H \times W$ matrix.
* The final output for all $K$ filters is a single 1D array of size $H \times W \times K$.

---

## Parallel Strategy and Kernel Implementation

The solution utilizes a highly parallel approach with a **one-thread-per-output-element** mapping.

### 1. Thread Mapping

* **Total Threads:** The kernel is launched with a total of $H \times W \times K$ threads, matching the size of the final output matrix.
* **Thread Responsibility:** Each unique thread ID (`id`) is responsible for calculating one value in the final result:
    * `f_num = id / (H * W)`: Identifies the specific filter (0 to $K-1$).
    * `(x, y)`: Identifies the spatial coordinates (row and column) in the $H \times W$ output plane.

### 2. Alignment and Padding

* **Zero Padding:** Any cell required for the convolution operation that falls outside the matrix boundaries is assumed to be **0 (zero-padding)**. This is implemented via boundary checks (`if(x+j>=0 && x+j<h && y+l>=0 && y+l<w)`) inside the kernel loop.
* **Filter Alignment:** Filters are guaranteed to have **odd dimensions**. The kernel correctly aligns the center of the filter with the current input pixel, using offsets calculated as $\pm R/2$ and $\pm S/2$.
* **Stride:** The convolution operation uses a **stride length of 1**.

---

## Key GPU Optimizations (Mandatory)

The implementation relies on shared memory and coalescing to meet performance requirements.

### A. Dynamic Shared Memory for Filter Weights

The entire filter set is loaded into dynamically allocated **Shared Memory** (`extern __shared__ long int s_filter[]`) to enable fast, on-chip access by all threads in the block.

* **Filter Size Constraint:** The assignment specifies a constraint on the total filter size: $1\le K\times R\times S\times C\le4096$.
* **Cache Configuration:** Given that $4096 \times \text{sizeof(long int)} \approx 32\text{ KB}$, the kernel explicitly sets the cache configuration to prioritize shared memory:
    ```c
    cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferEqual); 
    ```
    This ensures an equal split (e.g., 32 KB) between L1 cache and Shared Memory, satisfying the filter storage requirement.

### B. Coalesced Global Memory Access

The operation to copy the filter from global memory (`filter`) to shared memory (`s_filter`) is structured to ensure **memory coalescing** during the load phase.

* **Coalescing Pattern:** Threads read consecutive memory locations, grouping multiple small memory requests into a single, efficient transaction:
    ```c
    for(int p=0; p < ceil((r*s*c*k*1.0)/1024.0); p++){
        if(1024*p+tid < r*s*c*k){
            s_filter[p*1024+tid] = filter[p*1024+tid];
        }
    }
    ```
* **Synchronization:** A barrier (`__syncthreads()`) ensures the entire filter is loaded into shared memory before any thread begins the computation loop.
