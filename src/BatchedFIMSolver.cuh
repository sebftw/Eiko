/*
 * Copyright (c) 2026, Sebastian Kazmarek Præsius. All rights reserved.
 * Licensed under the BSD 3-Clause License. See LICENSE file in the project root for details.
 */

#ifndef BATCHEDFIMSOLVER_H
#define BATCHEDFIMSOLVER_H

#include "eiko_kernels.cuh"  // Defines CUDA kernels.

// To be able to get which GPU architecture this code was compiled for:
__device__ __constant__ int COMPILED_ARCH = 
#if defined(__CUDA_ARCH__)
    __CUDA_ARCH__;
#else
    0;
#endif


/**
 * @brief Executes a batched, multi-channel Fast Iterative Method (FIM) solver on the GPU.
 *
 * This class persists device memory and instantiated CUDA graphs across multiple executions,
 * eliminating CPU overhead and memory allocation latency for iterative workloads.
 *
 * @tparam CHANNELS   The number of independent sources/channels per grid element.
 */
template <typename Config>
class BatchedFIMSolver {
private:
    // ---------------------------------------------------------
    // CONFIGURATION
    // ---------------------------------------------------------
    int fixed_grid_size;
    bool use_cond_graph;
	bool use_cuda_graph;
	
	// Capacity tracking.
    int max_total_blocks = 0;
	size_t current_u_bytes = 0;
	
	// Alias constants from the traits struct for ease of use.
    static constexpr int HALO = FIMConstants<Config>::HALO;
    static constexpr int EFF_W = FIMConstants<Config>::EFF_W;
    static constexpr int EFF_H = FIMConstants<Config>::EFF_H;
    static constexpr int EFF_D = FIMConstants<Config>::EFF_D;
    static constexpr int THREADS_PER_BLOCK = FIMConstants<Config>::THREADS_PER_BLOCK;
	
    // ---------------------------------------------------------
    // DEVICE MEMORY POINTERS
    // ---------------------------------------------------------
    int *d_current_active_block_list = nullptr;
    int *d_next_active_block_list = nullptr;
    int *d_is_enqueued = nullptr;
    int *d_next_num_active_blocks = nullptr;
    int *d_num_active_blocks = nullptr;
    int *d_jobs_claimed = nullptr;
    int **d_list_ptrs = nullptr;
	int *d_iteration = nullptr;
	unsigned long long *d_start_time = nullptr;
	float *d_sign_mode = nullptr;
	void* d_u_temp = nullptr;

    // ---------------------------------------------------------
    // GRAPH HANDLES & PARAMETERS
    // ---------------------------------------------------------
    cudaGraph_t mainGraph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaGraphConditionalHandle handle = 0;
    cudaGraphNode_t msNode1;  // Clears d_next_num_active_blocks.
    cudaGraphNode_t msNode2;  // Clears d_is_enqueued.
    cudaGraphNode_t fimNode;  // Runs FIM algorithm.
    cudaGraphNode_t mgtNode;  // Prepares for next FIM iteration.
    cudaGraphNode_t condNode; // While loop.
	
	cudaStream_t active_stream = 0;
	cudaEvent_t launch_done = nullptr;
	void* fim_kernel_ptr;
	
	/**
     * @brief (Re)allocates memory on the GPU.
     */
	void ensure_capacity(int req_total_blocks) {
        if (req_total_blocks <= max_total_blocks) return; // We have enough room!

		if(max_total_blocks == 0)
			// First run: Grow to match exact input size.
			max_total_blocks = req_total_blocks;
		else
			// Subsequent runs: Grow by 1.5x to amortize reallocation costs (standard vector behavior)
			max_total_blocks = static_cast<int>(req_total_blocks * 1.5); 

        // Free old memory (safe to pass nullptr on the very first allocation)
		CUDA_CHECK(cudaFree(d_current_active_block_list));
        CUDA_CHECK(cudaFree(d_next_active_block_list));
        CUDA_CHECK(cudaFree(d_is_enqueued));

        // Reallocate with new max capacity
        CUDA_CHECK(cudaMalloc(&d_current_active_block_list, max_total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_next_active_block_list, max_total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_is_enqueued, max_total_blocks * sizeof(int)));
    }
	
public:
    /**
     * @brief Constructor allocates all device memory and instantiates the graph topology.
     */
    BatchedFIMSolver() {
        // Hardware Query.
        int device_id;
        CUDA_CHECK(cudaGetDevice(&device_id));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
		
		// Enable for disable CUDA Graph acceleration (Hardware check).
        use_cuda_graph = (prop.major >= 7);
        use_cond_graph =  (prop.major >= 9); // Hopper+

        // Software check.
        #if !CUDA_HAS_CONDITIONAL_GRAPHS
            use_cond_graph = false; // Explicitly disabled on older toolkits.
        #endif
        int compiled_arch = 0;  // Read the architecture the code was compiled for
        CUDA_CHECK(cudaMemcpyFromSymbol(&compiled_arch, COMPILED_ARCH, sizeof(int)));
        use_cond_graph &= (compiled_arch >= 900);

		if (!use_cond_graph) {
			// Then don't use graphs at all, as it seems an unconditional graph is slower than just launching without graphs (at least for relatively large problem sizes).
			use_cuda_graph = false;
		}
		
		// Get the maximum number of thread blocks that fit on the GPU for the FIM kernel.
        fim_kernel_ptr = (void*) &batched_fim_kernel_new<Config::TILE_W, Config::TILE_H, Config::TILE_D,
            Config::NX, Config::NY, Config::NZ,
            Config::CHANNELS, Config::CHANNELS_F, Config::CHANNELS_V,
            Config::THREE_DIMENSIONAL,
            Config::MSFM, Config::MAX_LOCAL_ITERS, Config::DOUBLE_BUFFERED_SMEM, 
            Config::BACKWARD_PASS, Config::GATED_X>;

		int num_sms, blocks_per_sm;
        CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, fim_kernel_ptr, THREADS_PER_BLOCK, 0
        ));
		
        fixed_grid_size = num_sms * blocks_per_sm;
		
		/*
		std::cout << "GPU SMs: " << num_sms 
		  << " | Blocks/SM: " << blocks_per_sm 
		  << " | Optimal Grid Size: " << fixed_grid_size << std::endl;*/
		
		
		// ALLOCATE FIXED-SIZE VARIABLES ONCE
        CUDA_CHECK(cudaMalloc(&d_next_num_active_blocks, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_active_blocks, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_jobs_claimed, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_iteration, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_list_ptrs, 2 * sizeof(int*)));
        CUDA_CHECK(cudaMalloc(&d_start_time, sizeof(unsigned long long)));
		CUDA_CHECK(cudaMalloc(&d_sign_mode, sizeof(float)));
		CUDA_CHECK(cudaEventCreateWithFlags(&launch_done, cudaEventDisableTiming));
		
		/*
		int min_grid_size; int potential_block_size;
		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &potential_block_size, fim_kernel_ptr, 0, TILE_W * TILE_H * (THREE_DIMENSIONAL ? TILE_D : 1)
        ));
		std::cout << "Min Grid Size: " << min_grid_size << " | Potential Block Size: " << potential_block_size << std::endl;
		*/
    }

        /**
     * @brief Executes the solver by updating the existing graph.
     */
    void solve(void* d_u, size_t pitch_u, const void* d_f, size_t pitch_f, 
               int width, int height, int depth = 1, float dx = 1, int batch_size = 1,
			   const void* d_tof = nullptr, size_t pitch_tof = 0, // These arguments are only used for a backwards pass, where what was originally d_u must be passed here (since it converges to the time-of-flight), while the residual should be passed as d_f. d_u is purely the lambda workspace and is zeroed on entry to every pass.
			   void* d_v = nullptr, size_t pitch_v = 0,
			   bool broadcast_f = false, cudaStream_t stream = 0) {
		this->active_stream = stream;
		
		// The previous solve() may still be running under the conditional-graph path, and a
		// graphExec cannot have its node parameters updated while an execution is pending.
		// cudaEventSynchronize on an already-signalled event is essentially free.
        if (use_cond_graph && graphExec != nullptr)
            CUDA_CHECK(cudaEventSynchronize(launch_done));
		
        // Check Capacity and Allocate if needed
        int blocks_per_row = (width + EFF_W - 1)  / EFF_W;
        int blocks_per_col = (height + EFF_H - 1) / EFF_H;
        int blocks_per_slice = (depth + EFF_D - 1)  / EFF_D;
        int blocks_per_batch = blocks_per_row * blocks_per_col * blocks_per_slice;
        int total_blocks = blocks_per_batch * batch_size;
        
        ensure_capacity(total_blocks);
        
        assert(Config::THREE_DIMENSIONAL || depth == 1);  // "Depth must be 1 for 2D eikonal equation solving."
        
        // Calculate the total bytes for the u buffer to manage the temporary workspace.
        // d_u is pitched: pitch_u bytes per row, (batch_size * height * depth) rows, matching
        // the kernel's row indexing. The padding bytes are covered deliberately, so that the
        // memset, the D2D copy and the final subtraction all span exactly the same range.
        size_t u_bytes = pitch_u * static_cast<size_t>(batch_size) * height * depth;
		
        if (Config::BACKWARD_PASS) {
            // Ensure the temporary accumulation buffer is large enough for the two-pass split
            if (u_bytes > current_u_bytes) {
                if (d_u_temp) CUDA_CHECK(cudaFree(d_u_temp));
                CUDA_CHECK(cudaMalloc(&d_u_temp, u_bytes));
                current_u_bytes = u_bytes;
            }
        }

        // Update persistent class variables (so fimArgs sees the new values)
        void* fimArgs[] = { 
            &d_u, &pitch_u, &d_f, &pitch_f, 
            &d_list_ptrs, &d_num_active_blocks, &d_next_num_active_blocks, 
            &d_is_enqueued, &d_jobs_claimed, &d_iteration, 
            &width, &height, &depth, &dx, 
            &blocks_per_row, &blocks_per_col, &blocks_per_slice, &blocks_per_batch,
            &broadcast_f, &d_tof, &pitch_tof, &d_v, &pitch_v, &d_sign_mode
        };
        
        void* mgtArgs[] = { 
            &d_num_active_blocks, &d_next_num_active_blocks, &d_list_ptrs, 
            &d_jobs_claimed, &d_iteration, &fixed_grid_size, &handle, &d_start_time 
        };
        
        // Define dynamic block dimensions based on mode
        dim3 block_dims(Config::TILE_W, Config::TILE_H, Config::TILE_D);
        
        // =========================================================
        // GRAPH PROGRAMMING (once per solve(), not once per pass)
        // Every kernel argument above is fixed for the duration of this call; only the
        // *contents* of d_sign_mode differ between passes, and that is a device-side copy
        // rather than a node parameter. Re-programming inside the pass loop would therefore
        // write identical values, while racing against the previous pass's pending launch.
        // =========================================================
        if (use_cuda_graph) {
            // LAZY INITIALIZATION: Build the graph on the first run.
            if (mainGraph == nullptr) {
                CUDA_CHECK(cudaGraphCreate(&mainGraph, 0));
                cudaGraph_t bodyGraph = mainGraph;

                #if CUDA_HAS_CONDITIONAL_GRAPHS
                if (use_cond_graph) {
                    // Create conditional loop handle with auto-reset (AssignDefault)
                    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, mainGraph, 1, cudaGraphCondAssignDefault));
                    
                    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
                    cParams.conditional.handle = handle;
                    cParams.conditional.type = cudaGraphCondTypeWhile; 
                    cParams.conditional.size = 1;
                    
                    // Add conditional node directly to the main graph with NO dependencies
                    //CUDA_CHECK(cudaGraphAddNode(&condNode, mainGraph, nullptr, nullptr, 0, &cParams));
                    #if CUDART_VERSION >= 13000
                        // CUDA 13.0+: 6 arguments (includes the new cudaGraphEdgeData* parameter)
                        CUDA_CHECK(cudaGraphAddNode(&condNode, mainGraph, nullptr, nullptr, 0, &cParams));
                    #else
                        // CUDA 12.x: 5 arguments (edge data parameter did not exist on the base function)
                        CUDA_CHECK(cudaGraphAddNode(&condNode, mainGraph, nullptr, 0, &cParams));
                    #endif
                    
                    bodyGraph = cParams.conditional.phGraph_out[0];
                }
                #endif
                
                cudaMemsetParams ms1 = {0};
                ms1.dst = d_next_num_active_blocks; ms1.value = 0; ms1.elementSize = sizeof(int); ms1.height = 1; ms1.width = 1;
                CUDA_CHECK(cudaGraphAddMemsetNode(&msNode1, bodyGraph, nullptr, 0, &ms1));

                cudaMemsetParams ms2 = {0};
                ms2.dst = d_is_enqueued; ms2.value = 0; ms2.elementSize = sizeof(int); ms2.height = 1; ms2.width = total_blocks;
                CUDA_CHECK(cudaGraphAddMemsetNode(&msNode2, bodyGraph, nullptr, 0, &ms2));
                
                cudaGraphNode_t kernelDeps[2] = {msNode1, msNode2};
                
                cudaKernelNodeParams kParamsFIM = {0};
                kParamsFIM.func = fim_kernel_ptr;
                kParamsFIM.gridDim = dim3(fixed_grid_size); 
                kParamsFIM.blockDim = block_dims;
                kParamsFIM.kernelParams = fimArgs; // Points to our persistent vars!
                CUDA_CHECK(cudaGraphAddKernelNode(&fimNode, bodyGraph, kernelDeps, 2, &kParamsFIM));
                
                cudaKernelNodeParams kParamsMgt = {0};
                kParamsMgt.func = (void*)queue_management_kernel;
                kParamsMgt.gridDim = dim3(1); kParamsMgt.blockDim = dim3(1);
                kParamsMgt.kernelParams = mgtArgs;
                CUDA_CHECK(cudaGraphAddKernelNode(&mgtNode, bodyGraph, &fimNode, 1, &kParamsMgt));
                
                CUDA_CHECK(cudaGraphInstantiate(&graphExec, mainGraph, nullptr, nullptr, 0));
                
            } else {
                // ALREADY BUILT: Just update the nodes with low overhead
                cudaMemsetParams ms1 = {0};
                ms1.dst = d_next_num_active_blocks; ms1.value = 0; ms1.elementSize = sizeof(int); ms1.height = 1; ms1.width = 1;
                CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(graphExec, msNode1, &ms1));

                // Update ms2 to use total_blocks so we don't clear memory we aren't using
                cudaMemsetParams ms2 = {0};
                ms2.dst = d_is_enqueued; ms2.value = 0; ms2.elementSize = sizeof(int); ms2.height = 1; ms2.width = total_blocks;
                CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(graphExec, msNode2, &ms2));

                cudaKernelNodeParams updatedParams = {0};
                updatedParams.func = fim_kernel_ptr;
                updatedParams.gridDim = dim3(fixed_grid_size);
                updatedParams.blockDim = block_dims;
                updatedParams.kernelParams = fimArgs;
                CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec, fimNode, &updatedParams));

                cudaKernelNodeParams mgtParams = {0};
                mgtParams.func = (void*)queue_management_kernel;
                mgtParams.gridDim = dim3(1); mgtParams.blockDim = dim3(1);
                mgtParams.kernelParams = mgtArgs;
                CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec, mgtNode, &mgtParams));
            }
        }

        // =========================================================
        // MULTI-PASS EXECUTION LOOP
        // Forward Pass = 1 Pass (Mode 0)
        // Backward Pass = 2 Passes (Mode 1: Positive, Mode -1: Negative)
        // =========================================================
        int num_passes = Config::BACKWARD_PASS ? 2 : 1;

        for (int pass = 0; pass < num_passes; ++pass) {
            
            // Between passes, save the positive solution and clear the working grid for the negative pass
            if (Config::BACKWARD_PASS) {
                if (pass == 1) {
                    // Save the positive solution before clearing for the negative pass.
                    CUDA_CHECK(cudaMemcpyAsync(d_u_temp, d_u, u_bytes, cudaMemcpyDeviceToDevice, stream));
                }
                // Monotone accumulation from below requires lambda == 0 on entry to EVERY pass.
                // Any nonzero start above the fixed point becomes a permanent floor under fmaxf.
                CUDA_CHECK(cudaMemsetAsync(d_u, 0, u_bytes, stream));
            }

            // Set the execution mode for the current pass
            float h_sign_mode = Config::BACKWARD_PASS ? (pass == 0 ? 1.0f : -1.0f) : 1.0f;
            CUDA_CHECK(cudaMemcpyAsync(d_sign_mode, &h_sign_mode, sizeof(float), cudaMemcpyHostToDevice, stream));

            // Initialize state arrays. (d_list_ptrs must be re-seeded every pass: the queue
            // management kernel swaps the two pointers in place as it ping-pongs.)
            int* h_list_ptrs[2] = {d_current_active_block_list, d_next_active_block_list};
            CUDA_CHECK(cudaMemcpyAsync(d_list_ptrs, h_list_ptrs, 2 * sizeof(int*), cudaMemcpyHostToDevice, stream));
            
            CUDA_CHECK(cudaMemsetAsync(d_is_enqueued, 0, total_blocks * sizeof(int), stream));
            CUDA_CHECK(cudaMemcpyAsync(d_num_active_blocks, &total_blocks, sizeof(int), cudaMemcpyHostToDevice, stream));
            // fixed_grid_size is a persistent class member, and pageable H2D copies stage the
            // host bytes before returning, so the async form is safe. Unlike the blocking
            // version it is also correctly ordered against a non-blocking `stream`.
            CUDA_CHECK(cudaMemcpyAsync(d_jobs_claimed, &fixed_grid_size, sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(d_iteration, 0, sizeof(int), stream));
            
            // Initial active block list contains all blocks: 0, 1, 2, ..., total_blocks-1.
            thrust::sequence(thrust::cuda::par.on(stream), 
                     d_current_active_block_list, 
                     d_current_active_block_list + total_blocks);
            
            // Graph Execution
            if (use_cuda_graph) {
                // Launch the Graph!
                if (use_cond_graph) {
                    // Tier 1: Fully Asynchronous GPU-only Loop.
                    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
                    if (pass == num_passes - 1)
                        CUDA_CHECK(cudaEventRecord(launch_done, stream));
                    // Warning: Since this call is fully asynchronous, the caller is responsible for synchronizing, if necessary.
                } else {
                    // Tier 2: Static Graph CPU Loop
                    int h_num_active_blocks = total_blocks;
                    int outer_iteration_cnt = 0;
                    while (h_num_active_blocks > 0) {
                        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
                        CUDA_CHECK(cudaMemcpyAsync(&h_num_active_blocks, d_num_active_blocks, sizeof(int), cudaMemcpyDeviceToHost, stream));
                        #ifdef MATLAB_MEX_FILE
                        if ((outer_iteration_cnt % 50 == 0) && utIsInterruptPending()) { // Yield to MATLAB.
                            utSetInterruptPending(false);
                            throw std::runtime_error("Execution halted by user (CTRL+C pressed).");
                        }
                        #endif
                        #ifdef WITH_TORCH
                        // PyTorch native signal check.
                        torch::check_signals();
                        #endif
                        #ifdef WITH_JAX
                        // JAX / Standard Pybind11 signal check.
                        // (Requires acquiring GIL briefly to check Python state)
                        py::gil_scoped_acquire acquire;
                        py::check_signals();
                        #endif
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        outer_iteration_cnt++;
                    }
                }
            } else {
                // Tier 3: Manual Execution Fallback (Non-graph CPU Loop).
                int h_num_active_blocks = total_blocks;
                int outer_iteration_cnt = 0;
                while (h_num_active_blocks > 0) {
                    // A. Memsets.
                    CUDA_CHECK(cudaMemsetAsync(d_next_num_active_blocks, 0, sizeof(int), stream));
                    CUDA_CHECK(cudaMemsetAsync(d_is_enqueued, 0, total_blocks * sizeof(int), stream));

                    // B. FIM Kernel Launch.
                    CUDA_CHECK(cudaLaunchKernel(fim_kernel_ptr, dim3(min(fixed_grid_size, h_num_active_blocks)), block_dims, fimArgs, 0, stream));
                    
                    // C. Queue Management Launch (passing a null handle).
                    CUDA_CHECK(cudaLaunchKernel((void*)queue_management_kernel, dim3(1), dim3(1), mgtArgs, 0, stream));

                    // D. Synchronize and Check Condition.
                    CUDA_CHECK(cudaMemcpyAsync(&h_num_active_blocks, d_num_active_blocks, sizeof(int), cudaMemcpyDeviceToHost, stream));
                    #ifdef MATLAB_MEX_FILE
                    if ((outer_iteration_cnt % 50 == 0) && utIsInterruptPending()) { // Yield to MATLAB.
                        utSetInterruptPending(false);
                        throw std::runtime_error("Execution halted by user (CTRL+C pressed).");
                    }
                    #endif
                    #ifdef WITH_TORCH
                    // PyTorch native signal check.
                    torch::check_signals();
                    #endif
                    #ifdef WITH_JAX
                    // JAX / Standard Pybind11 signal check.
                    // (Requires acquiring GIL briefly to check Python state)
                    py::gil_scoped_acquire acquire;
                    py::check_signals();
                    #endif
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    outer_iteration_cnt++;
                }
            }
        } // End of num_passes loop

        // After both passes complete, recombine the split residual: d_u = d_u_temp - d_u,
        // i.e. lambda = lambda_positive - lambda_negative.
        if (Config::BACKWARD_PASS) {
            size_t total_elements = u_bytes / sizeof(float);
            
            // Map raw pointers to thrust device pointers
            thrust::device_ptr<float> ptr_dest(reinterpret_cast<float*>(d_u));
            thrust::device_ptr<float> ptr_src(reinterpret_cast<float*>(d_u_temp));
            
            // Subtract Pass 2 from Pass 1: result = ptr_src - ptr_dest
            thrust::transform(thrust::cuda::par.on(stream),
                              ptr_src, ptr_src + total_elements,   // First input (Pass 1)
                              ptr_dest,                            // Second input (Pass 2)
                              ptr_dest,                            // Output destination (overwrites d_u with the final difference)
                              thrust::minus<float>());
        }
    }
	

    /**
     * @brief Destructor frees all resources.
     */
    ~BatchedFIMSolver() {
		CUDA_CHECK(cudaStreamSynchronize(active_stream));
        if (use_cuda_graph && graphExec != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(graphExec));
            CUDA_CHECK(cudaGraphDestroy(mainGraph));
        }
		CUDA_CHECK(cudaFree(d_current_active_block_list));
        CUDA_CHECK(cudaFree(d_next_active_block_list));
        CUDA_CHECK(cudaFree(d_is_enqueued));
        CUDA_CHECK(cudaFree(d_next_num_active_blocks));
        CUDA_CHECK(cudaFree(d_num_active_blocks));
        CUDA_CHECK(cudaFree(d_jobs_claimed));
        CUDA_CHECK(cudaFree(d_list_ptrs));
        CUDA_CHECK(cudaFree(d_iteration));
        CUDA_CHECK(cudaFree(d_start_time));
		CUDA_CHECK(cudaFree(d_sign_mode));
		if (d_u_temp) CUDA_CHECK(cudaFree(d_u_temp));
		CUDA_CHECK(cudaEventDestroy(launch_done));
    }
};

#endif //BATCHEDFIMSOLVER_H