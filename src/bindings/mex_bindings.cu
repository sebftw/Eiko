/*
 * Copyright (c) 2026, Sebastian Kazmarek Præsius. All rights reserved.
 * Licensed under the BSD 3-Clause License. See LICENSE file in the project root for details.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdint.h>
#include <string>
#include <stdexcept>

#include "eiko_dispatch.cuh"  // Centralized solver dispatcher and config factory.

// ------------------------------------------------------------------------
// Dispatch Functors
// ------------------------------------------------------------------------
template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
struct AllocatorOp {
    void operator()(void*& solver_ptr) {
        solver_ptr = new StandardFIMSolver<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>();
    }
};

template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
struct DeleterOp {
    void operator()(void*& solver_ptr) {
        delete static_cast<StandardFIMSolver<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>*>(solver_ptr);
        solver_ptr = nullptr;
    }
};

template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
struct SolveOp {
    void operator()(void* solver_ptr, void* d_u, const void* d_f, void* d_v, const void* d_tof, 
                    int width, int height, int depth, float dx, int batch_size, 
                    bool broadcast_f, cudaStream_t stream) {
        
        // Fetch the exact configuration mapped to this specific template instance.
        constexpr auto Config = MakeDefaultConfig<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>();

        // Safely calculate pitches using the actual configuration constants.
        size_t pitch_u   = width * sizeof(float) * Config.CHANNELS;
        size_t pitch_f   = width * sizeof(float) * Config.CHANNELS_F;
        size_t pitch_v   = HAS_V ? (width * sizeof(float) * Config.CHANNELS_V) : 0;
        size_t pitch_tof = IS_BACKWARD ? pitch_u : 0;

        // Execute.
        auto* solver = static_cast<StandardFIMSolver<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>*>(solver_ptr);
        solver->solve(d_u, pitch_u, d_f, pitch_f, width, height, depth, dx, batch_size, 
                      d_tof, pitch_tof, d_v, pitch_v, broadcast_f, stream);
    }
};

// ------------------------------------------------------------------------
// MEX Entry Point
// ------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Initialize the MathWorks GPU API
    mxInitGPU();

    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("FIM:InvalidArgs", "First argument must be a command string ('new', 'delete', 'solve').");
    }

    char cmd[64];
    mxGetString(prhs[0], cmd, sizeof(cmd));
    std::string command(cmd);

    if (command == "new") {
        // Args: 'new', is_3d, is_backward, msfm, has_v, gated_x
        bool is_3d       = mxGetScalar(prhs[1]);
        bool is_backward = mxGetScalar(prhs[2]);
        bool msfm        = mxGetScalar(prhs[3]);
        bool has_v       = mxGetScalar(prhs[4]);
        bool gated_x     = mxGetScalar(prhs[5]);
        void* solver_ptr = nullptr;

        dispatch_fim<AllocatorOp>(is_3d, is_backward, msfm, has_v, gated_x, solver_ptr);

        // Return the pointer to MATLAB as a uint64
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t*)mxGetData(plhs[0])) = reinterpret_cast<uint64_t>(solver_ptr);

    } else if (command == "delete") {
        // Args: 'delete', pointer, is_3d, is_backward, msfm, has_v, gated_x
        uint64_t ptr_val = *((uint64_t*)mxGetData(prhs[1]));
        void* solver_ptr = reinterpret_cast<void*>(ptr_val);
        
        bool is_3d       = mxGetScalar(prhs[2]);
        bool is_backward = mxGetScalar(prhs[3]);
        bool msfm        = mxGetScalar(prhs[4]);
        bool has_v       = mxGetScalar(prhs[5]);
        bool gated_x     = mxGetScalar(prhs[6]);

        if (solver_ptr)
            dispatch_fim<DeleterOp>(is_3d, is_backward, msfm, has_v, gated_x, solver_ptr);

    } else if (command == "solve") {
        // Args: 'solve', pointer, u, f, v, tof, dx, broadcast_f, is_3d, is_backward, msfm, has_v, gated_x
        uint64_t ptr_val = *((uint64_t*)mxGetData(prhs[1]));
        void* solver_ptr = reinterpret_cast<void*>(ptr_val);

        const mxArray* u_arr   = prhs[2];
        const mxArray* f_arr   = prhs[3];
        const mxArray* v_arr   = prhs[4];
        const mxArray* tof_arr = prhs[5];
        
        float dx = 1.0f; // Set your default value
        if (!mxIsEmpty(prhs[6])) {
            dx = static_cast<float>(mxGetScalar(prhs[6]));
        }
        bool broadcast_f = mxGetScalar(prhs[7]);
        bool is_3d       = mxGetScalar(prhs[8]);
        bool is_backward = mxGetScalar(prhs[9]);
        bool msfm        = mxGetScalar(prhs[10]);
        bool has_v       = mxGetScalar(prhs[11]);
        bool gated_x     = mxGetScalar(prhs[12]);

        // Data upload to GPU.
        // If the array is on the CPU, MATLAB automatically allocates GPU
        // memory and safely uploads the data to the device right here.
        mxGPUArray* gpu_u = mxGPUCopyFromMxArray(u_arr); 
        const mxGPUArray* gpu_f = mxGPUCreateFromMxArray(f_arr);

        mxGPUArray* gpu_v = nullptr;
        if (has_v && !mxIsEmpty(v_arr)) {
            gpu_v = mxGPUCopyFromMxArray(v_arr);
        }
        
        const mxGPUArray* gpu_tof = nullptr;
        if (is_backward && !mxIsEmpty(tof_arr)) {
            gpu_tof = mxGPUCreateFromMxArray(tof_arr);
        }

        // Data Type Post-Validation.
        // Since CPU arrays are allowed, ensure they were passed as 'single' (float).
        if (mxGPUGetClassID(gpu_u) != mxSINGLE_CLASS || mxGPUGetClassID(gpu_f) != mxSINGLE_CLASS) {
            mxGPUDestroyGPUArray(gpu_u);
            mxGPUDestroyGPUArray(gpu_f);
            if (gpu_v) mxGPUDestroyGPUArray(gpu_v);
            if (gpu_tof) mxGPUDestroyGPUArray(gpu_tof);
            mexErrMsgIdAndTxt("FIM:InvalidArgs", "Inputs 'u' and 'f' must be single-precision floating point arrays.");
        }
        if (gpu_v && mxGPUGetClassID(gpu_v) != mxSINGLE_CLASS) {
            mxGPUDestroyGPUArray(gpu_u);
            mxGPUDestroyGPUArray(gpu_f);
            mxGPUDestroyGPUArray(gpu_v);
            if (gpu_tof) mxGPUDestroyGPUArray(gpu_tof);
            mexErrMsgIdAndTxt("FIM:InvalidArgs", "Input 'v' must be a single-precision floating point array.");
        }
        if (gpu_tof && mxGPUGetClassID(gpu_tof) != mxSINGLE_CLASS) {
            mxGPUDestroyGPUArray(gpu_u);
            mxGPUDestroyGPUArray(gpu_f);
            if (gpu_v) mxGPUDestroyGPUArray(gpu_v);
            mxGPUDestroyGPUArray(gpu_tof);
            mexErrMsgIdAndTxt("FIM:InvalidArgs", "Input 'tof' must be a single-precision floating point array.");
        }

        // Query dimensions via the GPU structures (Safe for both CPU-uploaded and native GPU data).
        mwSize ndims = mxGPUGetNumberOfDimensions(gpu_u);
        const mwSize* dims = mxGPUGetDimensions(gpu_u);

        // Note: MATLAB is column-major. We map MATLAB's 1st dim to Width (fastest moving) 
        // and 2nd dim to Height to preserve memory contiguity without transposing.
        int width  = static_cast<int>(dims[0]);
        int height = (ndims > 1) ? static_cast<int>(dims[1]) : 1;
        int depth  = is_3d ? ((ndims > 2) ? static_cast<int>(dims[2]) : 1) : 1;
        int batch_size = is_3d ? (ndims > 3 ? static_cast<int>(dims[3]) : 1) 
                               : (ndims > 2 ? static_cast<int>(dims[2]) : 1);
        
        // Clean up host-side array dimensions allocation
        mxFree(const_cast<mwSize*>(dims));

        // u and v utilize the write-allowed extraction route to guarantee CoW compliance.
        void* d_u             = mxGPUGetData(gpu_u);
        const void* d_f       = mxGPUGetDataReadOnly(gpu_f);
        void* d_v             = gpu_v ? mxGPUGetData(gpu_v) : nullptr;
        const void* d_tof     = gpu_tof ? mxGPUGetDataReadOnly(gpu_tof) : nullptr;

        // Execute on the default stream (stream 0).
        std::string error_message = "";
        try {
            dispatch_fim<SolveOp>(
                is_3d, is_backward, msfm, has_v, gated_x,
                solver_ptr, d_u, d_f, d_v, d_tof, 
                width, height, depth, dx, batch_size, 
                broadcast_f, static_cast<cudaStream_t>(0)
            );
        } catch (const std::exception& e) {
            error_message = e.what();
        } catch (...) {
            error_message = "An unknown C++ exception occurred during SolveOp execution.";
        }

        if (!error_message.empty()) {
            mxGPUDestroyGPUArray(gpu_u);
            mxGPUDestroyGPUArray(gpu_f);
            if (gpu_v) mxGPUDestroyGPUArray(gpu_v);
            if (gpu_tof) mxGPUDestroyGPUArray(gpu_tof);
            
            // Now it is safe to throw the MATLAB error
            mexErrMsgIdAndTxt("FIM:SolveOpError", "Solver failed: %s", error_message.c_str());
            return;
        }

        // Wrap device arrays back into MATLAB mxArrays and assign them to outputs
        // Note: Even if inputs originally came from the CPU, the output returned to MATLAB 
        // will now be a proper, modified `gpuArray`.
        plhs[0] = mxGPUCreateMxArrayOnGPU(gpu_u);
        if (nlhs > 1) {
            if (has_v && gpu_v) {
                plhs[1] = mxGPUCreateMxArrayOnGPU(gpu_v);
            } else {
                plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
            }
        }

        // 7. Clean up CPU-side wrappers
        mxGPUDestroyGPUArray(gpu_u);
        mxGPUDestroyGPUArray(gpu_f);
        if (gpu_v) mxGPUDestroyGPUArray(gpu_v);
        if (gpu_tof) mxGPUDestroyGPUArray(gpu_tof);
    }
}