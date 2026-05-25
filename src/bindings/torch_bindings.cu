/*
 * Copyright (c) 2026, Sebastian Kazmarek Præsius. All rights reserved.
 * Licensed under the BSD 3-Clause License. See LICENSE file in the project root for details.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <stdexcept>
#include <string>

// Include the centralized dispatcher and configuration factory.
#include "eiko_dispatch.cuh"


// ------------------------------------------------------------------------
// Functors for dispatch_fim
// ------------------------------------------------------------------------
template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
struct AllocatorOp {
    void operator()(void*& solver_ptr) {
        // Instantiate using the clean Standard alias defined in fim_dispatch.cuh
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
    void operator()(void* solver_ptr, void* d_u, void* d_f, 
                    int width, int height, int depth, float dx, int batch_size, 
                    void* d_tof, void* d_v, 
                    bool broadcast_f, cudaStream_t stream) {
        
		// Retrieve compile-time configuration for exact memory calculation.
        constexpr auto Config = MakeDefaultConfig<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>();

        size_t pitch_u   = width * sizeof(float) * Config.CHANNELS;
        size_t pitch_f   = width * sizeof(float) * Config.CHANNELS_F;
        size_t pitch_v   = HAS_V ? (width * sizeof(float) * Config.CHANNELS_V) : 0;
        size_t pitch_tof = IS_BACKWARD ? pitch_u : 0;

        auto* solver = static_cast<StandardFIMSolver<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>*>(solver_ptr);
        solver->solve(d_u, pitch_u, d_f, pitch_f, width, height, depth, dx, batch_size, 
                      d_tof, pitch_tof, d_v, pitch_v, broadcast_f, stream);
    }
};

// ------------------------------------------------------------------------
// Stateful PyTorch Wrapper
// ------------------------------------------------------------------------
class FIMPyTorchWrapper {
private:
    void* solver_ptr = nullptr;
    bool is_3d;
    bool is_backward;
    bool msfm;
    bool has_v; // Track V state to ensure runtime checks match compile-time.
    bool gated_x;

    // Helper to extract device pointer and pitch safely
    template <typename T>
    void extract_tensor_data(torch::Tensor t, void*& ptr, size_t& pitch, int width) {
        if (t.defined()) {
            ptr = t.data_ptr();
            pitch = width * t.element_size();
        } else {
            ptr = nullptr;
            pitch = 0;
        }
    }

public:
    // Initialize the solver by dispatching to the correct configuration
    FIMPyTorchWrapper(bool is_3d, bool is_backward, bool msfm, bool has_v, bool gated_x) 
        : is_3d(is_3d), is_backward(is_backward), msfm(msfm), has_v(has_v), gated_x(gated_x) {
        
        dispatch_fim<AllocatorOp>(is_3d, is_backward, msfm, has_v, gated_x, solver_ptr);
        
        if (!solver_ptr) {
            throw std::runtime_error("Failed to initialize BatchedFIMSolver.");
        }
    }

    ~FIMPyTorchWrapper() {
        if (solver_ptr) {
            dispatch_fim<DeleterOp>(is_3d, is_backward, msfm, has_v, gated_x, solver_ptr);
        }
    }

    // The main execution block
    void solve(torch::Tensor u, torch::Tensor f, std::optional<torch::Tensor> v, 
               std::optional<torch::Tensor> tof, float dx) {
        
        // Safety guard to ensure runtime inputs match the compile-time allocation.
        TORCH_CHECK(v.has_value() == has_v, 
                    "Solver was initialized with has_v=", has_v, 
                    ", but v tensor presence is ", v.has_value());

		// --------------------------------------------------------------------
        // Memory Continuity & Data Type Guards
        // Using .contiguous() silently copies data. If 'u' is modified in-place,
        // changes to a copy are lost. We strictly enforce continuity instead.
        // --------------------------------------------------------------------
		// Enforce contiguous memory layout to calculate pitches securely.
        TORCH_CHECK(u.is_contiguous(), "Input 'u' must be contiguous in memory.");
        TORCH_CHECK(f.is_contiguous(), "Input 'f' must be contiguous in memory.");
        TORCH_CHECK(u.scalar_type() == torch::kFloat32, "Input 'u' must be float32.");
        TORCH_CHECK(f.scalar_type() == torch::kFloat32, "Input 'f' must be float32.");
		
        u = u.contiguous();
        f = f.contiguous();
        torch::Tensor v_tensor = v.has_value() ? v.value().contiguous() : torch::Tensor();
        torch::Tensor tof_tensor = tof.has_value() ? tof.value().contiguous() : torch::Tensor();

        if (has_v) {
            TORCH_CHECK(v.value().is_contiguous(), "Input 'v' must be contiguous.");
            TORCH_CHECK(v.value().scalar_type() == torch::kFloat32, "Input 'v' must be float32.");
        }
        if (tof.has_value()) {
            TORCH_CHECK(tof.value().is_contiguous(), "Input 'tof' must be contiguous.");
            TORCH_CHECK(tof.value().scalar_type() == torch::kFloat32, "Input 'tof' must be float32.");
        }

        // Infer dimensions based on u
        int ndims = u.dim();
        int batch_size = ndims > 2 + is_3d ? u.size(0) : 1;
        int depth = is_3d ? u.size(ndims - 3) : 1;
        int height = u.size(ndims - 2);
        int width = u.size(ndims - 1);
        
        // Broadcast detection for f
        bool broadcast_f = false;
        if (f.dim() == u.dim() - 1) {
            broadcast_f = true; // Missing batch dimension entirely
        } else if (f.dim() == u.dim() && f.size(0) == 1 && batch_size > 1) {
            broadcast_f = true; // Singleton batch dimension
        }

        // Extract pointers directly
        void* d_u = u.data_ptr();
        void* d_f = f.data_ptr();
        void* d_v = has_v ? v.value().data_ptr() : nullptr;
        
        // TOF Fallback Logic for Adjoint / Backward solvers
        void* d_tof = nullptr;
        if (is_backward) {
            d_tof = tof.has_value() ? tof.value().data_ptr() : d_u;
        }

        // Retrieve current PyTorch CUDA stream (ensures compatibility with torch.cuda.stream contexts)
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // Dispatch the execution
        dispatch_fim<SolveOp>(
            is_3d, is_backward, msfm, has_v, gated_x, 
            solver_ptr, d_u, d_f, 
            width, height, depth, dx, batch_size, 
            d_tof, d_v, broadcast_f, stream
        );
    }
};

// ------------------------------------------------------------------------
// PyBind11 Registration
// ------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<FIMPyTorchWrapper>(m, "BatchedFIMSolver")
        .def(pybind11::init<bool, bool, bool, bool, bool>(), 
             pybind11::arg("is_3d") = false, 
             pybind11::arg("is_backward") = false, 
             pybind11::arg("msfm") = false,
             pybind11::arg("has_v") = false,
             pybind11::arg("gated_x") = false)
        .def("solve", &FIMPyTorchWrapper::solve,
             pybind11::arg("u"), pybind11::arg("f"), 
             pybind11::arg("v") = pybind11::none(), 
             pybind11::arg("tof") = pybind11::none(), 
             pybind11::arg("dx") = 1.0f);
}