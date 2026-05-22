#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <iostream>

#include "eiko_dispatch.cuh"  // Centralized solver dispatcher and config factory.

// ---------------------------------------------------------
// OPAQUE STRUCT
// Must match the Python struct.pack('iiiifiiiiiii', ...)
// ---------------------------------------------------------
struct alignas(16) FIMOpaque {
    int width;
    int height;
    int depth;
    int batch_size;
    float dx;
    int is_3d;
    int is_backward;
    int msfm;
    int has_v;
    int broadcast_f;
    int gated_x;
    int has_tof;
};

// ---------------------------------------------------------
// DISPATCH FUNCTOR
// ---------------------------------------------------------'
template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
struct FIMSolveOp {
    void operator()(cudaStream_t stream, void** buffers, const FIMOpaque& cfg) {
        
        // A. Retrieve the exact compile-time config for memory calculations
        constexpr auto Config = MakeDefaultConfig<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>();

        // B. Map XLA Buffers. 
        // JAX places inputs first, followed by outputs.
        void* d_u_in  = buffers[0];
        void* d_f     = buffers[1];
        
        int buf_idx = 2;
        
        void* d_v_in = nullptr;
        size_t pitch_v = 0;
        if constexpr (HAS_V) {
            d_v_in    = buffers[buf_idx++];
            pitch_v   = cfg.width * sizeof(float) * Config.CHANNELS_V;
        }

        void* d_tof = nullptr;
        size_t pitch_tof = 0;
        if constexpr (IS_BACKWARD) {
            if (cfg.has_tof) {
                d_tof     = buffers[buf_idx++];
                pitch_tof = cfg.width * sizeof(float) * Config.CHANNELS;
            } else {
                d_tof     = d_u_in;
                pitch_tof = cfg.width * sizeof(float) * Config.CHANNELS;
            }
        }

        // The final buffer contains the outputs.
        void* d_u_out = nullptr;
        void* d_v_out = nullptr;

        if constexpr (HAS_V) {
            // XLA Multiple Outputs: The final pointer is a void** array of pointers.
            void** tuple_outs = reinterpret_cast<void**>(buffers[buf_idx]);
            d_u_out = tuple_outs[0];
            d_v_out = tuple_outs[1];
        } else {
            // XLA Single Output: The final pointer is the direct buffer.
            d_u_out = buffers[buf_idx];
        }

        // C. Handle JAX Immutability
        // We clone the inputs into the safe output buffers that the solver will mutate.
        size_t num_elements = static_cast<size_t>(cfg.batch_size) * cfg.depth * cfg.height * cfg.width;
        size_t u_bytes = num_elements * sizeof(float) * Config.CHANNELS;
        cudaMemcpyAsync(d_u_out, d_u_in, u_bytes, cudaMemcpyDeviceToDevice, stream);
        
        if constexpr (HAS_V) {
            size_t v_bytes = num_elements * sizeof(float) * Config.CHANNELS_V;
            cudaMemcpyAsync(d_v_out, d_v_in, v_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        // D. Calculate Primary Pitches
        size_t pitch_u = cfg.width * sizeof(float) * Config.CHANNELS;
        size_t pitch_f = cfg.width * sizeof(float) * Config.CHANNELS_F;
        
        // E. Instantiate using the Standard Alias and Execute
        StandardFIMSolver<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X> solver;
        
        solver.solve(
            d_u_out, pitch_u, 
            d_f, pitch_f, 
            cfg.width, cfg.height, cfg.depth, 
            cfg.dx, cfg.batch_size, 
            d_tof, pitch_tof, 
            d_v_out, pitch_v,  // <--- Pass the safe output buffer
            static_cast<bool>(cfg.broadcast_f), stream
        );
        
        // solver goes out of scope here. Its destructor cleans up cudaMallocs 
        // and synchronizes the stream, safely returning control to XLA.
    }
};

// ---------------------------------------------------------
// XLA CUSTOM CALL TARGET
// The C-ABI compliant entry point called by the JAX runtime.
// ---------------------------------------------------------
extern "C" void jax_fim_solve(cudaStream_t stream, void** buffers, 
                              const char* opaque, size_t opaque_len, 
                              void* status) {
    
    // Unpack scalar configurations from Python
    const FIMOpaque& cfg = *reinterpret_cast<const FIMOpaque*>(opaque);

    // Call the template dispatcher. 
    // This resolves the 4 boolean flags into 1 of the 16 compiled template instances.
    dispatch_fim<FIMSolveOp>(
			static_cast<bool>(cfg.is_3d), 
            static_cast<bool>(cfg.is_backward), 
            static_cast<bool>(cfg.msfm), 
            static_cast<bool>(cfg.has_v),
            static_cast<bool>(cfg.gated_x),
        stream, buffers, cfg
    );
}

// ---------------------------------------------------------
// PYBIND11 REGISTRATION
// ---------------------------------------------------------
PYBIND11_MODULE(eiko_jax_impl, m) {
    m.def("registrations", []() {
        pybind11::dict dict;
        // Wrap the C function pointer in a PyCapsule so JAX can read it
        dict["jax_fim_solve"] = pybind11::capsule((void*)jax_fim_solve, "xla._CUSTOM_CALL_TARGET");
        return dict;
    });
}