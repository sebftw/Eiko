/*
 * Copyright (c) 2026, Sebastian Kazmarek Præsius. All rights reserved.
 * Licensed under the BSD 3-Clause License. See LICENSE file in the project root for details.
 */

#ifndef EIKO_KERNELS_H
#define EIKO_KERNELS_H

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>  // for std::setw
#include <cassert>  // for assert

#if __has_include(<cccl/thrust/sequence.h>)
    #include <cccl/thrust/sequence.h>
    #include <cccl/thrust/execution_policy.h>
    #include <cccl/thrust/device_ptr.h>
    #include <cccl/thrust/copy.h>
    #include <cccl/thrust/iterator/counting_iterator.h>
#else
    #include <thrust/sequence.h>
    #include <thrust/execution_policy.h>
    #include <thrust/device_ptr.h>
    #include <thrust/copy.h>
    #include <thrust/iterator/counting_iterator.h>
#endif

#include <cuda/std/functional>

#ifdef MATLAB_MEX_FILE
extern "C" bool utIsInterruptPending(void);
extern "C" bool utSetInterruptPending(bool);
#endif

#ifdef WITH_JAX
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#ifdef WITH_TORCH
#include <torch/extension.h>
#endif

// =========================================================
// ERROR CHECKING MACRO
// =========================================================
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ------------------------------------------------------------------------
// CUDA TOOLKIT BACKWARD COMPATIBILITY LAYER (Pre-CUDA 12.4)
// ------------------------------------------------------------------------
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    #define CUDA_HAS_CONDITIONAL_GRAPHS 1
#else
    #define CUDA_HAS_CONDITIONAL_GRAPHS 0
    
    // Define a fallback type alias so the variable declarations compile.
    using cudaGraphConditionalHandle = unsigned long long;
    
    // Issue a warning to notify the user.
    #if defined(_MSC_VER)
        #pragma message("Warning: CUDA Toolkit version is pre-12.4. Conditional CUDA Graphs are disabled; falling back to CPU-driven loop synchronization.")
    #else
        #warning "CUDA Toolkit version is pre-12.4. Conditional CUDA Graphs are disabled; falling back to CPU-driven loop synchronization."
    #endif
#endif

// To enable the compiler to recognize 8-byte and 16-byte alignments
// for vectorized ld.global.v2 and ld.global.v4 instructions:
template <int CHANNELS> struct CudaVec;
template <> struct CudaVec<1> { using Type = union { float v; float f[1]; }; };
template <> struct CudaVec<2> { using Type = union { float2 v; float f[2]; };; };
template <> struct CudaVec<4> { using Type = union { float4 v; float f[4]; };; };

constexpr float INFINITY_PLACEHOLDER = std::numeric_limits<float>::infinity();


struct FIMConfig {
	// Base parameters
	bool THREE_DIMENSIONAL = false;
	int NX = 1;                     // Pixels per thread in X.
    int NY = 1;                     // Pixels per thread in Y.
    int NZ = 1;                     // Pixels per thread in Z.
    int TILE_W = 16;                // Thread block size in X (width).
    int TILE_H = 16;                // Thread block size in Y (height).
    int TILE_D = 1;                 // Thread block size in Z (depth).
    int CHANNELS = 1;               // Number of channels/sources to solve per thread.
    int CHANNELS_F = 1;             // Number of speed field channels.
	int CHANNELS_V = 0;
    int MAX_LOCAL_ITERS = 30;       // Iteration limit for block-local updates.
    bool MSFM = false;              // Multi-Stencils Fast Marching flag.
    bool DOUBLE_BUFFERED_SMEM = false; // Use double buffering in shared memory.
	bool BACKWARD_PASS = false;     // To perform a backwards pass (gradient calculation).
    bool GATED_X = false;           // To only allow information to travel forward in x.
	
	// Overloading the << operator for easy printing.
    friend std::ostream& operator<<(std::ostream& os, const FIMConfig& c) {
        auto b_str = [](bool b) { return b ? "true" : "false"; };

        os << "--- FIM Configuration ---\n"
           << std::left << std::setw(25) << "Dimension:"       << (c.THREE_DIMENSIONAL ? "3D" : "2D") << "\n"
           << std::left << std::setw(25) << "Pixels/Thread:"   << "NX=" << c.NX << ", NY=" << c.NY << ", NZ=" << c.NZ << "\n"
           << std::left << std::setw(25) << "Tile Size:"       << "W=" << c.TILE_W << ", H=" << c.TILE_H << ", D=" << c.TILE_D << "\n"
           << std::left << std::setw(25) << "Channels (U/F):"  << c.CHANNELS << " / " << c.CHANNELS_F << "\n"
           << std::left << std::setw(25) << "Max Local Iters:" << c.MAX_LOCAL_ITERS << "\n"
           << std::left << std::setw(25) << "MSFM Enabled:"    << b_str(c.MSFM) << "\n"
           << std::left << std::setw(25) << "Double Buffered:" << b_str(c.DOUBLE_BUFFERED_SMEM) << "\n"
           << "-------------------------";
        return os;
    }
};

template <typename Config>
struct FIMConstants {
    static constexpr int HALO = 1;
    // Replace the dot (.) with the scope resolution operator (::)
    static constexpr int EFF_W = Config::TILE_W * Config::NX;
    static constexpr int EFF_H = Config::TILE_H * Config::NY;
    static constexpr int EFF_D = Config::TILE_D * Config::NZ;
    static constexpr int SMEM_W = EFF_W + 2 * HALO;
    static constexpr int SMEM_H = EFF_H + 2 * HALO;
    static constexpr int SMEM_D = EFF_D + 2 * HALO;
    static constexpr int THREADS_PER_BLOCK = Config::TILE_W * Config::TILE_H * (Config::THREE_DIMENSIONAL ? Config::TILE_D : 1);
};


// =========================================================
// KERNEL MACROS & DEVICE FUNCTIONS
// =========================================================
__device__ __forceinline__ unsigned long long get_global_time() {
    unsigned long long global_time;
    // Fetch the constant-rate nanosecond timer
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
    return global_time;
}

__device__ __forceinline__ unsigned int get_tid_x() {
	// Helper function to avoid tid.x being cached in registers.
    unsigned int tid;
    // 'mov.u32' moves the value from the special register %tid.x into 'tid'
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
	return tid;
    //return threadIdx.x;
}

__device__ __forceinline__ unsigned int get_tid_y() {
    unsigned int tid;
    // 'mov.u32' moves the value from the special register %tid.x into 'tid'
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(tid));
	return tid;
    // return threadIdx.y;
}

__device__ __forceinline__ unsigned int get_tid_z() {
    unsigned int tid;
    // 'mov.u32' moves the value from the special register %tid.x into 'tid'
    asm volatile("mov.u32 %0, %%tid.z;" : "=r"(tid));
    return tid;
	// return threadIdx.z;
}

__launch_bounds__(1)
__global__ void queue_management_kernel(
    int* d_num_active, 
    int* d_next_num_active, 
    int** d_active_block_lists,
	int* d_jobs_claimed,
	int* d_iteration,
	int blocks_launched,
    cudaGraphConditionalHandle handle,
    unsigned long long* d_start_time) 
{
    // Transfer next queue size to current queue size
    *d_num_active = *d_next_num_active;
    
    // Swap the ping-pong index (0 -> 1, 1 -> 0)
    int *ping_pong = d_active_block_lists[0];
	d_active_block_lists[0] = d_active_block_lists[1];
	d_active_block_lists[1] = ping_pong;
	
	// Reset the fetch counter for the next loop.
	*d_jobs_claimed = blocks_launched;
	
	// Count number of iterations.
    int iterations = *d_iteration;
	*d_iteration = iterations + 1;

	
#if __CUDA_ARCH__ >= 900 && CUDA_HAS_CONDITIONAL_GRAPHS
    // The compiler will ONLY parse this block if compiling for SM 9.0 (Hopper) or higher.

    // Calculate elapsed time.
    unsigned long long current_time = get_global_time();
    
    if(iterations == 0)
        *d_start_time = current_time;  // Start timer.
    
    // 2 seconds in nanoseconds.
    const unsigned long long TIMEOUT_NS = 2ULL * 1000ULL * 1000ULL * 1000ULL; 
	unsigned long long elapsed_ns = current_time - *d_start_time;
    bool is_timeout = (elapsed_ns >= TIMEOUT_NS);

	// Update the loop condition for the Graph hardware.
	if (handle) {
		cudaGraphSetConditional(handle, ((*d_next_num_active > 0) & !is_timeout) ? 1 : 0);
	}
#endif
}


// =========================================================
// EIKONAL EQUATION UPDATE FUNCTIONS
// =========================================================

struct AdjointWeights {
    float inv_normalizer;
    float mx, px; // -X, +X
    float my, py; // -Y, +Y
    float mz, pz; // -Z, +Z
    float diag[12]; // MSFM Face-Diagonals.
};


template<bool MSFM = false, bool THREE_DIMENSIONAL = false, bool GATED_X = false, typename FetchCard, typename FetchDiag>
__device__ __forceinline__ AdjointWeights precompute_adjoint_weights(
    FetchCard fetch_card, FetchDiag fetch_diag, int ch, float T_curr, float dx) 
{
    AdjointWeights w = {0.0f};
	float normalizer = 0.0f; // Local accumulator.

    // GUARD: If this voxel was unreachable in the forward pass, skip it.
    if (T_curr >= INFINITY_PLACEHOLDER) return w;
    
    // Fetch T for all neighbors.
	float T_mx = fetch_card(0, 0, ch); // -X
    float T_px = fetch_card(0, 1, ch); // +X
    float T_my = fetch_card(1, 0, ch); // -Y
    float T_py = fetch_card(1, 1, ch); // +Y

    // =========================================================
    // 1. NORMALIZER (Outgoing Information Flux)
    // ADDITIVE ADJOINT: Based on the derivative of the upwind Eikonal equation.
    // =========================================================
	normalizer += fmaxf(T_curr - T_mx, 0.0f);
    if constexpr (!GATED_X) normalizer += fmaxf(T_curr - T_px, 0.0f);
    normalizer += fmaxf(T_curr - T_my, 0.0f);
    normalizer += fmaxf(T_curr - T_py, 0.0f);

    /*
    float u_x = fminf(T_mx, T_px);
    if constexpr (GATED_X) u_x = T_mx;
    float u_y = fminf(T_my, T_py);
    
    normalizer += fmaxf(T_curr - u_x, 0.0f);
    normalizer += fmaxf(T_curr - u_y, 0.0f);
    
    if constexpr (THREE_DIMENSIONAL) {
        float T_mz = fetch_card(2, 0, ch); // -Z
        float T_pz = fetch_card(2, 1, ch); // +Z
        float u_z = fminf(T_mz, T_pz);
        normalizer += fmaxf(T_curr - u_z, 0.0f);
    }*/

    // =========================================================
    // 2. INCOMING FLUX WEIGHTS
    // =========================================================
    // eps breaks cyclic dependencies caused by floating point noise.
    // If dx is available, scaling this (e.g., 1e-4f * dx) is even more robust.
    const float eps = 0; //1e-4f * dx; 

    // A downstream neighbor contributes if its T is strictly greater than T_curr + eps.
    if constexpr (!GATED_X) w.mx = (T_mx < INFINITY_PLACEHOLDER) ? fmaxf(T_mx - T_curr - eps, 0.0f) : 0.0f;
    w.px = (T_px < INFINITY_PLACEHOLDER) ? fmaxf(T_px - T_curr - eps, 0.0f) : 0.0f;
    w.my = (T_my < INFINITY_PLACEHOLDER) ? fmaxf(T_my - T_curr - eps, 0.0f) : 0.0f;
    w.py = (T_py < INFINITY_PLACEHOLDER) ? fmaxf(T_py - T_curr - eps, 0.0f) : 0.0f;

    if constexpr (THREE_DIMENSIONAL) {
        float T_mz = fetch_card(2, 0, ch); // -Z
        float T_pz = fetch_card(2, 1, ch); // +Z

        w.mz = (T_mz < INFINITY_PLACEHOLDER) ? fmaxf(T_mz - T_curr - eps, 0.0f) : 0.0f;
        w.pz = (T_pz < INFINITY_PLACEHOLDER) ? fmaxf(T_pz - T_curr - eps, 0.0f) : 0.0f;
    }
    
    if constexpr (MSFM) {
        // All 12 of the diagonals are face diagonals (distance = sqrt(2)*dx)
        // Therefore, distance squared = 2 * dx^2.
        float geom_scale = 0.70710678f; // Geometric scaling for diagonals.
        const int corners = THREE_DIMENSIONAL ? 12 : 4;
        
        #pragma unroll
        for(int d = 0; d < corners; ++d) {
            float T_d = fetch_diag(d, ch);
            
            bool block_outgoing = false;
            bool block_incoming = false;
            
            if constexpr (GATED_X) {
                // Diagonals involving +X: 1 (+X,-Y), 3 (+X,+Y), 5 (+X,-Z), 7 (+X,+Z)
                if (d == 1 || d == 3 || d == 5 || d == 7) block_outgoing = true;
                
                // Diagonals involving -X: 0 (-X,-Y), 2 (-X,+Y), 4 (-X,-Z), 6 (-X,+Z)
                if (d == 0 || d == 2 || d == 4 || d == 6) block_incoming = true;
            }

            // Add to normalizer (outgoing flux)
            if (!block_outgoing)
                normalizer += fmaxf(T_curr - T_d, 0.0f) * geom_scale;
            
            // Incoming flux weight
            if (!block_incoming)
                w.diag[d] = (T_d < INFINITY_PLACEHOLDER) ? fmaxf(T_d - T_curr - eps, 0.0f) * geom_scale : 0.0f;
            else
                w.diag[d] = 0.0f; // Force to 0 for gated directions
        }
    }
    
    // If norm is too small (probably a source pixel), store 1.0f. Otherwise, store the reciprocal.
    /*if (normalizer > 1e-7f) {
        w.inv_normalizer = 1.0f / normalizer;
    } else {
        w.inv_normalizer = 1.0f; 
    }*/

    // Epsilon scaled to your grid to prevent division by zero.
    // Using 1e-4f to 1e-5f is usually safe for single-precision adjoint Eikonal solvers
    //const float reg_eps = 1e-5f * dx;
    
    // Continuous Tikhonov regularization.
    //w.inv_normalizer = normalizer / (normalizer * normalizer + reg_eps * reg_eps);

    // =========================================================
    // REGULARIZATION & SINK HANDLING
    // =========================================================
    // If normalizer is approximately 0, this pixel is a local minimum (a source).
    // Source points are terminal sinks in the adjoint pass. The gradient w.r.t 
    // the initial condition (u_init) is simply the numerator: (R_curr + flux_in).
    // By setting inv_normalizer = 1.0f, solve_adjoint will return exactly that.
    
    if (normalizer > 1e-6f) {
        // Standard PDE grid point
        w.inv_normalizer = 1.0f / normalizer;
    } else {
        // Terminal Sink / Source Point
        w.inv_normalizer = 1.0f; 
    }

    // Absolute minimums (source pixels) act as the drain.
    return w;
}

template<bool MSFM = false, bool THREE_DIMENSIONAL = false, typename FetchCard, typename FetchDiag>
__device__ __forceinline__ float solve_adjoint(
    FetchCard fetch_card, FetchDiag fetch_diag, int ch, const AdjointWeights& w, float R_curr) 
{
    
    float flux_in = 0.0f;
    
    // Gather flux from downstream neighbors.
	flux_in += fetch_card(0, 0, ch) * w.mx;
    flux_in += fetch_card(0, 1, ch) * w.px;
    flux_in += fetch_card(1, 0, ch) * w.my;
    flux_in += fetch_card(1, 1, ch) * w.py;
    
    if constexpr (THREE_DIMENSIONAL) {
        flux_in += fetch_card(2, 0, ch) * w.mz;
        flux_in += fetch_card(2, 1, ch) * w.pz;
    }
	
	if constexpr (MSFM) {
		const int corners = THREE_DIMENSIONAL ? 12 : 4;
        #pragma unroll
        for(int d = 0; d < corners; ++d) {
            // Fetch neighbor's lambda.
			flux_in += fetch_diag(d, ch) * w.diag[d];
        }
    }
	
    // Divide total residual + flux by the normalizer.
    return (R_curr + flux_in) * w.inv_normalizer; 
}

// Auxiliary function to solve the 2D Eikonal update for any orthogonal pair
__device__ __forceinline__ float solve_directional_update(float val1, float val2, float step, float step_sq2) {
    float res = fminf(val1, val2) + step;
    float diff = fabsf(val1 - val2);  // Use 1e9 in place of inf to avoid NaN here.
    
    if (diff < step)
        res = 0.5f * (val1 + val2 + sqrtf(step_sq2 - diff * diff));
    return res;
}

// Auxiliary function to solve the 3D Eikonal update.
__device__ __forceinline__ float solve_directional_update_3d(float v1, float v2, float v3, float step, float step_sq2) {
    // Branchless sort using pure min/max logic (100% safe for INFINITY)
    float a = fminf(v1, fminf(v2, v3));
    float c = fmaxf(v1, fmaxf(v2, v3));
    
    // Median of 3 trick: avoids generating NaN during the sort
    float b = fmaxf(fminf(v1, v2), fminf(fmaxf(v1, v2), v3)); 

    // Attempt the 3D quadratic update
    float sum = a + b + c;
    float sum_sq = a * a + b * b + c * c;
    
    // If 'c' is INFINITY, this equation becomes INF - INF, producing NaN
    float disc = sum * sum - 3.0f * (sum_sq - step * step);
    
    // If disc is NaN, (NaN >= 0.0f) naturally evaluates to FALSE!
    // It silently and safely skips the 3D block.
    if (disc >= 0.0f) {
        float res = (sum + sqrtf(disc)) * 0.333333333f; 
        
        if (res >= c)
            return res;
    }
    
    // If the 3D update fails (or was skipped by NaN), fall back to the 2D update.
    // 'a' and 'b' are guaranteed to be correctly sorted, finite values (or safe Infinities).
    return solve_directional_update(a, b, step, step_sq2);
}

// Auxiliary function to solve the 3D MSFM Anisotropic update
// Combines 2 planar diagonals (v_diag1, v_diag2) with 1 perpendicular cardinal (v_card)
__device__ __forceinline__ float solve_anisotropic_update_3d(float v_diag1, float v_diag2, float v_card, float fdx_sq2) {
    float S = v_diag1 + v_diag2 + 2.0f * v_card;
    float Q = v_diag1 * v_diag1 + v_diag2 * v_diag2 + 2.0f * v_card * v_card;
    
    // Note: fdx_sq2 is defined as (2.0f * dx^2). 
    // The formula requires 8 * dx^2, which is exactly 4.0f * fdx_sq2.
    float disc = S * S - 4.0f * Q + 4.0f * fdx_sq2;
    
    // If disc is NaN, this safely evaluates to false
    if (disc >= 0.0f) {
        float res = (S + sqrtf(disc)) * 0.25f;
        
        // Characteristic validity: The result must strictly bound all participating values.
        if (res >= v_diag1 && res >= v_diag2 && res >= v_card) {
            return res;
        }
    }
    
    // If the 3D characteristic is invalid or skipped, return "infinity".
    // The fminf pool in the main kernel will naturally select a valid 2D/1D fallback.
    return INFINITY_PLACEHOLDER; 
}

template<bool msfm = false, bool three_dimensional = false, int CHANNELS_V = 0, bool GATED_X = false, typename Vtype, typename FetchCard, typename FetchDiag>
__device__ __forceinline__ float solve_eikonal(FetchCard& fetch_card, FetchDiag& fetch_diag, int ch, float fdx, Vtype& calculated_v) {
    
	// The first argument to fetch_card is dimension, 1: X, 2: Y, 3: Z. Its second argument is direction: +- 1.
	// The first argument to fetch_diag is corner, where 0-3 are the 2D square's corners and 0-11 are the 3D cube's corners.
	
	// -----------------------------------------------------------------
    // Cardinal Update (Standard, producing an overestimate)
    // -----------------------------------------------------------------
    float u_x;
    if constexpr (GATED_X) {
        // GATING: If gated, we only accept the -X (left) neighbor. 
        // We ignore +X (right) to force strictly +X propagation.
        u_x = fetch_card(0, 0, ch);
    } else {
        u_x = fminf(fetch_card(0, 0, ch), fetch_card(0, 1, ch)); // left, right
    }
    float u_y = fminf(fetch_card(1, 0, ch), fetch_card(1, 1, ch)); // up, down
	float u_z = INFINITY_PLACEHOLDER;  // Default to inf for 2D.
	
	float fdxsq2 = 2.0f * fdx * fdx;
    float res_c;
	if constexpr(!three_dimensional) {
		res_c = solve_directional_update(u_x, u_y, fdx, fdxsq2);
	} else {
		u_z = fminf(fetch_card(2, 0, ch), fetch_card(2, 1, ch)); // front, back
		res_c = solve_directional_update_3d(u_x, u_y, u_z, fdx, fdxsq2);
	}
    
	// -----------------------------------------------------------------
    // Diagonal Update (MSFM, more accurate, producing an overestimate)
    // -----------------------------------------------------------------
	float res_T = res_c;
	if constexpr (msfm) {
		float res_d;

		if constexpr(!three_dimensional) {
			// Standard 2D MSFM (XY plane only)
			// GATING: 3 is (+X, +Y) and 1 is (+X, -Y). Block both if gated.
            float u_d1 = GATED_X ? fetch_diag(0, ch) : fminf(fetch_diag(0, ch), fetch_diag(3, ch)); // ul vs dr
            float u_d2 = GATED_X ? fetch_diag(2, ch) : fminf(fetch_diag(1, ch), fetch_diag(2, ch)); // ur vs dl
			float h_diag = 1.41421356f * fdx; // sqrt(2) * dx
			
			res_d = solve_directional_update(u_d1, u_d2, h_diag, 2.0f * fdxsq2);
		} else {
			float h_diag = 1.41421356f * fdx;
			float h_diag_sq2 = 2.0f * fdxsq2; 

			// XY Plane Diagonals + Perpendicular Z Cardinal
			float u_xy_1 = GATED_X ? fetch_diag(0, ch) : fminf(fetch_diag(0, ch), fetch_diag(3, ch));
            float u_xy_2 = GATED_X ? fetch_diag(2, ch) : fminf(fetch_diag(1, ch), fetch_diag(2, ch));
			float res_xy   = solve_directional_update(u_xy_1, u_xy_2, h_diag, h_diag_sq2);
			float res_3d_z = solve_anisotropic_update_3d(u_xy_1, u_xy_2, u_z, fdxsq2);

			// XZ Plane Diagonals + Perpendicular Y Cardinal
            // 7 is (+X, +Z) and 5 is (+X, -Z). Block both if gated.
			float u_xz_1 = GATED_X ? fetch_diag(4, ch) : fminf(fetch_diag(4, ch), fetch_diag(7, ch));
            float u_xz_2 = GATED_X ? fetch_diag(6, ch) : fminf(fetch_diag(5, ch), fetch_diag(6, ch));
			float res_xz   = solve_directional_update(u_xz_1, u_xz_2, h_diag, h_diag_sq2);
			float res_3d_y = solve_anisotropic_update_3d(u_xz_1, u_xz_2, u_y, fdxsq2);

			// YZ Plane Diagonals + Perpendicular X Cardinal
			float u_yz_1 = fminf(fetch_diag(8, ch), fetch_diag(11, ch));
			float u_yz_2 = fminf(fetch_diag(9, ch), fetch_diag(10, ch));
			float res_yz   = solve_directional_update(u_yz_1, u_yz_2, h_diag, h_diag_sq2);
			float res_3d_x = solve_anisotropic_update_3d(u_yz_1, u_yz_2, u_x, fdxsq2);

			// Pool everything: planar MSFM updates and 3D Anisotropic MSFM updates
			float min_planar = fminf(res_xy, fminf(res_xz, res_yz));
			float min_3d     = fminf(res_3d_z, fminf(res_3d_y, res_3d_x));
			res_d = fminf(min_planar, min_3d);
		}
		// Return the minimum of the cardinal and diagonal stencils
		res_T = fminf(res_c, res_d);
	}
	
	// -----------------------------------------------------------------
    // Eulerian Advection (Post-Update Weighting)
    // -----------------------------------------------------------------
	if constexpr(CHANNELS_V > 0) {
		float total_w = 0.0f;
		
		// Ensure calculated_v is zeroed before accumulating.
        for (int i = 0; i < CHANNELS_V; i++)
            calculated_v.f[i] = 0.0f;
		
		// Lambda to safely accumulate upwind vector contributions.
		auto accumulate_upwind = [&](float T_neighbor, auto fetch_v_func, float spatial_weight = 1.0f) {
			if (T_neighbor < res_T) {  //  && T_neighbor != INFINITY_PLACEHOLDER
				// No inv_fdx_sq needed. Just the Delta T * relative distance factor.
				float w = (res_T - T_neighbor) * spatial_weight;
				total_w += w;
				for (int i = 0; i < CHANNELS_V; i++)
					calculated_v.f[i] += w * fetch_v_func(i);
			}
		};
		
		// Evaluate Cardinal Neighbors.
		accumulate_upwind(fetch_card(0, 0, ch), [&](int i){ return fetch_card(0, 0, i, true); });
        if constexpr (!GATED_X) accumulate_upwind(fetch_card(0, 1, ch), [&](int i){ return fetch_card(0, 1, i, true); });
        accumulate_upwind(fetch_card(1, 0, ch), [&](int i){ return fetch_card(1, 0, i, true); });
        accumulate_upwind(fetch_card(1, 1, ch), [&](int i){ return fetch_card(1, 1, i, true); });
		
		if constexpr (three_dimensional) {
			accumulate_upwind(fetch_card(2, 0, ch), [&](int i){ return fetch_card(2, 0, i, true); });
            accumulate_upwind(fetch_card(2, 1, ch), [&](int i){ return fetch_card(2, 1, i, true); });
		}
		
		// Evaluate MSFM Diagonal Neighbors.
		if constexpr (msfm) {
			accumulate_upwind(fetch_diag(0, ch), [&](int i){ return fetch_diag(0, i, true); }, 0.5f);
            if constexpr (!GATED_X) accumulate_upwind(fetch_diag(1, ch), [&](int i){ return fetch_diag(1, i, true); }, 0.5f);
            accumulate_upwind(fetch_diag(2, ch), [&](int i){ return fetch_diag(2, i, true); }, 0.5f);
            if constexpr (!GATED_X) accumulate_upwind(fetch_diag(3, ch), [&](int i){ return fetch_diag(3, i, true); }, 0.5f);
			
			if constexpr (three_dimensional) {
                accumulate_upwind(fetch_diag(4, ch), [&](int i){ return fetch_diag(4, i, true); }, 0.5f);
                if constexpr (!GATED_X) accumulate_upwind(fetch_diag(5, ch), [&](int i){ return fetch_diag(5, i, true); }, 0.5f);
                accumulate_upwind(fetch_diag(6, ch), [&](int i){ return fetch_diag(6, i, true); }, 0.5f);
                if constexpr (!GATED_X) accumulate_upwind(fetch_diag(7, ch), [&](int i){ return fetch_diag(7, i, true); }, 0.5f);
                
                // YZ Plane (No X-components, always included)
                accumulate_upwind(fetch_diag(8, ch), [&](int i){ return fetch_diag(8, i, true); }, 0.5f);
                accumulate_upwind(fetch_diag(9, ch), [&](int i){ return fetch_diag(9, i, true); }, 0.5f);
                accumulate_upwind(fetch_diag(10, ch), [&](int i){ return fetch_diag(10, i, true); }, 0.5f);
                accumulate_upwind(fetch_diag(11, ch), [&](int i){ return fetch_diag(11, i, true); }, 0.5f);
            }
		}

        float inv_w = (total_w > 1e-30f) ? 1.0f / total_w : 0.0f;

        // Standard Advection Normalization (Divide by total weight)
	    for(int i = 0; i < CHANNELS_V; ++i)
		    calculated_v.f[i] *= inv_w;
	}
	
	return res_T;
}


// =========================================================
// MAIN KERNEL
// =========================================================
template <
    int TILE_W, int TILE_H, int TILE_D,
    int NX, int NY, int NZ,
    int CHANNELS, int CHANNELS_F, int CHANNELS_V_in,
    bool THREE_DIMENSIONAL,
    bool MSFM, int MAX_LOCAL_ITERS, bool DOUBLE_BUFFERED_SMEM,
    bool BACKWARD_PASS, bool GATED_X
>
__launch_bounds__(TILE_W*TILE_H*TILE_D)
__global__ void batched_fim_kernel_new(
	void* __restrict__ d_u, size_t pitch_u_bytes,
	const void* __restrict__ d_f, size_t pitch_f_bytes,
	int** __restrict__ d_active_block_lists, // Array of 2 pointers [List A, List B]
	const int* __restrict__ d_num_active_blocks,   // Pointer to dynamic size
	int* __restrict__ d_next_num_active_blocks, int* __restrict__ d_is_enqueued,
	int* __restrict__ d_jobs_claimed,
	const int* __restrict__ d_iteration,
	int grid_width, int grid_height, int grid_depth, float dx,
	int blocks_per_row, int blocks_per_col, int blocks_per_slice, int blocks_per_batch, bool broadcast_batch_f,
	const void* __restrict__ d_tof, size_t pitch_tof_bytes,  // For time-of-flight (d_u) during a backwards pass.
	void* __restrict__ d_v, size_t pitch_v_bytes  // For directional field in forward pass.
) {
	
	constexpr int CHANNELS_V = BACKWARD_PASS ? 0 : CHANNELS_V_in;
	// Alias the constants locally.
	constexpr int HALO = 1;
    constexpr int EFF_W = TILE_W * NX;
    constexpr int EFF_H = TILE_H * NY;
    constexpr int EFF_D = TILE_D * NZ;
    constexpr int SMEM_W = EFF_W + 2 * HALO;
    constexpr int SMEM_H = EFF_H + 2 * HALO;
    constexpr int SMEM_D = EFF_D + 2 * HALO;
    constexpr int THREADS_PER_BLOCK = TILE_W * TILE_H * (THREE_DIMENSIONAL ? TILE_D : 1);
	constexpr int CHANNELS_V_CLAMPED = CHANNELS_V > 1 ? CHANNELS_V : 1;
	static_assert(THREE_DIMENSIONAL || (NZ == 1 && TILE_D == 1));  // If 2D, Nz and depth must be 1.
	static_assert(CHANNELS_F <= CHANNELS);  // Since u contains the input and output, we cannot have more F channels than U channels
	static_assert(CHANNELS_V == 0 || !BACKWARD_PASS);  // Backward pass not implemented for CHANNELS_V > 0.
	
	// Map the CHANNELS integer to CUDA vector type (1 = float, 2 = float2, 4 = float4).
	using VecType = typename CudaVec<CHANNELS>::Type;
	using VecTypeV = typename CudaVec<CHANNELS_V_CLAMPED>::Type; // static_assert(CHANNELS_V == 0 || CHANNELS = 1);  // We do not support multiple channels currently if CHANNELS_V > 1.
	using VecTypeF = typename CudaVec<CHANNELS_F>::Type;
	
	// 2D Standard: 4 Cardinal + 1 Self = 5
	// 2D MSFM: 4 Cardinal + 4 Corners + 1 Self = 9
	// 3D Standard: 6 Cardinal + 1 Self = 7
	// 3D MSFM: 6 Cardinal + 12 Edges + 8 Corners + 1 Self = 27
	constexpr int Nborders = 27;
	__shared__ bool s_needs_neighbor[Nborders]; // 0: Left, 1: Right, 2: Up, 3: Down, 4: Front, 5: Back, ..., 26: Self.
	
	// Shared memory layout.
	__shared__ int s_job_idx;
	__shared__ int s_data_block_id;
	__shared__ int s_iteration;
	__shared__ int s_num_active_blocks;
	__shared__ int s_batch_id;  // Cache in shared memory.
	alignas(sizeof(float) * CHANNELS) __shared__ float s_u_flat[1+DOUBLE_BUFFERED_SMEM][SMEM_D][SMEM_H][SMEM_W][CHANNELS];  
	[[maybe_unused]] alignas(sizeof(float) * CHANNELS_V_CLAMPED) __shared__ float s_v_flat[1+DOUBLE_BUFFERED_SMEM][SMEM_D][SMEM_H][SMEM_W][CHANNELS_V_CLAMPED];
	#define SU(PAGE, CH, Z, Y, X) s_u_flat[(PAGE)][(Z)][(Y)][(X)][(CH)]
	#define SV(PAGE, CH, Z, Y, X) s_v_flat[(PAGE)][(Z)][(Y)][(X)][(CH)]
	// No need to actually double buffer the smem as a chaotic reading order is not hurtful and will in fact only accelerate the convergence.
	
	// Thread identification.
	int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	if constexpr (!THREE_DIMENSIONAL) {
		// 2D thread identification.
		tid = threadIdx.y * blockDim.x + threadIdx.x;
		grid_depth = 1;
	}

	// Get current job list.
	const int* __restrict__ active_block_list = d_active_block_lists[0];
	int* __restrict__ next_active_block_list = d_active_block_lists[1];
	
	int job_idx = blockIdx.x;
	if(tid == 0) {
		s_iteration = *d_iteration;  // Move to shared memory for faster loading (but we dont want to "waste" registers holding it!).
		s_num_active_blocks = *d_num_active_blocks;
        if(job_idx < s_num_active_blocks) s_data_block_id = active_block_list[job_idx];
	}
	__syncthreads();
	
    // Thread 0..Nborders-1 initalizes these flags.
	if (tid < Nborders)
		s_needs_neighbor[tid] = false;
	
	// Per-thread register caches for stencil fields
	alignas(sizeof(float) * CHANNELS)   float my_u[NZ][NY][NX][CHANNELS];
	alignas(sizeof(float) * CHANNELS_F) float my_f[NZ][NY][NX][CHANNELS_F];
	alignas(sizeof(float) * CHANNELS_V_CLAMPED) float my_v[NZ][NY][NX][CHANNELS_V_CLAMPED];
	
	// Cardinal Offsets: [Axis][Direction]
	// Axis 0 is X, Axis 1 is Y. Direction 0 is negative, Direction 1 is positive.
	constexpr int dX[3][2] = {{-1, 1}, {0, 0}, {0, 0}};
	constexpr int dY[3][2] = {{0, 0}, {-1, 1}, {0, 0}};
	constexpr int dZ[3][2] = {{0, 0}, {0, 0}, {-1, 1}};
	
	// Diagonal Offsets (4 corners in 2D, 12 corners in 3D)
	const int dX_diag[12] = {-1, 1, -1, 1,  -1, 1, -1, 1,   0, 0, 0, 0};
	const int dY_diag[12] = {-1, -1, 1, 1,   0, 0, 0, 0,  -1, 1, -1, 1};
	const int dZ_diag[12] = { 0, 0, 0, 0,   -1,-1, 1, 1,  -1,-1, 1, 1};
	
		
	// Loop until the queue is empty.
	int data_block_id = s_data_block_id;
	while(job_idx < s_num_active_blocks) {
		// Decode the Batch ID and Local Block ID
		int batch_id = data_block_id / blocks_per_batch;
		int local_block_id = data_block_id - batch_id * blocks_per_batch; // = data_block_id % blocks_per_batch;
		// TODO: Let the job list contain the x, y, z, and b positions, so we don't have to spend time computing it.
		
		if(tid == 0)
			s_batch_id = batch_id;  // Cache in shared memory to free registers.

		int block_start_x = (local_block_id % blocks_per_row) * EFF_W;
		int block_start_y = (THREE_DIMENSIONAL ? (local_block_id / blocks_per_row) % blocks_per_col : local_block_id / blocks_per_row) * EFF_H;
		int block_start_z = (THREE_DIMENSIONAL ? (local_block_id / (blocks_per_row * blocks_per_col)) * EFF_D : 0);
		
		// Base coordinates LOCAL to the specific batch's domain (this thread's Nx * Ny sub-block)
		int base_gx = block_start_x + get_tid_x() * NX;
		int base_gy = block_start_y + get_tid_y() * NY;
		int base_gz = THREE_DIMENSIONAL ? block_start_z + get_tid_z() * NZ : 0; // Assuming threadIdx.z maps to tz
		int base_sx = get_tid_x() * NX + HALO; 
		int base_sy = get_tid_y() * NY + HALO;
		int base_sz = THREE_DIMENSIONAL ? get_tid_z() * NZ + HALO : 0;
		
		// Precompute boundary validity into a single 32-bit register.
		uint32_t valid_mask = 0;
		constexpr float oob_val = BACKWARD_PASS ? 0.0f : INFINITY_PLACEHOLDER;
		#pragma unroll
		for (int iz = 0; iz < NZ; ++iz)
		for (int iy = 0; iy < NY; ++iy)
		for (int ix = 0; ix < NX; ++ix) {
			int gx = base_gx + ix;
			int gy = base_gy + iy;
			int gz = THREE_DIMENSIONAL ? base_gz + iz : 0;
			
			if (gx < grid_width && gy < grid_height && (!THREE_DIMENSIONAL || gz < grid_depth)) {
				[[maybe_unused]] int flat_idx = iz * NX * NY + iy * NX + ix;
				valid_mask |= (1U << flat_idx);
			}
		}
		static_assert(NZ * NY * NX <= sizeof(valid_mask) * 8, 
										  "Per-thread stencil exceeds the bit-width of valid_mask!");
		// #define VALID_MASK (valid_mask & (1U << flat_idx))
		#define VALID_MASK (gx < grid_width && gy < grid_height && (!THREE_DIMENSIONAL || gz < grid_depth))
		
		// =================================================================
		// Load CORE into registers & shared memory
		// =================================================================
		// Thread mapping allows hardware to naturally coalesce stride-Nx accesses.
		#pragma unroll
		for (int iz = 0; iz < NZ; ++iz)
		for (int iy = 0; iy < NY; ++iy)
		for (int ix = 0; ix < NX; ++ix) {
			[[maybe_unused]] int flat_idx = iz * NX * NY + iy * NX + ix; // Evaluates at compile-time.
			int gx = base_gx + ix;
			int gy = base_gy + iy;
			int gz = THREE_DIMENSIONAL ? base_gz + iz : 0;
			
			for(int c = 0; c < CHANNELS;   ++c) my_u[iz][iy][ix][c] = oob_val; // Initialize U as "infinity".
			for(int c = 0; c < CHANNELS_V;   ++c) my_v[iz][iy][ix][c] = 0;
			for(int c = 0; c < CHANNELS_F;   ++c) my_f[iz][iy][ix][c] = 0;
			// TODO: Move infinity placeholder to config (and do the same with F_placeholder)
			
			// Load U ONLY if strictly inside the grid.
			if (VALID_MASK) {
				// For 2D, grid_depth is 1 and gz is 0, so this formula works for both!
				int global_row_u = batch_id * (grid_height * grid_depth) + gz * grid_height + gy;
				VecType core_val = ((VecType*)((char*)d_u + global_row_u * pitch_u_bytes))[gx];
				for(int c = 0; c < CHANNELS; ++c) my_u[iz][iy][ix][c] = core_val.f[c];
				
				if constexpr (CHANNELS_V > 0) {
					VecTypeV core_v = ((VecTypeV*)((char*)d_v + global_row_u * pitch_v_bytes))[gx];
					for(int c = 0; c < CHANNELS_V; ++c) my_v[iz][iy][ix][c] = core_v.f[c];
				}
				 
				if constexpr (BACKWARD_PASS) {
					// In the backwards pass, do not use clamping for F (the residual). Set out-of-bounds values to 0 instead.
					int global_row_f = (broadcast_batch_f ? 0 : batch_id * (grid_height * grid_depth)) + gz * grid_height + gy;
					VecTypeF core_f = ((VecTypeF*)((char*)d_f + global_row_f * pitch_f_bytes))[gx];
					for(int c = 0; c < CHANNELS_F; ++c)
						my_f[iz][iy][ix][c] = core_f.f[c];
				}
			}
			
			// We only have to place the perimeter in shared memory.
			bool is_perimeter = (ix == 0 || ix == NX - 1 || iy == 0 || iy == NY - 1);
			if (THREE_DIMENSIONAL) is_perimeter = is_perimeter || (iz == 0 || iz == NZ - 1);
			if (is_perimeter)
				for(int p = 0; p < DOUBLE_BUFFERED_SMEM+1; p++) {
					for(int c = 0; c < CHANNELS; ++c)
						SU(p, c, base_sz + iz, base_sy + iy, base_sx + ix) = my_u[iz][iy][ix][c];
					for(int c = 0; c < CHANNELS_V; ++c)
						SV(p, c, base_sz + iz, base_sy + iy, base_sx + ix) = my_v[iz][iy][ix][c];
				}
			
			if (BACKWARD_PASS) continue; // F is already handled above in that case.
			
			// Load F, but use the clamped coordinates (Constant/NN Extrapolation).
			int clamp_gx = min(gx, grid_width - 1);
			int clamp_gy = min(gy, grid_height - 1);
			int clamp_gz = THREE_DIMENSIONAL ? min(gz, grid_depth - 1) : 0;
			int global_row_f = (broadcast_batch_f ? 0 : batch_id * (grid_height * grid_depth)) + clamp_gz * grid_height + clamp_gy;
			
			// Also clamp slowness value to zero if negative, as the values should be strictly positive! (zero slowness corresponds to infinite speed of sound).
			VecTypeF core_f = ((VecTypeF*)((char*)d_f + global_row_f * pitch_f_bytes))[clamp_gx];
			for(int c = 0; c < CHANNELS_F; ++c) my_f[iz][iy][ix][c] = fmaxf(core_f.f[c], 0.0f);
		}
		
		// TODO: Just try loading the entire rectangle/cube into shared memory using a single flattened loop. This will require one extra LDS to get the core values afterwards.
		//       It should also be possible to load the NEXT job in the background on newer hardware, potentially saving time.
		
		// =================================================================
		// Load HALOS (Compile-time branched for maximum 2D & 3D speed
		// =================================================================
		if constexpr (!THREE_DIMENSIONAL) {
			// --- TOP and BOTTOM Halo ---
			int halo_width = EFF_W + 2 * MSFM;
			int halo_height = EFF_H + 2 * MSFM;
			int total_halo_elems = 2 * halo_width + 2 * halo_height;

			for (int i = tid; i < total_halo_elems; i += THREADS_PER_BLOCK) {
				int dest_sx, dest_sy, load_gx, load_gy;
				
				if (i < 2 * halo_width) {
					// --- TOP and BOTTOM Halo ---
					int side = i / halo_width;
					int x_offset = (i % halo_width) - MSFM; 
					
					dest_sy = (side == 0) ? 0 : EFF_H + HALO;
					dest_sx = x_offset + HALO;
					load_gx = block_start_x + dest_sx - HALO;
					load_gy = (dest_sy == 0) ? block_start_y - HALO : block_start_y + EFF_H - 1 + HALO;
				} else {
					// --- Left and Right Halo ---
					int j = i - (2 * halo_width); // Adjusted index for second half
					int side = j / halo_height;
					int y_offset = (j % halo_height) - MSFM;
					
					dest_sx = (side == 0) ? 0 : EFF_W + HALO;
					dest_sy = y_offset + HALO;
					load_gx = (dest_sx == 0) ? block_start_x - HALO : block_start_x + EFF_W - 1 + HALO;
					load_gy = block_start_y + dest_sy - HALO;
				}

				// --- Shared Load / Store ---
				int global_load_row = batch_id * (grid_height * grid_depth) + load_gy;
				
				VecType val;
				VecTypeV val_v;
				for (int c = 0; c < CHANNELS; ++c) val.f[c] = oob_val;
				for (int c = 0; c < CHANNELS_V; ++c) val_v.f[c] = 0;
				
				bool in_bounds = 0 <= load_gx && load_gx < grid_width && 0 <= load_gy && load_gy < grid_height;
				if (in_bounds) {
					val = ((const VecType*)((char*)d_u + global_load_row * pitch_u_bytes))[load_gx];
					if constexpr (CHANNELS_V > 0)
						val_v = ((VecTypeV*)((char*)d_v + global_load_row * pitch_v_bytes))[load_gx];
				}
				
				for (int p = 0; p < DOUBLE_BUFFERED_SMEM+1; p++)
					for (int c = 0; c < CHANNELS; ++c)
						SU(p, c, 0, dest_sy, dest_sx) = val.f[c];
				
				for (int p = 0; p < DOUBLE_BUFFERED_SMEM+1; p++)
					for (int c = 0; c < CHANNELS_V; ++c)
						SV(p, c, 0, dest_sy, dest_sx) = val_v.f[c];
			}
		} else {
			// --- 3D EXPLICIT LOADING (Overlapping planes to catch all corners instantly) ---
			int z_area = (EFF_W + 2 * MSFM) * (EFF_H + 2 * MSFM);
			int y_area = (EFF_W + 2 * MSFM) * EFF_D;
			int x_area = EFF_H * EFF_D;

			int total_halo_elems = 2 * z_area + 2 * y_area + 2 * x_area;

			for (int i = tid; i < total_halo_elems; i += THREADS_PER_BLOCK) {
				int load_gx, load_gy, load_gz;
				int dest_sx, dest_sy, dest_sz;

				if (i < 2 * z_area) {
					// --- Z-Faces (Front/Back) - Covers full extended XY planes ---
					int side = i / z_area; // 0: -Z, 1: +Z
					[[maybe_unused]] int flat_idx = i % z_area;
					int x_offset = (flat_idx % (EFF_W + 2 * MSFM)) - MSFM;
					int y_offset = (flat_idx / (EFF_W + 2 * MSFM)) - MSFM;
					
					load_gx = block_start_x + x_offset;
					load_gy = block_start_y + y_offset;
					load_gz = (side == 0) ? block_start_z - HALO : block_start_z + EFF_D - 1 + HALO;
					
					dest_sx = x_offset + HALO;
					dest_sy = y_offset + HALO;
					dest_sz = (side == 0) ? 0 : EFF_D + HALO;

				} else if (i < 2 * z_area + 2 * y_area) {
					// --- Y-Faces (Top/Bottom) - Covers extended X, core Z ---
					int j = i - (2 * z_area);
					int side = j / y_area; 
					[[maybe_unused]] int flat_idx = j % y_area;
					int x_offset = (flat_idx % (EFF_W + 2 * MSFM)) - MSFM;
					int z_offset = flat_idx / (EFF_W + 2 * MSFM); 
					
					load_gx = block_start_x + x_offset;
					load_gy = (side == 0) ? block_start_y - HALO : block_start_y + EFF_H - 1 + HALO;
					load_gz = block_start_z + z_offset;
					
					dest_sx = x_offset + HALO;
					dest_sy = (side == 0) ? 0 : EFF_H + HALO;
					dest_sz = z_offset + HALO;
				} else {
					// --- X-Faces (Left/Right) - Covers core Y, core Z ---
					int k = i - (2 * z_area + 2 * y_area);
					int side = k / x_area;
					[[maybe_unused]] int flat_idx = k % x_area;
					int y_offset = flat_idx % EFF_H;
					int z_offset = flat_idx / EFF_H;
					
					load_gx = (side == 0) ? block_start_x - HALO : block_start_x + EFF_W - 1 + HALO;
					load_gy = block_start_y + y_offset;
					load_gz = block_start_z + z_offset;
					
					dest_sx = (side == 0) ? 0 : EFF_W + HALO;
					dest_sy = y_offset + HALO;
					dest_sz = z_offset + HALO;
				}

				// --- Common Boundary Checks and Loading Logic ---
				bool in_bounds = (load_gx >= 0 && load_gx < grid_width && 
								  load_gy >= 0 && load_gy < grid_height && 
								  load_gz >= 0 && load_gz < grid_depth);
				
				int global_load_row = batch_id * (grid_height * grid_depth) + load_gz * grid_height + load_gy;
				VecType val;
				VecTypeV val_v;
				for (int c = 0; c < CHANNELS; ++c) val.f[c] = oob_val;
				for (int c = 0; c < CHANNELS_V_CLAMPED; ++c) val_v.f[c] = 0.0f;
				if (in_bounds) {
					val = ((const VecType*)((char*)d_u + global_load_row * pitch_u_bytes))[load_gx];
					if constexpr (CHANNELS_V > 0)
						val_v = ((const VecTypeV*)((char*)d_v + global_load_row * pitch_v_bytes))[load_gx];
				}
				float* ptr = reinterpret_cast<float*>(&val);
				
				for (int c = 0; c < CHANNELS; ++c) {
					float final_val = in_bounds ? val.f[c] : oob_val;
					SU(0, c, dest_sz, dest_sy, dest_sx) = final_val;
					if(DOUBLE_BUFFERED_SMEM) SU(1, c, dest_sz, dest_sy, dest_sx) = final_val;
				}
				
				if constexpr (CHANNELS_V > 0) {
					for (int c = 0; c < CHANNELS_V; ++c) {
						float final_v = in_bounds ? val_v.f[c] : 0.0f;
						SV(0, c, dest_sz, dest_sy, dest_sx) = final_v;
						if(DOUBLE_BUFFERED_SMEM) SV(1, c, dest_sz, dest_sy, dest_sx) = final_v;
					}
				}
			}
		}
		
		__syncthreads();  // Wait till threads have moved the data to shared memory.
		
		// Precomputation in case of a backwards pass.
		AdjointWeights my_weights[NZ][NY][NX][CHANNELS];

        // Causality mask for block queueing.
		uint32_t causal_trigger_mask = 0;
		if constexpr (!BACKWARD_PASS) {
			causal_trigger_mask = 0x3F; // 0b111111: Enable all 6 cardinal boundaries for forward pass
		}
		
		if constexpr (BACKWARD_PASS) {
			#pragma unroll
			for (int iz = 0; iz < NZ; ++iz)
			for (int iy = 0; iy < NY; ++iy)
			for (int ix = 0; ix < NX; ++ix) {
				[[maybe_unused]] int flat_idx = iz * NX * NY + iy * NX + ix;
				int gx = base_gx + ix;
				int gy = base_gy + iy;
				int gz = THREE_DIMENSIONAL ? base_gz + iz : 0;

				if (VALID_MASK) {
					
					// Localized fetcher for this specific thread's unrolled pixel
					auto fetch_T = [&](int axis, int dir, int ch, float T_c) -> float {
						int nx = gx + dX[axis][dir];
						int ny = gy + dY[axis][dir];
						int nz = THREE_DIMENSIONAL ? gz + dZ[axis][dir] : 0;
						if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height && (!THREE_DIMENSIONAL || nz < grid_depth)) {
							int row = batch_id * (grid_height * grid_depth) + nz * grid_height + ny;
							return ((const VecType*)((const char*) d_tof + row * pitch_tof_bytes))[nx].f[ch];
						}
						return T_c; // Safely zeroes out fmaxf(T - T_curr) and fmaxf(T_curr - T)
					};
					
					auto fetch_T_diag = [&](int d, int ch, float T_c) -> float {
						int nx = gx + dX_diag[d];
						int ny = gy + dY_diag[d];
						int nz = THREE_DIMENSIONAL ? gz + dZ_diag[d] : 0;
						if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height && (!THREE_DIMENSIONAL || nz < grid_depth)) {
							int row = batch_id * (grid_height * grid_depth) + nz * grid_height + ny;
							return ((const VecType*)((const char*) d_tof + row * pitch_tof_bytes))[nx].f[ch];
						}
						return T_c; // Safely zeroes out fmaxf(T - T_curr) and fmaxf(T_curr - T)
					};

					#pragma unroll
					for (int ch = 0; ch < CHANNELS; ch++) {
						// Fetch the center pixel directly
						int center_row = batch_id * (grid_height * grid_depth) + gz * grid_height + gy;
						float T_curr = ((const VecType*)((const char*) d_tof + center_row * pitch_tof_bytes))[gx].f[ch];
						
						my_weights[iz][iy][ix][ch] = precompute_adjoint_weights<MSFM, THREE_DIMENSIONAL, GATED_X>(
							[&](int a, int d, int c) { return fetch_T(a, d, c, T_curr); }, 
							[&](int d, int c) { return fetch_T_diag(d, c, T_curr); }, ch, T_curr, dx
						);

                        // Populate the causal gating mask (with MSFM support).
						// We check if information flows out of the block boundary in any valid direction.
                        const float eps = 1e-4f * dx;
                        //T_curr -= eps;
						bool sends_mx = (ix == 0)      && (T_curr - eps > fetch_T(0, 0, ch, T_curr));
						bool sends_px = (ix == NX - 1) && (T_curr - eps > fetch_T(0, 1, ch, T_curr));
						bool sends_my = (iy == 0)      && (T_curr - eps > fetch_T(1, 0, ch, T_curr));
						bool sends_py = (iy == NY - 1) && (T_curr - eps > fetch_T(1, 1, ch, T_curr));
						bool sends_mz = false, sends_pz = false;
						
						if constexpr (THREE_DIMENSIONAL) {
							sends_mz = (iz == 0)      && (T_curr - eps > fetch_T(2, 0, ch, T_curr));
							sends_pz = (iz == NZ - 1) && (T_curr - eps > fetch_T(2, 1, ch, T_curr));
						}

						if constexpr (MSFM) {
							// -X components in dX_diag: 0, 2, 4, 6
							if (ix == 0) {
								sends_mx |= (T_curr > fetch_T_diag(0, ch, T_curr)) || (T_curr > fetch_T_diag(2, ch, T_curr));
								if constexpr (THREE_DIMENSIONAL) 
									sends_mx |= (T_curr > fetch_T_diag(4, ch, T_curr)) || (T_curr > fetch_T_diag(6, ch, T_curr));
							}
							// +X components in dX_diag: 1, 3, 5, 7
							if (ix == NX - 1) {
								sends_px |= (T_curr > fetch_T_diag(1, ch, T_curr)) || (T_curr > fetch_T_diag(3, ch, T_curr));
								if constexpr (THREE_DIMENSIONAL) 
									sends_px |= (T_curr > fetch_T_diag(5, ch, T_curr)) || (T_curr > fetch_T_diag(7, ch, T_curr));
							}
							// -Y components in dY_diag: 0, 1, 8, 10
							if (iy == 0) {
								sends_my |= (T_curr > fetch_T_diag(0, ch, T_curr)) || (T_curr > fetch_T_diag(1, ch, T_curr));
								if constexpr (THREE_DIMENSIONAL) 
									sends_my |= (T_curr > fetch_T_diag(8, ch, T_curr)) || (T_curr > fetch_T_diag(10, ch, T_curr));
							}
							// +Y components in dY_diag: 2, 3, 9, 11
							if (iy == NY - 1) {
								sends_py |= (T_curr > fetch_T_diag(2, ch, T_curr)) || (T_curr > fetch_T_diag(3, ch, T_curr));
								if constexpr (THREE_DIMENSIONAL) 
									sends_py |= (T_curr > fetch_T_diag(9, ch, T_curr)) || (T_curr > fetch_T_diag(11, ch, T_curr));
							}
							if constexpr (THREE_DIMENSIONAL) {
								// -Z components in dZ_diag: 4, 5, 8, 9
								if (iz == 0) 
									sends_mz |= (T_curr > fetch_T_diag(4, ch, T_curr)) || (T_curr > fetch_T_diag(5, ch, T_curr)) || 
												(T_curr > fetch_T_diag(8, ch, T_curr)) || (T_curr > fetch_T_diag(9, ch, T_curr));
								// +Z components in dZ_diag: 6, 7, 10, 11
								if (iz == NZ - 1) 
									sends_pz |= (T_curr > fetch_T_diag(6, ch, T_curr)) || (T_curr > fetch_T_diag(7, ch, T_curr)) || 
												(T_curr > fetch_T_diag(10, ch, T_curr)) || (T_curr > fetch_T_diag(11, ch, T_curr));
							}
						}

						// Compress into the 6-bit mask
						if (sends_mx) causal_trigger_mask |= (1 << 0);
						if (sends_px) causal_trigger_mask |= (1 << 1);
						if (sends_my) causal_trigger_mask |= (1 << 2);
						if (sends_py) causal_trigger_mask |= (1 << 3);
						if constexpr (THREE_DIMENSIONAL) {
							if (sends_mz) causal_trigger_mask |= (1 << 4);
							if (sends_pz) causal_trigger_mask |= (1 << 5);
						}
					}
				}
			}
		}
	
		// Main Fast Iterative Method with Local Updates
		uint32_t update_flags = 0;  // Pack multiple booleans into this 32-bit int to conserve registers.
		//uint64_t update_flags = 0;
		
		// Diagonal neighbors.
		const int GOOD_LARGE_ITER_COUNT = max(max(max(SMEM_W,SMEM_H),SMEM_D), MAX_LOCAL_ITERS); // In the first iteration, we want information to propagate all the way from one side of the block-wide stencil to the other.
		const int local_iters = s_iteration > 0 ? MAX_LOCAL_ITERS : GOOD_LARGE_ITER_COUNT;
		int thread_converged = 1;
		for (int iter = 0; iter < local_iters; iter++) {
			int r = iter & 1;
			int w = 1 - r;
			if constexpr (!DOUBLE_BUFFERED_SMEM) {
				// When only one page exists, enable chaotic Gauss-Seidel updates, wherein a thread might read an updated pixel value before it has been "officially" published by syncthreads below.
				// The result is non-determinstic code. However, since the algorithm is monotonic, this only speeds up convergence, and the final result when fully converged is still deterministic.
				r = 0;
				w = 0;
			}
			
			#pragma unroll
			for (int iz = 0; iz < NZ; ++iz)
			for (int iy = 0; iy < NY; ++iy)
			for (int ix = 0; ix < NX; ++ix) {
				[[maybe_unused]] int flat_idx = iz * NX * NY + iy * NX + ix;
				int gx = base_gx + ix;
				int gy = base_gy + iy;
				int gz = THREE_DIMENSIONAL ? base_gz + iz : 0;
				int sx = base_sx + ix;
				int sy = base_sy + iy;
				int sz = THREE_DIMENSIONAL ? base_sz + iz : 0;
				
				// Lambda to fetch Cardinal Neighbors for U
				auto fetch_card = [&](int axis, int dir, int ch, bool get_V = false) -> float {
					int nx = ix + dX[axis][dir];
					int ny = iy + dY[axis][dir];
					int nz = THREE_DIMENSIONAL ? iz + dZ[axis][dir] : 0;
					bool is_in_register = (0 <= nx && nx < NX && 0 <= ny && ny < NY && (!THREE_DIMENSIONAL || (0 <= nz && nz < NZ)));
					// ^ Evaluated at compile-time.
					
					if(get_V)
						if (is_in_register)
							return my_v[nz][ny][nx][ch];
						else
							return SV(r, ch, sz + dZ[axis][dir], sy + dY[axis][dir], sx + dX[axis][dir]);
					else
						if (is_in_register)
							return my_u[nz][ny][nx][ch];
						else
							return SU(r, ch, sz + dZ[axis][dir], sy + dY[axis][dir], sx + dX[axis][dir]);
				};

				// Lambda to fetch Diagonal Neighbors
				auto fetch_diag = [&](int d, int ch, bool get_V = false) -> float {
					int nx = ix + dX_diag[d];
					int ny = iy + dY_diag[d];
					int nz = THREE_DIMENSIONAL ? iz + dZ_diag[d] : 0;
					bool is_in_register = (0 <= nx && nx < NX && 0 <= ny && ny < NY && (!THREE_DIMENSIONAL || (0 <= nz && nz < NZ)));
					
					if(get_V)
						if (is_in_register)
							return my_v[nz][ny][nx][ch];
						else
							return SV(r, ch, sz + dZ_diag[d], sy + dY_diag[d], sx + dX_diag[d]);
					else
						if (is_in_register)
							return my_u[nz][ny][nx][ch];
						else
							return SU(r, ch, sz + dZ_diag[d], sy + dY_diag[d], sx + dX_diag[d]);
				};
				
				if (VALID_MASK) {
					#pragma unroll
					for (int ch = 0; ch < CHANNELS; ch++) {
						
						// Update own register.
						float calculated_u, diff;
						VecTypeV calculated_v = {0};
						bool updated;
						if constexpr (!BACKWARD_PASS) {
							// Standard FIM
							calculated_u = solve_eikonal<MSFM, THREE_DIMENSIONAL, CHANNELS_V, GATED_X>(
								fetch_card, fetch_diag, ch,
								my_f[iz][iy][ix][min(ch, CHANNELS_F-1)] * dx,
								calculated_v
							);
							
							// Different possible criteria:
							// 1. updated = calculated_u < my_u[iz][iy][ix][ch];       // Absolute absolute.
							// 2. updated = diff > 1e-5f * dx;                         // Absolute w.r.t. grid spacing.
                            // 3. updated = diff/(fabsf(calculated_u) + 1e-9f) > 1e-5f // Relative, i.e.: dx/(|u|+eps) > 0.001, where eps prevents division by zero.
                            // 4. updated = diff > fmaxf(1e-6f * dx, fabsf(1e-5f * calculated_u));  // Absolute and relative: (diff > 1e-6f * dx) && (diff > 1e-5f * |u|) (relative for large u, absolute for small u).
							
							diff = my_u[iz][iy][ix][ch] - calculated_u;
							updated = calculated_u < my_u[iz][iy][ix][ch];  // Perfectly fine stopping criteria since everything decreases monotonically. This makes the code extremely robust and virtually deterministic (even if the path taken to the result is non-deterministic).
							my_u[iz][iy][ix][ch] = fminf(my_u[iz][iy][ix][ch], calculated_u);
						} else {
							// For backward pass, where my_f acts as the residual map (R).
							calculated_u = solve_adjoint<MSFM, THREE_DIMENSIONAL>(
								fetch_card, fetch_diag, 
								ch, my_weights[iz][iy][ix][ch], 
								my_f[iz][iy][ix][min(ch, CHANNELS_F-1)] * dx * dx
							);

                            // Instead of: my_u[iz][iy][ix][ch] = calculated_u;
                            diff = fabsf(my_u[iz][iy][ix][ch] - calculated_u);
                            updated = diff > fmaxf(1e-5f * dx, fabsf(1e-4f * calculated_u)); // fmaxf(1e-6f * dx, fabsf(1e-5f * calculated_u));

                            // The backwared system has a spectral radius of 1, meaning numerical noise might bounce around making it appear non-convergent.
                            // Dampening (under-relaxation), puts the spectral radius slightly below 1, ensuring convergence, without affecting the final output much.
                            
                            constexpr float omega = 0.8f; // Use a damping factor (omega between 0.8 and 0.95).
                            calculated_u = omega * calculated_u + (1.0f - omega) * my_u[iz][iy][ix][ch];
                            
							my_u[iz][iy][ix][ch] = calculated_u;  // Lambda may fluctuate during the backwards pass.

                            // FIXME: Implement another algorithm for the backwards pass. This algorithm is not well suited, unfortunately.
						}
						
						/*
						if constexpr (BACKWARD_PASS) {
							// Write, but do not enqueue.
							// my_u[iz][iy][ix][ch] = calculated_u;
							int flag_idx = iz * NX * NY + iy * NX + ix + 8;
							update_flags |= (1U << flag_idx);
							update_flags |= (1 << 6); // thread_updated_ever
							// updated = false;
							if(iter == 2)
								updated = false;
						}*/
						
						if (updated) {
							thread_converged = 0;
							if (ix == 0)      update_flags |= (causal_trigger_mask & (1 << 0)); // -X
							if (ix == NX - 1) update_flags |= (causal_trigger_mask & (1 << 1)); // +X
							if (iy == 0)      update_flags |= (causal_trigger_mask & (1 << 2)); // -Y
							if (iy == NY - 1) update_flags |= (causal_trigger_mask & (1 << 3)); // +Y
							if constexpr (THREE_DIMENSIONAL) {
								if (iz == 0)      update_flags |= (causal_trigger_mask & (1 << 4)); // -Z
								if (iz == NZ - 1) update_flags |= (causal_trigger_mask & (1 << 5)); // +Z
							}
							update_flags |= (1 << 6); // thread_updated_ever
							
							// Shift by 8 for internal pixel flags.
							int flag_idx = iz * NX * NY + iy * NX + ix + 8;
							update_flags |= (1U << flag_idx);
							
							// Verify at compile-time that update_flags is actually large enough to hold all flags.
							static_assert(NZ * NY * NX + 8 <= sizeof(update_flags) * 8, 
										  "Grid volume + offset exceeds the bit-width of update_flags!");
										  
							if constexpr (!BACKWARD_PASS)
								for(int i = 0; i < CHANNELS_V; i++)
									my_v[iz][iy][ix][i] = calculated_v.f[i];
						}
					}
					
					bool is_boundary = ix == 0 || ix == NX - 1 || iy == 0 || iy == NY - 1;
                    if (THREE_DIMENSIONAL) is_boundary |= iz == 0 || iz == NZ - 1;
					if (is_boundary) {
						// Publish only the thread's core boundaries to shared memory.
                        for (int ch = 0; ch < CHANNELS; ch++)
                            SU(w, ch, sz, sy, sx) = my_u[iz][iy][ix][ch];
						
                        if constexpr (CHANNELS_V > 0)
                            for(int i = 0; i < CHANNELS_V; i++)
                                SV(w, i, sz, sy, sx) = my_v[iz][iy][ix][i];
                    }
				}
			}
			
			// Synchronize to ensure all threads have written their updated value to SU.
			if (__syncthreads_and(thread_converged)) {
				update_flags |= (1 << 7);  // fully_converged
				break;
			}
			thread_converged = 1;
		}
        
		if (tid == 32) {
            bool fully_converged = update_flags & (1 << 7);
			job_idx = atomicAdd(d_jobs_claimed, 1);
			if (!fully_converged)
				s_needs_neighbor[13] = true; // Queue self.
			static_assert(TILE_W * TILE_H * TILE_D >= 33);
		}
		
		// Final Global Output
		batch_id = s_batch_id;
		#pragma unroll
		for (int iz = 0; iz < NZ; ++iz)
		for (int iy = 0; iy < NY; ++iy)
		for (int ix = 0; ix < NX; ++ix) {
			[[maybe_unused]] int flat_idx = iz * NX * NY + iy * NX + ix;
			int gx = base_gx + ix;
			int gy = base_gy + iy;
			int gz = THREE_DIMENSIONAL ? base_gz + iz : 0;
			int flag_idx = iz * NX * NY + iy * NX + ix + 8;
			// if(flag_idx >= sizeof(update_flags) * 8) flag_idx = 6;  // In case of overflow, just always write the remaining pixels if any pixel were modified (since we cannot track individual flags). Alternatively, use a long int for flag storage in this case.
			if ((update_flags & (1U << flag_idx)) && (VALID_MASK)) {
				int global_row_u = batch_id * (grid_height * grid_depth) + gz * grid_height + gy;
				
				VecType out_val = {0};
				for(int c = 0; c < CHANNELS; ++c) out_val.f[c] = my_u[iz][iy][ix][c];
				((VecType*)((char*)d_u + global_row_u * pitch_u_bytes))[gx] = out_val;
				
				VecTypeV out_v = {0};
				for(int c = 0; c < CHANNELS_V; ++c)
					out_v.f[c] = my_v[iz][iy][ix][c];
				if constexpr (CHANNELS_V > 0)
					((VecTypeV*)((char*)d_v + global_row_u * pitch_v_bytes))[gx] = out_v;
			}
		}
		
		int next_active_block_idx = s_num_active_blocks;
		if (tid == 32 && job_idx < s_num_active_blocks)
			next_active_block_idx = active_block_list[job_idx];
		
		// =========================================================
		// FUSED COMPACTION & BOUNDARY CHECKING: Enqueue neighbors if the outer boundaries of the block changed.
		// =========================================================
		// Pre-calculate boundary capabilities for this specific thread.
		bool is_mx = (get_tid_x() == 0)          && (base_gx > 0);
		bool is_px = (get_tid_x() == TILE_W - 1) && (base_gx + NX < grid_width);
		bool is_my = (get_tid_y() == 0)          && (base_gy > 0);
		bool is_py = (get_tid_y() == TILE_H - 1) && (base_gy + NY < grid_height);

		// Define the iteration bounds [-1, 0, 1] for this thread's updates
		int min_dx = ((update_flags & (1 << 0)) && is_mx) ? -1 : 0;
		int max_dx = ((update_flags & (1 << 1)) && is_px) ?  1 : 0;
		int min_dy = ((update_flags & (1 << 2)) && is_my) ? -1 : 0;
		int max_dy = ((update_flags & (1 << 3)) && is_py) ?  1 : 0;

		int min_dz = 0, max_dz = 0;
		if constexpr (THREE_DIMENSIONAL) {
			bool is_mz = (get_tid_z() == 0)          && (base_gz > 0);
			bool is_pz = (get_tid_z() == TILE_D - 1) && (base_gz + NZ < grid_depth);
			min_dz = ((update_flags & (1 << 4)) && is_mz) ? -1 : 0;
			max_dz = ((update_flags & (1 << 5)) && is_pz) ?  1 : 0;
		}

		// 3x3x3 Topological Iteration.
		for (int z = min_dz; z <= max_dz; ++z) {
			for (int y = min_dy; y <= max_dy; ++y) {
				for (int x = min_dx; x <= max_dx; ++x) {
					int dist = abs(x) + abs(y) + abs(z);
					if (dist == 0) continue; // Center (self) handled separately by tid 0
					if (!MSFM && dist > 1) continue; // Skip diagonals if cardinal-only

					// Map (dx, dy, dz) from [-1, 0, 1] space to [0...26] 1D array index
					[[maybe_unused]] int flat_idx = (z + 1) * 9 + (y + 1) * 3 + (x + 1);
                    // Enqueue neighbor.
				    s_needs_neighbor[flat_idx] = true;
				}
			}
		}
		
		// Grab a new job.
		if (tid == 32) {
			s_job_idx = job_idx;
			s_data_block_id = next_active_block_idx;
		}
		
		__syncthreads(); // WAIT for all warps to finish writing to s_needs_neighbor.
		
		
		// =========================================================
		// DYNAMIC JOB SCHEDULING
		// =========================================================
		if (tid < 27) {
			// Because dist == 0 was skipped, s_needs_neighbor[13] is inherently false.
			// This loop now cleanly only processes actual external neighbors!
			if (s_needs_neighbor[tid]) {
				// Reverse engineer the coordinates from the 1D thread ID
				int dy = ((tid / 3) % 3) - 1;
				int dx = (tid % 3) - 1;

				int target = data_block_id;
				target += dx;
				target += dy * blocks_per_row;
				if constexpr (THREE_DIMENSIONAL) {
					int dz = (tid / 9) - 1;
					target += dz * (blocks_per_row * blocks_per_col);
				}

				// Execute the atomic operation for the neighbor
				if (atomicExch(&d_is_enqueued[target], 1) == 0) {
					int idx = atomicAdd(d_next_num_active_blocks, 1);
					next_active_block_list[idx] = target;
				}
			}
			s_needs_neighbor[tid] = false; // Reset the flags for the next pass
		}
		
		job_idx = s_job_idx;
		data_block_id = s_data_block_id;
	}
}

// Inverse beamforming kernel.
template <int CHANNELS_PER_BLOCK, int block_size>
__global__ void projection_kernel_cubic(
    float* __restrict__ rf_data, size_t rf_data_pitch,
    const float* __restrict__ image, size_t image_pitch,
    const float* __restrict__ lookup_map, size_t lookup_map_pitch,
	int image_w, int image_h
) {
	const int BIN_COUNT = 1024;
	const int GHOST_PAD = 2; // GHOST_PAD handles any cubic window out-of-bounds issues.
    const int STRIDE = BIN_COUNT + (GHOST_PAD * 2);
    const int BANK_PAD = (STRIDE % 32 == 0) ? 1 : 0; // BANK_PAD ensures odd stride to minimize bank conflicts (any odd number is co-prime with 32).
    const int PADDED_STRIDE = STRIDE + BANK_PAD;
    __shared__ float s_projections[CHANNELS_PER_BLOCK][PADDED_STRIDE];

    const int channel_offset = blockIdx.y * CHANNELS_PER_BLOCK;

    // Initialize shared memory
	for (int b = threadIdx.x; b < PADDED_STRIDE; b += blockDim.x)
		for (int c = 0; c < CHANNELS_PER_BLOCK; ++c)
			s_projections[c][b] = 0.0f;
	
    __syncthreads();  // Wait till shared memory is initialized.
	
	// TODO: 2D thread blocks. If multiple emitters, register block on emissions so we can reuse those TOF a few times for each loaded TOF.
	const int image_size = image_w * image_h;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < image_size; idx += block_size * gridDim.x) {
        const float pixel_val = image[idx];
		// if (pixel_val == 0.0f) continue; // This early exit is most relevant in the case of apodization.

        #pragma unroll
        for (int c = 0; c < CHANNELS_PER_BLOCK; ++c) {
            int current_channel = channel_offset + c;
            float dest_idx = lookup_map[current_channel * image_size + idx];

            // Cubic Lagrange Interpolation Setup.
            int bin_0 = (int) dest_idx;
            float t = dest_idx - (float) bin_0;
			
            // Lagrange Weights
			// Re-arranged using Horner's method for faster FMA generation.
			float w_m1 = t * (t * (-0.1666667f * t + 0.5f) - 0.3333333f); // bin_0 - 1
			float w_0  = t * (t * ( 0.5f * t - 1.0f) - 0.5f) + 1.0f;      // bin_0
			float w_1  = t * (t * (-0.5f * t + 0.5f) + 1.0f);             // bin_0 + 1
			float w_2  = t * (t *  0.1666667f - 0.1666667f);              // bin_0 + 2
			
			// Single bounding box check.
            if (bin_0 >= -1 && bin_0 <= BIN_COUNT) {
				int smem_idx = bin_0 + GHOST_PAD;
				atomicAdd(&s_projections[c][smem_idx - 1], pixel_val * w_m1);
				atomicAdd(&s_projections[c][smem_idx],     pixel_val * w_0);
				atomicAdd(&s_projections[c][smem_idx + 1], pixel_val * w_1);
				atomicAdd(&s_projections[c][smem_idx + 2], pixel_val * w_2);
			}
        }
    }

    __syncthreads();

    // Flush Shared Memory to Global Memory
	for (int b = threadIdx.x; b < BIN_COUNT; b += blockDim.x) {
        int smem_idx = b + GHOST_PAD;
        
        #pragma unroll
        for (int c = 0; c < CHANNELS_PER_BLOCK; ++c) {
            float val = s_projections[c][smem_idx];
            if (val != 0.0f) {  // Uses != 0.0f as cubic can have negative weights.
                int global_channel = channel_offset + c;
                atomicAdd(&rf_data[global_channel * BIN_COUNT + b], val);
            }
        }
    }
}

#endif // EIKO_KERNELS_H
