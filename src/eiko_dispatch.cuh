#ifndef EIKO_DISPATCH_H
#define EIKO_DISPATCH_H

#include <utility>  // for std::forward, std::index_sequence, and std::make_index_sequence
#include <array>    // for std::array.
#include <cstdint>  // for std::size_t

#include "BatchedFIMSolver.cuh"  // Defines BatchedFIMSolver.

namespace detail {
    // Helper function to generate the jump table at compile-time.
    // std::index_sequence<Is...> provides a parameter pack of integers (0, 1, 2, ..., 31).
    template<template<bool, bool, bool, bool, bool> class Op, typename... Args, std::size_t... Is>
    constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
        
        // Define a function pointer signature that perfectly matches the forwarded runtime arguments.
        // The return type is void to match your original dispatch_fim signature.
        using FuncPtr = void (*)(Args&&...);

        // Construct and return an array of 32 function pointers.
        // The unpack operator (...) expands the lambda for each integer in the 'Is' sequence.
        return std::array<FuncPtr, sizeof...(Is)>{
            [](Args&&... args) -> void {
                // For each integer index (0 to 31), extract its 5 bits.
                // Each bit drives one of the compile-time boolean template parameters.
                Op<
                    ((Is & 16) != 0), // Bit 4: is_3d
                    ((Is & 8) != 0),  // Bit 3: is_backward
                    ((Is & 4) != 0),  // Bit 2: msfm
                    ((Is & 2) != 0),  // Bit 1: has_v
                    ((Is & 1) != 0)   // Bit 0: is_gated_x
                >{}(std::forward<Args>(args)...); // Instantiate Op and invoke it with the forwarded args
            }...
        };
    }
}

// The Dispatcher function.
template<template<bool, bool, bool, bool, bool> class Op, typename... Args>
void dispatch_fim(bool is_3d, bool is_backward, bool msfm, bool has_v, bool is_gated_x, Args&&... args) {
    
    // Initialize the jump table once at compile-time.
    // std::make_index_sequence<32> generates the indices 0 through 31 needed by the helper.
    // 'static constexpr' guarantees this array is built during compilation and stored in read-only memory.
    static constexpr auto table = detail::make_dispatch_table<Op, Args...>(std::make_index_sequence<32>{});

    // Pack the 5 runtime boolean flags into a single 5-bit integer index (0 to 31).
    // We shift each boolean into its corresponding bit position to match the extraction logic above.
    std::size_t index = 
        (static_cast<std::size_t>(is_3d)       << 4) | 
        (static_cast<std::size_t>(is_backward) << 3) | 
        (static_cast<std::size_t>(msfm)        << 2) | 
        (static_cast<std::size_t>(has_v)       << 1) | 
        (static_cast<std::size_t>(is_gated_x));        // Bit 0 needs no shift

    // Perform an O(1) jump directly to the correctly instantiated template function.
    table[index](std::forward<Args>(args)...);
}

// Default generator.
template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
constexpr FIMConfig MakeDefaultConfig() {
    FIMConfig cfg{};
    cfg.THREE_DIMENSIONAL = IS_3D;
    cfg.BACKWARD_PASS = IS_BACKWARD;
    cfg.MSFM = MSFM;
    cfg.CHANNELS = 1; cfg.CHANNELS_F = 1; cfg.DOUBLE_BUFFERED_SMEM = false;
    cfg.CHANNELS_V = static_cast<int>(HAS_V);
    cfg.GATED_X = GATED_X;

    if (IS_3D) {
        cfg.NX = 4; cfg.NY = 1; cfg.NZ = 1;
        cfg.TILE_W = 16; cfg.TILE_H = 16; cfg.TILE_D = 1;
    } else {
        cfg.NX = IS_BACKWARD ? 3 : 2;
        cfg.NY = 1; cfg.NZ = 1;
        cfg.TILE_W = IS_BACKWARD ? 8 : 32;
        cfg.TILE_H = 8; cfg.TILE_D = 1;
    }
    cfg.MAX_LOCAL_ITERS = IS_BACKWARD ? 100 : 25;
    return cfg;
}

// Default dispatch.
template <bool IS_3D, bool IS_BACKWARD, bool MSFM, bool HAS_V, bool GATED_X>
using StandardFIMSolver = BatchedFIMSolver<MakeDefaultConfig<IS_3D, IS_BACKWARD, MSFM, HAS_V, GATED_X>()>;

#endif // EIKO_DISPATCH_H
