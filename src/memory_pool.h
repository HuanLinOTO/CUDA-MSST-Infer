#pragma once
// memory_pool.h — CUDA caching memory allocator for CudaInfer
//
// PyTorch-style block-level caching that eliminates cudaMalloc/cudaFree overhead.
// Memory is retained in free-lists keyed by size and reused across operations.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

namespace cudasep {

class CudaMemoryPool {
public:
    static CudaMemoryPool& instance() {
        static CudaMemoryPool pool;
        return pool;
    }

    // Allocate GPU memory (returns cached block if available)
    void* allocate(size_t requested_bytes) {
        if (requested_bytes == 0) return nullptr;

        // Round up to granularity to improve cache hit rate
        size_t alloc_size = round_up(requested_bytes);

        std::lock_guard<std::mutex> lock(mutex_);

        // Look for a cached block of exact size or slightly larger
        auto it = free_blocks_.lower_bound(alloc_size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            // Found a suitable block. Check if it's not too much larger (within 2x)
            if (it->first <= alloc_size * 2) {
                void* ptr = it->second.back();
                it->second.pop_back();
                if (it->second.empty()) {
                    free_blocks_.erase(it);
                }
                active_blocks_[ptr] = it->first;
                stats_.cache_hits++;
                stats_.active_bytes += it->first;
                return ptr;
            }
        }

        // No suitable cached block — allocate new
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, alloc_size);
        if (err != cudaSuccess) {
            // Try to free some cached memory and retry
            release_cached_memory();
            err = cudaMalloc(&ptr, alloc_size);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CudaMemoryPool: cudaMalloc failed for ") +
                    std::to_string(alloc_size) + " bytes: " + cudaGetErrorString(err));
            }
        }

        active_blocks_[ptr] = alloc_size;
        stats_.allocations++;
        stats_.total_allocated += alloc_size;
        stats_.active_bytes += alloc_size;
        if (stats_.active_bytes > stats_.peak_bytes)
            stats_.peak_bytes = stats_.active_bytes;

        return ptr;
    }

    // Return memory to the cache (does NOT call cudaFree)
    void deallocate(void* ptr) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = active_blocks_.find(ptr);
        if (it == active_blocks_.end()) {
            // Not from our pool — just free it
            cudaFree(ptr);
            return;
        }

        size_t size = it->second;
        active_blocks_.erase(it);
        free_blocks_[size].push_back(ptr);
        stats_.active_bytes -= size;
        stats_.deallocations++;
    }

    // Release all cached (unused) memory back to CUDA
    void release_cached_memory() {
        // NOTE: must be called with mutex held or externally synchronized
        size_t freed = 0;
        for (auto& [size, blocks] : free_blocks_) {
            for (void* ptr : blocks) {
                cudaFree(ptr);
                freed += size;
            }
        }
        free_blocks_.clear();
        stats_.total_freed += freed;
    }

    // Release ALL memory (active + cached)
    void release_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_cached_memory();
        for (auto& [ptr, size] : active_blocks_) {
            cudaFree(ptr);
            stats_.total_freed += size;
        }
        active_blocks_.clear();
        stats_.active_bytes = 0;
    }

    struct Stats {
        size_t allocations = 0;
        size_t deallocations = 0;
        size_t cache_hits = 0;
        size_t total_allocated = 0;
        size_t total_freed = 0;
        size_t active_bytes = 0;
        size_t peak_bytes = 0;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void print_stats() const {
        auto s = get_stats();
        size_t cached = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [size, blocks] : free_blocks_) {
                cached += size * blocks.size();
            }
        }
        std::cout << "[MemPool] allocs=" << s.allocations
                  << " deallocs=" << s.deallocations
                  << " cache_hits=" << s.cache_hits
                  << " active=" << (s.active_bytes / 1024 / 1024) << "MB"
                  << " cached=" << (cached / 1024 / 1024) << "MB"
                  << " peak=" << (s.peak_bytes / 1024 / 1024) << "MB"
                  << std::endl;
    }

    ~CudaMemoryPool() {
        release_all();
    }

private:
    CudaMemoryPool() = default;

    // Round up to allocation granularity
    // Small: round to 512 bytes
    // Medium: round to 2MB
    // Large: round to 20MB
    static size_t round_up(size_t bytes) {
        if (bytes <= 512) return 512;
        if (bytes <= 1024 * 1024) {
            // Round to nearest 512 bytes
            return (bytes + 511) & ~511ULL;
        }
        if (bytes <= 10 * 1024 * 1024) {
            // Round to nearest 2MB
            return (bytes + (2 * 1024 * 1024 - 1)) & ~(2ULL * 1024 * 1024 - 1);
        }
        // Large: round to nearest 20MB
        return (bytes + (20 * 1024 * 1024 - 1)) & ~(20ULL * 1024 * 1024 - 1);
    }

    mutable std::mutex mutex_;
    std::map<size_t, std::vector<void*>> free_blocks_;        // size -> list of free ptrs
    std::unordered_map<void*, size_t> active_blocks_;          // ptr -> allocated size
    Stats stats_;
};

} // namespace cudasep
