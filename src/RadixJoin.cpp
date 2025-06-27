#include "RadixJoin.hpp"
#include <iostream>
#include <vector>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::ceil, std::log2
#include <numeric>   // For std::partial_sum, std::iota
#include <thread>
#include <functional> // For std::ref, std::bind
#include <atomic>
#include <unordered_map>

uint32_t Config::L3_CACHE_SIZE = 16 * 1024 * 1024; // 16 MB
uint32_t Config::NUM_CORES = 8;

RadixJoin::RadixJoin(relation_t &R, relation_t &S) : R_(R), S_(S) {
}

RadixJoin::~RadixJoin() {
}

namespace { // Anonymous namespace for helper structures specific to this file

// Custom Linear Probing Hash Table for Radix Join (build on R, probe with S)
// Assumes keys from R (build side) are unique within an R-partition.
// Uses Struct-of-Arrays (SoA) layout.
class SimpleLinearProbeHashTable {
public:
    std::vector<uint64_t> keys_;
    std::vector<uint64_t> rids_;
    std::vector<bool> occupied_;

    uint64_t table_size_;
    uint64_t num_elements_;

    inline uint64_t hash_function(uint64_t key) const {
        return std::hash<uint64_t>{}(key) % table_size_;
    }

public:
    SimpleLinearProbeHashTable(const tuple_t* r_partition_data, uint64_t r_partition_count)
        : num_elements_(0) {
        if (r_partition_count == 0) {
            table_size_ = 0;
            return;
        }

        table_size_ = std::max(1UL, r_partition_count * 2UL);


        keys_.resize(table_size_);
        rids_.resize(table_size_);
        occupied_.resize(table_size_, false); // Initialize all slots as not occupied

        for (uint64_t i = 0; i < r_partition_count; ++i) {
            const auto& r_tuple = r_partition_data[i];
            uint64_t current_key = r_tuple.key;
            uint64_t current_rid = r_tuple.rid;

            uint64_t slot_idx = hash_function(current_key);

            while (occupied_[slot_idx]) {
                // Since keys in R are unique, we don't expect to find the same key here.
                // If occupied, just move to the next slot (linear probing).
                slot_idx = (slot_idx + 1) % table_size_;
            }

            keys_[slot_idx] = current_key;
            rids_[slot_idx] = current_rid;
            occupied_[slot_idx] = true;
            num_elements_++;
        }
    }

    // Looks for s_key (from S partition) in the hash table.
    // If found, sets out_r_rid to the rid from R and returns true.
    // Otherwise, returns false.
    bool lookup(uint64_t s_key, uint64_t& out_r_rid) const {
        if (table_size_ == 0) { // Empty table
            return false;
        }

        uint64_t slot_idx = hash_function(s_key);
        uint64_t initial_slot_idx = slot_idx;

        do {
            if (!occupied_[slot_idx]) {
                // Encountered an empty slot, key not found
                return false;
            }
            if (keys_[slot_idx] == s_key) {
                // Key found
                out_r_rid = rids_[slot_idx];
                return true;
            }
            slot_idx = (slot_idx + 1) % table_size_;
        } while (slot_idx != initial_slot_idx); // Stop if we've wrapped around

        // Wrapped around the entire table without finding the key or an empty slot
        return false;
    }
};

} // end anonymous namespace


/*
*RADIX JOIN - Implement Radix join with the following requirements
* 1. Multithreaded
* 2. Use Radix Partitioning to create chunks fitting into the cache, but only one pass needed, i.e., figure out how
*    many bits you need to create N partitions
*
* Input: Use Member Variables R and S
*
* Return: result_relation_t - defined in Types.hpp
*/
result_relation_t &RadixJoin::join() {
    if (R_.number_tuples == 0 || S_.number_tuples == 0) {
        return result;
    }

    uint64_t size_R_bytes = R_.number_tuples * sizeof(tuple_t);
    uint64_t size_S_bytes = S_.number_tuples * sizeof(tuple_t);

    uint64_t smaller_relation_size_bytes = std::min(size_R_bytes, size_S_bytes);
    if (smaller_relation_size_bytes == 0) {
        return result;
    }

    double desired_partition_payload_size = static_cast<double>(Config::L3_CACHE_SIZE) / 8.0;

    double num_partitions_for_cache_double = 1.0;
    if (smaller_relation_size_bytes > 0 && desired_partition_payload_size > 0) {
        num_partitions_for_cache_double = static_cast<double>(smaller_relation_size_bytes) / desired_partition_payload_size;
    }

    uint32_t k_for_cache = 0;
    if (num_partitions_for_cache_double > 1.0) {
        k_for_cache = static_cast<uint32_t>(std::ceil(std::log2(num_partitions_for_cache_double)));
    }

    uint32_t k_for_cores = 0;
    if (Config::NUM_CORES > 1) {
        k_for_cores = static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(Config::NUM_CORES))));
    }

    uint32_t k_radix_bits = std::max({k_for_cache, k_for_cores});
    k_radix_bits = std::max(1u, k_radix_bits);
    k_radix_bits = std::min(k_radix_bits, 11u);

    uint32_t N_partitions = 1 << k_radix_bits;
    uint64_t radix_mask = N_partitions - 1;

    struct PartitionOutput {
        relation_t partitioned_relation;
        std::vector<uint64_t> histogram;
        std::vector<uint64_t> prefix_sum;
    };

    auto partition_relation =
        [&](const relation_t& input_relation, uint32_t /* current_k_radix_bits_unused */, uint32_t current_N_partitions, uint64_t current_radix_mask) -> PartitionOutput {
        PartitionOutput output;
        output.partitioned_relation.data = new tuple_t[input_relation.number_tuples];
        output.partitioned_relation.number_tuples = input_relation.number_tuples;
        output.histogram.resize(current_N_partitions, 0);
        output.prefix_sum.resize(current_N_partitions, 0);

        if (input_relation.number_tuples == 0) {
            return output;
        }

        std::vector<std::vector<uint64_t>> local_histograms(
            Config::NUM_CORES, std::vector<uint64_t>(current_N_partitions, 0));

        std::vector<std::thread> threads;
        uint64_t tuples_per_thread = (input_relation.number_tuples + Config::NUM_CORES - 1) / Config::NUM_CORES;

        for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
            threads.emplace_back([&, t_id]() {
                uint64_t start_idx = t_id * tuples_per_thread;
                uint64_t end_idx = std::min(start_idx + tuples_per_thread, input_relation.number_tuples);
                for (uint64_t i = start_idx; i < end_idx; ++i) {
                    uint64_t p_idx = input_relation.data[i].key & current_radix_mask;
                    local_histograms[t_id][p_idx]++;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        threads.clear();

        for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
            for (uint32_t p = 0; p < current_N_partitions; ++p) {
                output.histogram[p] += local_histograms[t_id][p];
            }
        }

        output.prefix_sum[0] = 0;
        for (uint32_t p = 1; p < current_N_partitions; ++p) {
            output.prefix_sum[p] = output.prefix_sum[p - 1] + output.histogram[p - 1];
        }

        std::vector<std::vector<uint64_t>> thread_starting_write_positions(
            Config::NUM_CORES, std::vector<uint64_t>(current_N_partitions));

        for (uint32_t p = 0; p < current_N_partitions; ++p) {
            uint64_t current_offset_for_p = output.prefix_sum[p];
            for (uint32_t t_idx = 0; t_idx < Config::NUM_CORES; ++t_idx) {
                thread_starting_write_positions[t_idx][p] = current_offset_for_p;
                current_offset_for_p += local_histograms[t_idx][p];
            }
        }

        for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
            threads.emplace_back([&, t_id]() {
                std::vector<uint64_t> current_thread_p_counters = thread_starting_write_positions[t_id];

                uint64_t start_idx = t_id * tuples_per_thread;
                uint64_t end_idx = std::min(start_idx + tuples_per_thread, input_relation.number_tuples);

                for (uint64_t i = start_idx; i < end_idx; ++i) {
                    const auto& tuple = input_relation.data[i];
                    uint64_t p_idx = tuple.key & current_radix_mask;
                    output.partitioned_relation.data[current_thread_p_counters[p_idx]] = tuple;
                    current_thread_p_counters[p_idx]++;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        threads.clear();

        return output;
    };

    PartitionOutput R_partitioned_output = partition_relation(R_, k_radix_bits, N_partitions, radix_mask);
    PartitionOutput S_partitioned_output = partition_relation(S_, k_radix_bits, N_partitions, radix_mask);

    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> thread_local_results(Config::NUM_CORES);

    std::vector<std::thread> join_threads;
    std::atomic<uint32_t> next_partition_to_process(0);

    for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
        join_threads.emplace_back([&, t_id]() {
            while(true) {
                uint32_t p_idx = next_partition_to_process.fetch_add(1, std::memory_order_relaxed);

                if (p_idx >= N_partitions) {
                    break;
                }

                uint64_t r_part_start_offset = R_partitioned_output.prefix_sum[p_idx];
                uint64_t r_part_count = R_partitioned_output.histogram[p_idx];

                uint64_t s_part_start_offset = S_partitioned_output.prefix_sum[p_idx];
                uint64_t s_part_count = S_partitioned_output.histogram[p_idx];

                if (r_part_count == 0 || s_part_count == 0) {
                    continue;
                }

                tuple_t* r_part_data = R_partitioned_output.partitioned_relation.data + r_part_start_offset;
                tuple_t* s_part_data = S_partitioned_output.partitioned_relation.data + s_part_start_offset;

                // Always build hash table on R_p (keys in R are unique), probe with S_p.
                // Use the custom SimpleLinearProbeHashTable
                SimpleLinearProbeHashTable ht(r_part_data, r_part_count);

                for (uint64_t i = 0; i < s_part_count; ++i) {
                    // Example of where prefetching was:
                    // if (i + PREFETCH_DISTANCE < s_part_count) {
                    //     __builtin_prefetch(&s_part_data[i + PREFETCH_DISTANCE], 0, 3);
                    // }

                    uint64_t s_key = s_part_data[i].key;
                    uint64_t s_rid = s_part_data[i].rid;
                    uint64_t r_rid_found;

                    if (ht.lookup(s_key, r_rid_found)) {
                        thread_local_results[t_id].emplace_back(r_rid_found, s_rid);
                    }
                }
            }
        });
    }

    for (auto& t : join_threads) {
        t.join();
    }

    uint64_t total_result_pairs = 0;
    for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
        total_result_pairs += thread_local_results[t_id].size();
    }
    result.data.reserve(total_result_pairs);
    for (uint32_t t_id = 0; t_id < Config::NUM_CORES; ++t_id) {
        result.data.insert(result.data.end(), thread_local_results[t_id].begin(), thread_local_results[t_id].end());
    }

    return result;
}
