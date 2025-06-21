#pragma once

#include <cstdint>
#include <vector>
#include <mutex>

struct tuple_t{
    uint64_t key;
    uint64_t rid;
    tuple_t(){
        key = 0;
        rid = 0;
    }
};



struct relation_t{
    tuple_t* data = nullptr;
    uint64_t number_tuples = 0;
    ~relation_t(){
        delete[] data;
    }
};


struct result_relation_t{
    // vector of row ids from relation S and R ,i.e., join partners
    // R.rid, S.rid
    std::vector<std::pair<uint64_t,uint64_t>> data;
    std::mutex lock; //optional
};