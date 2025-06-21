#include "RadixJoin.hpp"
#include <iostream>

uint32_t Config::L3_CACHE_SIZE = 16 * 1024 * 1024;
uint32_t Config::NUM_CORES = 8;

RadixJoin::RadixJoin(relation_t &R, relation_t &S) : R_(R), S_(S){
}

RadixJoin::~RadixJoin(){
}




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
result_relation_t &RadixJoin::join(){
    return result;
}



