#pragma once

#include "Types.hpp"
#include "Config.hpp"

#include <atomic>
#include <condition_variable>



class RadixJoin
{
  public:
    RadixJoin(relation_t &R, relation_t &S);
    RadixJoin(RadixJoin &&) = delete;
    RadixJoin(const RadixJoin &) = delete;
    RadixJoin &operator=(RadixJoin &&) = delete;
    RadixJoin &operator=(const RadixJoin &) = delete;
    ~RadixJoin();

    result_relation_t &join();
    


  private:
    relation_t &R_;
    relation_t &S_;
    result_relation_t result;


};