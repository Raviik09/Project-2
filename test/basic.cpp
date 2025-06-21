#include "catch.hpp"  
#include "RadixJoin.hpp"

#include <iostream>


///// ----------------------- BASIC TEST CASES ----------------------- ///// 


TEST_CASE("SIMPLE JOIN 1", "[radix-join-simpe]")
{
    size_t NUM_TUPLES = 100;

    tuple_t* data_R = new tuple_t[NUM_TUPLES];
    tuple_t* data_S = new tuple_t[NUM_TUPLES];
    relation_t R;
    R.data = data_R;
    R.number_tuples = NUM_TUPLES;
    relation_t S;
    S.data = data_S;
    S.number_tuples = NUM_TUPLES;


    for(size_t i = 0; i < NUM_TUPLES; i++){
        data_R[i].key = i + 1;
        data_R[i].rid = i;
        data_S[i].key = i  + 1;
        data_S[i].rid = i;
    }

    RadixJoin radix(R, S);
    result_relation_t &result = radix.join();
    REQUIRE(result.data.size() == 100);

    for(size_t i = 0; i < NUM_TUPLES; i++){
        tuple_t r_tup = data_R[result.data[i].first];
        tuple_t s_tup = data_S[result.data[i].second];
        REQUIRE(r_tup.key == s_tup.key);
    }


    
}




TEST_CASE("SIMPLE JOIN 2", "[radix-join-simpe]")
{
    size_t NUM_TUPLES = 100;

    tuple_t* data_R = new tuple_t[NUM_TUPLES];
    tuple_t* data_S = new tuple_t[NUM_TUPLES];
    relation_t R;
    R.data = data_R;
    R.number_tuples = NUM_TUPLES;
    relation_t S;
    S.data = data_S;
    S.number_tuples = NUM_TUPLES;


    for(size_t i = 0; i < NUM_TUPLES; i++){
        data_R[i].key = i + 1;
        data_R[i].rid = i;
        data_S[i].key = i  + 101;
        data_S[i].rid = i;
    }

    RadixJoin radix(R, S);
    result_relation_t &result = radix.join();
    REQUIRE(result.data.size() == 0);


    
}

TEST_CASE("SIMPLE JOIN 3", "[radix-join-simpe]")
{
    size_t NUM_TUPLES = 100000;

    tuple_t* data_R = new tuple_t[NUM_TUPLES];
    tuple_t* data_S = new tuple_t[NUM_TUPLES];
    relation_t R;
    R.data = data_R;
    R.number_tuples = NUM_TUPLES;
    relation_t S;
    S.data = data_S;
    S.number_tuples = NUM_TUPLES;


    for(size_t i = 0; i < NUM_TUPLES; i++){
        data_R[i].key = i + 1;
        data_R[i].rid = i;
        data_S[i].key = i  + 1;
        data_S[i].rid = i;
    }

    RadixJoin radix(R, S);
    result_relation_t &result = radix.join();
    REQUIRE(result.data.size() == 100000);

    for(size_t i = 0; i < NUM_TUPLES; i++){
        tuple_t r_tup = data_R[result.data[i].first];
        tuple_t s_tup = data_S[result.data[i].second];
        REQUIRE(r_tup.key == s_tup.key);
    }

    

}
