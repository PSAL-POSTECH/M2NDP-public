// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#ifndef HASHING_H
#define HASHING_H

namespace NDPSim {
unsigned ipoly_hash_function(uint64_t higher_bits, unsigned index,
                             unsigned bank_set_num);

unsigned bitwise_hash_function(uint64_t higher_bits, unsigned index,
                               unsigned bank_set_num);

unsigned PAE_hash_function(uint64_t higher_bits, unsigned index,
                           unsigned bank_set_num);

unsigned mini_hash_function(uint64_t higher_bits, unsigned index, 
                            unsigned bank_set_num);
}
#endif
