#include "MemoryFactory.h"
#include "HBM.h"

using namespace NDPSim;

namespace NDPSim
{

template <>
void MemoryFactory<HBM>::validate(int channels, int ranks, const Config& configs) {
    // assert(channels == 8 && "HBM comes with 8 channels");
}

// This function can be used by autoconf AC_CHECK_LIB since
// apparently it can't detect C++ functions.
// Basically just an entry in the symbol table
extern "C"
{
    void libramulator_is_present(void)
    {
        ;
    }
}
} // end namespace
