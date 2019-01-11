#ifndef _MEMORYCOUNTER_H
#define _MEMORYCOUNTER_H

#include <cuda_runtime.h>
#include "glog/logging.h"

#define SCALE (1 << 20)

struct MemoryCounter
{
    size_t availMem = 0;
    size_t totalMem = 0;
    size_t freeMem = 0;

    MemoryCounter()
    {
        LOG(INFO) << "MemoryCounter init";
        cudaMemGetInfo(&availMem, &totalMem);
        freeMem = availMem / SCALE;
    }

    ~MemoryCounter()
    {
        LOG(INFO) << "MemoryCounter destroyed";
    }

    void report(const char *model_name, bool free = false)
    {
        cudaMemGetInfo(&availMem, &totalMem);

        availMem /= SCALE;
        totalMem /= SCALE;

        printf("%s Memory avaliable: Free: %zu, Total: %zu, %s: %zu\n", model_name, availMem, totalMem, (!free) ? "Allocated\0" : "Freed\0", (!free) ? (freeMem - availMem) : (availMem - freeMem));
        freeMem = availMem;
    }
};

#endif _MEMORYCOUNTER_H
