#ifndef RTT_SPGEMM_TILESPGEMM_CUDA_COMPAT_H_
#define RTT_SPGEMM_TILESPGEMM_CUDA_COMPAT_H_

/*
 * Compatibility include for callers that used the original header name.
 * The former five-way numeric dispatch and its CUDA streams were removed.
 * All numeric tiles now use dmma_numeric_kernel.
 */
#include "dmma_spgemm.h"

#endif
