// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCFCheckPadBaseline.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H
#define O2_GPU_GPU_TPC_CF_CHECK_PAD_BASELINE_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "PackedCharge.h"

#include "clusterFinderDefs.h"

namespace o2::gpu
{

class GPUTPCCFCheckPadBaseline : public GPUKernelTemplate
{

 public:
  // Base types
  using Vec8 = uint4;
  using Vec4 = uint64_t;
  using Vec_t = Vec4; // FIXME: Vec4 for testing only, switch to Vec8

  enum {
    PadsPerCacheline = 8,
    TimebinsPerCacheline = 4,
    EntriesPerCacheline = PadsPerCacheline * TimebinsPerCacheline,
    EntriesPerVector = sizeof(Vec_t) / sizeof(PackedCharge),
    TimebinsPerVector = 1,
    PadsPerVector = EntriesPerVector / TimebinsPerVector,
    ThreadsPerCacheline = EntriesPerCacheline / EntriesPerVector,
    // FIXME: don't hardcode WarpSize
    NumCachelinesInSmem = 64 / (EntriesPerCacheline / EntriesPerVector),
    NumPadsInSmem = NumCachelinesInSmem * PadsPerCacheline,
  };

  union GPUSharedMemory {
    Vec_t asVec[TimebinsPerCacheline][(PadsPerCacheline * NumCachelinesInSmem) / PadsPerVector];
    PackedCharge asPacked[TimebinsPerCacheline][PadsPerCacheline * NumCachelinesInSmem];
  };

  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }

  GPUhdi() constexpr static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer);

 private:
  GPUd() static CfChargePos padToCfChargePos(int32_t& pad, const GPUTPCClusterFinder&);
  GPUd() static void updatePadBaseline(int32_t pad, const GPUTPCClusterFinder&, int32_t totalCharges, int32_t consecCharges, tpccf::Charge maxCharge);
};

} // namespace o2::gpu

#endif
