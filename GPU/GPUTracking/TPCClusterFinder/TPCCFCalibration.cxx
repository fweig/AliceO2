// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCFCalibration.cxx
/// \author Felix Weiglhofer

#include "TPCCFCalibration.h"

#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/CalDet.h"
#include "GPUSettings.h"

using namespace GPUCA_NAMESPACE::gpu;

TPCCFCalibration::TPCCFCalibration()
{
  GPUTPCGeometry geo{};
  int offset = 0;
  for (int r = 0; r < TPC_NUM_OF_ROWS; r++) {
    mPadOffsetPerRow[r] = offset;
    offset += geo.NPads(r);
  }
}

TPCCFCalibration::TPCCFCalibration(const GPUSettingsRec& rec) : TPCCFCalibration()
{
  qmaxCutoff = rec.tpcCFqmaxCutoff;
  qtotCutoff = rec.tpcCFqtotCutoff;
  extendInnerChargeThreshold = rec.tpcCFinnerThreshold;
  minSplitChargeNum = rec.tpcCFminSplitNum;
  noiseSuppressionEpsilon = rec.tpcCFnoiseSuppressionEpsilon;
}

TPCCFCalibration::TPCCFCalibration(const GPUSettingsRec& rec, const o2::tpc::CalDet<float>& gainMap) : TPCCFCalibration(rec)
{
  for (int sector = 0; sector < o2::tpc::constants::MAXSECTOR; sector++) {
    for (int p = 0; p < TPC_PADS_IN_SECTOR; p++) {
      mGainCorrection[sector].set(p, gainMap.getValue(sector, p));
    }
  }
}
