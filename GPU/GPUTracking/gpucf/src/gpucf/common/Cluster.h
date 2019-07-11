#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Object.h>

#include <shared/ClusterNative.h>

#include <iosfwd>


namespace gpucf
{

class Cluster 
{

public:
    using FieldMask = unsigned char;

    enum Field : FieldMask
    { 
        Field_Q         = (1 << 0),
        Field_QMax      = (1 << 1),
        Field_timeMean  = (1 << 2),
        Field_padMean   = (1 << 3),
        Field_timeSigma = (1 << 4),
        Field_padSigma  = (1 << 5),

        Field_all = Field_Q 
                  | Field_QMax 
                  | Field_timeMean 
                  | Field_padMean 
                  | Field_timeSigma 
                  | Field_padSigma
    };

    float Q;
    float QMax;
    float padMean;
    float timeMean;
    float padSigma;
    float timeSigma;
    int cru;
    int row;

    bool atEdge      = false;
    bool splitInTime = false;
    bool splitInPad  = false;

    Cluster();
    Cluster(int, int, float, float, float, float, float, float);
    Cluster(const Digit &, const ClusterNative &);
    Cluster(int, int, const ClusterNative &);
    Cluster(int, const ClusterNative &);

    Object serialize() const;
    void deserialize(const Object &);

    bool hasNaN() const;
    bool hasNegativeEntries() const;

    int globalRow() const;

    bool operator==(const Cluster &) const;

    bool eq(const Cluster &, float, float, FieldMask) const;
};

std::ostream &operator<<(std::ostream &, const Cluster &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab: