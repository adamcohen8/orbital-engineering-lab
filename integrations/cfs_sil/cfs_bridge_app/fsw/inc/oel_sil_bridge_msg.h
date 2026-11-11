#ifndef OEL_SIL_BRIDGE_MSG_H
#define OEL_SIL_BRIDGE_MSG_H

#include "cfe.h"
#include "oel_sil_bridge_msgids.h"

typedef struct
{
    CFE_MSG_TelemetryHeader_t TelemetryHeader;
    uint8                     ValidFlags;
    double                    PosEciKm[3];
    double                    VelEciKmS[3];
    double                    QuatBn[4];
    double                    OmegaBodyRadS[3];
    double                    MassKg;
    double                    SunDirEci[3];
    double                    DensityKgM3;
} OEL_SIL_SensorMsg_t;

typedef struct
{
    CFE_MSG_CommandHeader_t CommandHeader;
    uint8                   Mode;
    double                  ThrustEciKmS2[3];
    double                  TorqueBodyNm[3];
    double                  WheelTorqueNm[3];
    uint32                  ValidTimeoutMs;
} OEL_SIL_ActuatorMsg_t;

#endif
