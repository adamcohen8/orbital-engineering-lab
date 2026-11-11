#ifndef RPO_BRIDGE_MSG_H
#define RPO_BRIDGE_MSG_H

#include "cfe.h"

#define RPO_BRIDGE_SIM_SENSOR_MID 0x1900
#define RPO_BRIDGE_ACT_CMD_MID    0x1901

typedef struct
{
    CFE_MSG_CommandHeader_t CommandHeader;
    uint8 Mode;
    double ThrustEciKmS2[3];
    double TorqueBodyNm[3];
    double WheelTorqueNm[3];
    uint32 ValidTimeoutMs;
} RPO_BRIDGE_ActCmd_t;

typedef struct
{
    CFE_MSG_TelemetryHeader_t TlmHeader;
    uint8 ValidFlags;
    double PosEciKm[3];
    double VelEciKmS[3];
    double QuatBn[4];
    double OmegaBodyRadS[3];
    double MassKg;
    double SunDirEci[3];
    double DensityKgM3;
} RPO_BRIDGE_SimSensor_t;

#endif
