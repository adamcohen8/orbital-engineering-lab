#ifndef OEL_SIL_BRIDGE_APP_H
#define OEL_SIL_BRIDGE_APP_H

#include "cfe.h"
#include "oel_sil_bridge_codec.h"
#include "oel_sil_v2_codec.h"
#include <netinet/in.h>
#include <stdbool.h>

typedef struct
{
    uint32             RunStatus;
    CFE_SB_PipeId_t    ActuatorPipe;
    int                SocketFd;
    struct sockaddr_in OelAddress;
    uint32             SensorPacketCount;
    uint32             ActuatorPacketCount;
    uint32             DecodeErrorCount;
    uint32             SendErrorCount;
    uint32             SequenceGapCount;
    uint32             LastSensorSequence;
    uint32             ActuatorSequence;
    uint64             LastSimTimeNs;
    uint32             PendingWireSequence;
    uint64             PendingInvocationId;
    uint64             PendingClockTicks;
    uint64             PendingTickPeriodNs;
    uint32             PendingClockReset;
    uint64             CommandSequence;
    char               SatelliteId[64];
    char               BootId[64];
    char               ClockId[96];
    bool               HavePendingInvocation;
    bool               HaveSensorSequence;
} OEL_SIL_BRIDGE_AppData_t;

extern OEL_SIL_BRIDGE_AppData_t OEL_SIL_BRIDGE_AppData;

void         OEL_SIL_BRIDGE_Main(void);
CFE_Status_t OEL_SIL_BRIDGE_Init(void);
void         OEL_SIL_BRIDGE_Cleanup(void);

#endif
