#ifndef RPO_BRIDGE_APP_H
#define RPO_BRIDGE_APP_H

#include "cfe.h"
#include "rpo_bridge_msg.h"
#include <stdbool.h>

#define RPO_BRIDGE_APP_NAME "RPO_BRIDGE"
#define RPO_BRIDGE_PIPE_NAME "RPO_BRIDGE_PIPE"
#define RPO_BRIDGE_PIPE_DEPTH 16
#define RPO_BRIDGE_ACT_CMD_VALID_TIMEOUT_MS 200

/*
 * Basic attitude-hold mission tuning.
 *
 * With RPO_BRIDGE_ATT_HOLD_CAPTURE_INITIAL enabled, the app captures the first
 * valid sensed attitude as the hold target. Set it to 0 to use the compile-time
 * desired quaternion below.
 */
#define RPO_BRIDGE_ATT_HOLD_CAPTURE_INITIAL 1
#define RPO_BRIDGE_ATT_HOLD_KP 0.10
#define RPO_BRIDGE_ATT_HOLD_KD 0.05
#define RPO_BRIDGE_ATT_HOLD_MAX_TORQUE_NM 0.05

#define RPO_BRIDGE_ATT_HOLD_Q0 1.0
#define RPO_BRIDGE_ATT_HOLD_Q1 0.0
#define RPO_BRIDGE_ATT_HOLD_Q2 0.0
#define RPO_BRIDGE_ATT_HOLD_Q3 0.0

#define RPO_BRIDGE_ATT_HOLD_W0_RAD_S 0.0
#define RPO_BRIDGE_ATT_HOLD_W1_RAD_S 0.0
#define RPO_BRIDGE_ATT_HOLD_W2_RAD_S 0.0

typedef struct
{
    CFE_SB_PipeId_t CmdPipe;
    CFE_SB_PipeId_t SensorPipe;
    CFE_ES_RunStatus_t RunStatus;
    uint32 CmdCounter;
    uint32 ErrCounter;
    bool AttHoldTargetCaptured;
    double DesiredQuatBn[4];
    double DesiredOmegaBodyRadS[3];
    RPO_BRIDGE_ActCmd_t LastActCmd;
} RPO_BRIDGE_AppData_t;

void RPO_BRIDGE_AppMain(void);
int32 RPO_BRIDGE_AppInit(void);
void RPO_BRIDGE_ProcessCommandPacket(const CFE_SB_Buffer_t *SBBufPtr);
void RPO_BRIDGE_ProcessSensorPacket(const CFE_SB_Buffer_t *SBBufPtr);
void RPO_BRIDGE_ResetAttitudeHoldTarget(void);

extern RPO_BRIDGE_AppData_t RPO_BRIDGE_AppData;

#endif
