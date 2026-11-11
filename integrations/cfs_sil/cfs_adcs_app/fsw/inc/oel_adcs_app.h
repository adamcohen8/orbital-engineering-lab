#ifndef OEL_ADCS_APP_H
#define OEL_ADCS_APP_H

#include "cfe.h"
#include "oel_sil_bridge_msg.h"
#include <stdbool.h>

#define OEL_ADCS_PIPE_NAME  "OEL_ADCS_PIPE"
#define OEL_ADCS_PIPE_DEPTH 32

#define OEL_ADCS_KP            0.10
#define OEL_ADCS_KD            0.05
#define OEL_ADCS_MAX_TORQUE_NM 0.05
#define OEL_ADCS_VALID_TIMEOUT_MS 500

#define OEL_ADCS_DESIRED_Q0 1.0
#define OEL_ADCS_DESIRED_Q1 0.0
#define OEL_ADCS_DESIRED_Q2 0.0
#define OEL_ADCS_DESIRED_Q3 0.0

#define OEL_ADCS_INIT_INF_EID       1
#define OEL_ADCS_PIPE_ERR_EID       2
#define OEL_ADCS_SUBSCRIBE_ERR_EID  3
#define OEL_ADCS_MESSAGE_ERR_EID    4
#define OEL_ADCS_TRANSMIT_ERR_EID   5

typedef struct
{
    uint32             RunStatus;
    CFE_SB_PipeId_t    SensorPipe;
    uint32             SensorCount;
    uint32             CommandCount;
    uint32             ErrorCount;
} OEL_ADCS_AppData_t;

extern OEL_ADCS_AppData_t OEL_ADCS_AppData;

void         OEL_ADCS_Main(void);
CFE_Status_t OEL_ADCS_Init(void);

#endif
