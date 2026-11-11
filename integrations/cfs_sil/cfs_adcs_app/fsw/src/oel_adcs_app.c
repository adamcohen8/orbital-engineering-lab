#include "oel_adcs_app.h"

#include <math.h>
#include <string.h>

OEL_ADCS_AppData_t OEL_ADCS_AppData;

static void OEL_ADCS_NormalizeQuaternion(double q[4])
{
    double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    int    i;

    if (norm <= 0.0)
    {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
        return;
    }
    for (i = 0; i < 4; ++i)
    {
        q[i] /= norm;
    }
}

static void OEL_ADCS_ComputeTorque(const OEL_SIL_SensorMsg_t *sensor, double torque[3])
{
    double q_current[4];
    double q_desired[4] = {OEL_ADCS_DESIRED_Q0, OEL_ADCS_DESIRED_Q1, OEL_ADCS_DESIRED_Q2,
                           OEL_ADCS_DESIRED_Q3};
    double q_error[4];
    double norm;
    double scale;
    int    i;

    memcpy(q_current, sensor->QuatBn, sizeof(q_current));
    OEL_ADCS_NormalizeQuaternion(q_current);
    OEL_ADCS_NormalizeQuaternion(q_desired);

    /* q_error = conjugate(q_desired) times q_current. */
    q_error[0] = q_desired[0] * q_current[0] + q_desired[1] * q_current[1] +
                 q_desired[2] * q_current[2] + q_desired[3] * q_current[3];
    q_error[1] = q_desired[0] * q_current[1] - q_desired[1] * q_current[0] -
                 q_desired[2] * q_current[3] + q_desired[3] * q_current[2];
    q_error[2] = q_desired[0] * q_current[2] + q_desired[1] * q_current[3] -
                 q_desired[2] * q_current[0] - q_desired[3] * q_current[1];
    q_error[3] = q_desired[0] * q_current[3] - q_desired[1] * q_current[2] +
                 q_desired[2] * q_current[1] - q_desired[3] * q_current[0];
    OEL_ADCS_NormalizeQuaternion(q_error);
    if (q_error[0] < 0.0)
    {
        for (i = 0; i < 4; ++i)
        {
            q_error[i] = -q_error[i];
        }
    }

    for (i = 0; i < 3; ++i)
    {
        torque[i] = -OEL_ADCS_KP * q_error[i + 1] - OEL_ADCS_KD * sensor->OmegaBodyRadS[i];
    }
    norm = sqrt(torque[0] * torque[0] + torque[1] * torque[1] + torque[2] * torque[2]);
    if (norm > OEL_ADCS_MAX_TORQUE_NM && OEL_ADCS_MAX_TORQUE_NM > 0.0)
    {
        scale = OEL_ADCS_MAX_TORQUE_NM / norm;
        for (i = 0; i < 3; ++i)
        {
            torque[i] *= scale;
        }
    }
}

static void OEL_ADCS_ProcessSensor(const CFE_SB_Buffer_t *buffer)
{
    const OEL_SIL_SensorMsg_t *sensor = (const OEL_SIL_SensorMsg_t *)buffer;
    OEL_SIL_ActuatorMsg_t      command;
    CFE_MSG_Size_t             message_size = 0;
    CFE_Status_t               status;
    int                        i;

    if (CFE_MSG_GetSize(&buffer->Msg, &message_size) != CFE_SUCCESS || message_size != sizeof(*sensor))
    {
        ++OEL_ADCS_AppData.ErrorCount;
        CFE_EVS_SendEvent(OEL_ADCS_MESSAGE_ERR_EID, CFE_EVS_EventType_ERROR,
                          "OEL ADCS: rejected sensor message size=%lu expected=%lu",
                          (unsigned long)message_size, (unsigned long)sizeof(*sensor));
        return;
    }

    memset(&command, 0, sizeof(command));
    CFE_MSG_Init(CFE_MSG_PTR(command.CommandHeader), CFE_SB_ValueToMsgId(OEL_SIL_ACTUATOR_MID), sizeof(command));
    command.ValidTimeoutMs = OEL_ADCS_VALID_TIMEOUT_MS;
    if ((sensor->ValidFlags & 0x02U) != 0U)
    {
        command.Mode = 1;
        OEL_ADCS_ComputeTorque(sensor, command.TorqueBodyNm);
        for (i = 0; i < 3; ++i)
        {
            command.WheelTorqueNm[i] = command.TorqueBodyNm[i];
        }
    }

    status = CFE_SB_TransmitMsg(CFE_MSG_PTR(command.CommandHeader), true);
    if (status == CFE_SUCCESS)
    {
        ++OEL_ADCS_AppData.CommandCount;
    }
    else
    {
        ++OEL_ADCS_AppData.ErrorCount;
        CFE_EVS_SendEvent(OEL_ADCS_TRANSMIT_ERR_EID, CFE_EVS_EventType_ERROR,
                          "OEL ADCS: actuator transmit failed RC=0x%08lX", (unsigned long)status);
    }
    ++OEL_ADCS_AppData.SensorCount;
}

void OEL_ADCS_Main(void)
{
    CFE_Status_t     status;
    CFE_SB_Buffer_t *buffer;

    status = OEL_ADCS_Init();
    if (status != CFE_SUCCESS)
    {
        OEL_ADCS_AppData.RunStatus = CFE_ES_RunStatus_APP_ERROR;
    }
    while (CFE_ES_RunLoop(&OEL_ADCS_AppData.RunStatus))
    {
        status = CFE_SB_ReceiveBuffer(&buffer, OEL_ADCS_AppData.SensorPipe, CFE_SB_PEND_FOREVER);
        if (status == CFE_SUCCESS)
        {
            OEL_ADCS_ProcessSensor(buffer);
        }
        else
        {
            ++OEL_ADCS_AppData.ErrorCount;
            OEL_ADCS_AppData.RunStatus = CFE_ES_RunStatus_APP_ERROR;
            CFE_EVS_SendEvent(OEL_ADCS_PIPE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL ADCS: Software Bus receive failed RC=0x%08lX", (unsigned long)status);
        }
    }
    CFE_ES_ExitApp(OEL_ADCS_AppData.RunStatus);
}

CFE_Status_t OEL_ADCS_Init(void)
{
    CFE_Status_t status;

    memset(&OEL_ADCS_AppData, 0, sizeof(OEL_ADCS_AppData));
    OEL_ADCS_AppData.RunStatus = CFE_ES_RunStatus_APP_RUN;
    status = CFE_EVS_Register(NULL, 0, CFE_EVS_EventFilter_BINARY);
    if (status == CFE_SUCCESS)
    {
        status = CFE_SB_CreatePipe(&OEL_ADCS_AppData.SensorPipe, OEL_ADCS_PIPE_DEPTH, OEL_ADCS_PIPE_NAME);
    }
    if (status == CFE_SUCCESS)
    {
        status = CFE_SB_Subscribe(CFE_SB_ValueToMsgId(OEL_SIL_SENSOR_MID), OEL_ADCS_AppData.SensorPipe);
    }
    if (status == CFE_SUCCESS)
    {
        CFE_EVS_SendEvent(OEL_ADCS_INIT_INF_EID, CFE_EVS_EventType_INFORMATION,
                          "OEL reference ADCS initialized kp=%.3f kd=%.3f max_torque=%.3f N m",
                          OEL_ADCS_KP, OEL_ADCS_KD, OEL_ADCS_MAX_TORQUE_NM);
    }
    else
    {
        CFE_ES_WriteToSysLog("OEL ADCS initialization failed RC=0x%08lX\n", (unsigned long)status);
    }
    return status;
}
