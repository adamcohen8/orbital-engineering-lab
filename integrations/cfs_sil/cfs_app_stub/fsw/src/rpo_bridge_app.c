#include "rpo_bridge_app.h"
#include <math.h>

RPO_BRIDGE_AppData_t RPO_BRIDGE_AppData;

static double RPO_BRIDGE_Vec3Norm(const double v[3])
{
    return sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
}

static void RPO_BRIDGE_NormalizeQuat(double q[4])
{
    double n = sqrt((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]) + (q[3] * q[3]));

    if (n <= 0.0)
    {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
        return;
    }

    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
}

static void RPO_BRIDGE_QuatConjugate(const double q[4], double out[4])
{
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

static void RPO_BRIDGE_QuatMultiply(const double a[4], const double b[4], double out[4])
{
    out[0] = (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3]);
    out[1] = (a[0] * b[1]) + (a[1] * b[0]) + (a[2] * b[3]) - (a[3] * b[2]);
    out[2] = (a[0] * b[2]) - (a[1] * b[3]) + (a[2] * b[0]) + (a[3] * b[1]);
    out[3] = (a[0] * b[3]) + (a[1] * b[2]) - (a[2] * b[1]) + (a[3] * b[0]);
    RPO_BRIDGE_NormalizeQuat(out);
}

static void RPO_BRIDGE_ComputeAttitudeHoldTorque(const RPO_BRIDGE_SimSensor_t *sensor, double torque_body_nm[3])
{
    double q_cur[4];
    double q_des[4];
    double q_des_conj[4];
    double q_err[4];
    double omega_err[3];
    double torque_norm;
    double scale;
    int i;

    for (i = 0; i < 4; ++i)
    {
        q_cur[i] = sensor->QuatBn[i];
        q_des[i] = RPO_BRIDGE_AppData.DesiredQuatBn[i];
    }
    RPO_BRIDGE_NormalizeQuat(q_cur);
    RPO_BRIDGE_NormalizeQuat(q_des);

    if ((RPO_BRIDGE_ATT_HOLD_CAPTURE_INITIAL != 0) && (!RPO_BRIDGE_AppData.AttHoldTargetCaptured))
    {
        for (i = 0; i < 4; ++i)
        {
            RPO_BRIDGE_AppData.DesiredQuatBn[i] = q_cur[i];
            q_des[i] = q_cur[i];
        }
        RPO_BRIDGE_AppData.AttHoldTargetCaptured = true;
    }

    RPO_BRIDGE_QuatConjugate(q_des, q_des_conj);
    RPO_BRIDGE_QuatMultiply(q_des_conj, q_cur, q_err);
    if (q_err[0] < 0.0)
    {
        for (i = 0; i < 4; ++i)
        {
            q_err[i] = -q_err[i];
        }
    }

    for (i = 0; i < 3; ++i)
    {
        omega_err[i] = sensor->OmegaBodyRadS[i] - RPO_BRIDGE_AppData.DesiredOmegaBodyRadS[i];
        torque_body_nm[i] = (-RPO_BRIDGE_ATT_HOLD_KP * q_err[i + 1]) - (RPO_BRIDGE_ATT_HOLD_KD * omega_err[i]);
    }

    torque_norm = RPO_BRIDGE_Vec3Norm(torque_body_nm);
    if ((RPO_BRIDGE_ATT_HOLD_MAX_TORQUE_NM > 0.0) && (torque_norm > RPO_BRIDGE_ATT_HOLD_MAX_TORQUE_NM))
    {
        scale = RPO_BRIDGE_ATT_HOLD_MAX_TORQUE_NM / torque_norm;
        for (i = 0; i < 3; ++i)
        {
            torque_body_nm[i] *= scale;
        }
    }
}

void RPO_BRIDGE_AppMain(void)
{
    int32 status;
    const CFE_SB_Buffer_t *SBBufPtr;

    RPO_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_RUN;
    status                       = RPO_BRIDGE_AppInit();

    if (status != CFE_SUCCESS)
    {
        RPO_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_ERROR;
    }

    while (CFE_ES_RunLoop(&RPO_BRIDGE_AppData.RunStatus))
    {
        status = CFE_SB_ReceiveBuffer(&SBBufPtr, RPO_BRIDGE_AppData.SensorPipe, CFE_SB_PEND_FOREVER);
        if (status == CFE_SUCCESS)
        {
            RPO_BRIDGE_ProcessSensorPacket(SBBufPtr);
        }
    }
}

int32 RPO_BRIDGE_AppInit(void)
{
    int32 status;

    CFE_ES_RegisterApp();
    CFE_PSP_MemSet(&RPO_BRIDGE_AppData, 0, sizeof(RPO_BRIDGE_AppData));
    RPO_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_RUN;

    status = CFE_SB_CreatePipe(&RPO_BRIDGE_AppData.SensorPipe, RPO_BRIDGE_PIPE_DEPTH, RPO_BRIDGE_PIPE_NAME);
    if (status != CFE_SUCCESS)
    {
        return status;
    }

    status = CFE_SB_Subscribe(RPO_BRIDGE_SIM_SENSOR_MID, RPO_BRIDGE_AppData.SensorPipe);
    if (status != CFE_SUCCESS)
    {
        return status;
    }

    RPO_BRIDGE_ResetAttitudeHoldTarget();

    return CFE_SUCCESS;
}

void RPO_BRIDGE_ProcessSensorPacket(const CFE_SB_Buffer_t *SBBufPtr)
{
    const RPO_BRIDGE_SimSensor_t *sensor = (const RPO_BRIDGE_SimSensor_t *)SBBufPtr;
    RPO_BRIDGE_ActCmd_t *cmd             = &RPO_BRIDGE_AppData.LastActCmd;
    int i;

    CFE_PSP_MemSet(cmd, 0, sizeof(*cmd));
    cmd->ValidTimeoutMs = RPO_BRIDGE_ACT_CMD_VALID_TIMEOUT_MS;
    if ((sensor->ValidFlags & 0x02) != 0)
    {
        cmd->Mode = 1;
        RPO_BRIDGE_ComputeAttitudeHoldTorque(sensor, cmd->TorqueBodyNm);
        for (i = 0; i < 3; ++i)
        {
            cmd->WheelTorqueNm[i] = cmd->TorqueBodyNm[i];
        }
    }
    else
    {
        cmd->Mode = 0;
        RPO_BRIDGE_AppData.ErrCounter++;
    }

    CFE_MSG_Init(CFE_MSG_PTR(cmd), RPO_BRIDGE_ACT_CMD_MID, sizeof(RPO_BRIDGE_ActCmd_t));
    CFE_SB_TransmitMsg(CFE_MSG_PTR(cmd), true);
    RPO_BRIDGE_AppData.CmdCounter++;
}

void RPO_BRIDGE_ProcessCommandPacket(const CFE_SB_Buffer_t *SBBufPtr)
{
    (void)SBBufPtr;
}

void RPO_BRIDGE_ResetAttitudeHoldTarget(void)
{
    RPO_BRIDGE_AppData.DesiredQuatBn[0] = RPO_BRIDGE_ATT_HOLD_Q0;
    RPO_BRIDGE_AppData.DesiredQuatBn[1] = RPO_BRIDGE_ATT_HOLD_Q1;
    RPO_BRIDGE_AppData.DesiredQuatBn[2] = RPO_BRIDGE_ATT_HOLD_Q2;
    RPO_BRIDGE_AppData.DesiredQuatBn[3] = RPO_BRIDGE_ATT_HOLD_Q3;
    RPO_BRIDGE_NormalizeQuat(RPO_BRIDGE_AppData.DesiredQuatBn);

    RPO_BRIDGE_AppData.DesiredOmegaBodyRadS[0] = RPO_BRIDGE_ATT_HOLD_W0_RAD_S;
    RPO_BRIDGE_AppData.DesiredOmegaBodyRadS[1] = RPO_BRIDGE_ATT_HOLD_W1_RAD_S;
    RPO_BRIDGE_AppData.DesiredOmegaBodyRadS[2] = RPO_BRIDGE_ATT_HOLD_W2_RAD_S;
    RPO_BRIDGE_AppData.AttHoldTargetCaptured = false;
}
