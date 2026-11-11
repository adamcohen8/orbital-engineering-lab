#include "oel_sil_bridge_app.h"
#include "oel_sil_bridge_eventids.h"
#include "oel_sil_bridge_platform_cfg.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

OEL_SIL_BRIDGE_AppData_t OEL_SIL_BRIDGE_AppData;

static const char *OEL_SIL_BRIDGE_FindLast(const char *text, const char *needle)
{
    const char *found = NULL;
    const char *cursor = text;
    while ((cursor = strstr(cursor, needle)) != NULL)
    {
        found = cursor;
        ++cursor;
    }
    return found;
}

static size_t OEL_SIL_BRIDGE_Count(const char *text, const char *needle)
{
    size_t count = 0U;
    const char *cursor = text;
    while ((cursor = strstr(cursor, needle)) != NULL)
    {
        ++count;
        cursor += strlen(needle);
    }
    return count;
}

static int32 OEL_SIL_BRIDGE_IsSafeIdentifier(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    if (cursor == NULL || *cursor == '\0')
    {
        return -1;
    }
    for (; *cursor != '\0'; ++cursor)
    {
        if (!(isalnum(*cursor) || *cursor == '-' || *cursor == '_' || *cursor == '.' || *cursor == '/'))
        {
            return -1;
        }
    }
    return 0;
}

static int32 OEL_SIL_BRIDGE_ReadJsonString(const char *start, const char *key, char *value, size_t capacity)
{
    const char *cursor = strstr(start, key);
    const char *end;
    size_t      size;
    if (cursor == NULL || value == NULL || capacity == 0U)
    {
        return -1;
    }
    cursor += strlen(key);
    end = strchr(cursor, '"');
    if (end == NULL)
    {
        return -1;
    }
    size = (size_t)(end - cursor);
    if (size >= capacity)
    {
        return -1;
    }
    memcpy(value, cursor, size);
    value[size] = '\0';
    return OEL_SIL_BRIDGE_IsSafeIdentifier(value);
}

static int32 OEL_SIL_BRIDGE_ReadJsonU64(const char *start, const char *key, uint64 *value)
{
    const char         *cursor = strstr(start, key);
    char               *end = NULL;
    unsigned long long parsed;
    if (cursor == NULL || value == NULL)
    {
        return -1;
    }
    cursor += strlen(key);
    errno = 0;
    parsed = strtoull(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || (*end != ',' && *end != '}'))
    {
        return -1;
    }
    *value = (uint64)parsed;
    return 0;
}

static int32 OEL_SIL_BRIDGE_ReadJsonVector(const char *start, const char *key, double *values, size_t count)
{
    const char *cursor = strstr(start, key);
    char       *end = NULL;
    size_t      index;
    if (cursor == NULL || values == NULL)
    {
        return -1;
    }
    cursor += strlen(key);
    for (index = 0; index < count; ++index)
    {
        values[index] = strtod(cursor, &end);
        if (end == cursor || !isfinite(values[index]))
        {
            return -1;
        }
        cursor = end;
        if (index + 1U < count)
        {
            cursor = strchr(cursor, ',');
            if (cursor == NULL)
            {
                return -1;
            }
            ++cursor;
        }
    }
    if (*cursor != ']')
    {
        return -1;
    }
    return 0;
}

static int32 OEL_SIL_BRIDGE_SendV2(uint8 kind, uint32 sequence, const char *payload)
{
    uint8_t packet[OEL_SIL_BRIDGE_MAX_DATAGRAM];
    int32_t packet_size;
    ssize_t sent;
    packet_size = OEL_SIL_V2_Encode(packet, sizeof(packet), kind, sequence,
                                    (const uint8_t *)payload, (uint32_t)strlen(payload));
    if (packet_size < 0)
    {
        return -1;
    }
    sent = sendto(OEL_SIL_BRIDGE_AppData.SocketFd, packet, (size_t)packet_size, 0,
                  (struct sockaddr *)&OEL_SIL_BRIDGE_AppData.OelAddress,
                  sizeof(OEL_SIL_BRIDGE_AppData.OelAddress));
    return sent == packet_size ? 0 : -1;
}

static int32 OEL_SIL_BRIDGE_DecodeInputBatch(const OEL_SIL_V2_PacketView_t *view,
                                             OEL_SIL_SensorMsg_t *sensor)
{
    char        json[OEL_SIL_BRIDGE_MAX_DATAGRAM + 1U];
    const char *invocation;
    const char *satellite;
    uint64      reset = 0;
    int         i;
    if (view->payload_size > OEL_SIL_BRIDGE_MAX_DATAGRAM)
    {
        return -1;
    }
    memcpy(json, view->payload, view->payload_size);
    json[view->payload_size] = '\0';
    if (json[0] != '{' || view->payload_size < 2U || json[view->payload_size - 1U] != '}' ||
        strchr(json, '\\') != NULL ||
        OEL_SIL_BRIDGE_Count(json, "\"schema\":\"oel.flight_software.input_batch.v1\"") != 1U ||
        OEL_SIL_BRIDGE_Count(json, "\"invocation_id\":") != 1U ||
        strstr(json, "\"contract_version\":\"1.0\"") == NULL ||
        strstr(json, "\"events\":[{\"delivery_time\":") == NULL ||
        strstr(json, "\"kind\":\"measurement\"") == NULL ||
        strstr(json, "\"measurement_type\":\"ideal_own_state.v1\"") == NULL ||
        strstr(json, "\"quality\":{\"status_codes\":[],\"validity\":\"valid\"}") == NULL ||
        strstr(json, "}],\"invocation_id\":") == NULL ||
        OEL_SIL_BRIDGE_ReadJsonVector(json, "\"attitude_quat_body_from_inertial\":[", sensor->QuatBn, 4U) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonVector(json, "\"angular_rate_body_rad_s\":[", sensor->OmegaBodyRadS, 3U) != 0)
    {
        return -2;
    }
    invocation = strstr(json, "\"invocation_time\":{");
    satellite = OEL_SIL_BRIDGE_FindLast(json, "\"satellite_id\":\"");
    if (invocation == NULL || satellite == NULL ||
        OEL_SIL_BRIDGE_ReadJsonU64(json, "\"invocation_id\":", &OEL_SIL_BRIDGE_AppData.PendingInvocationId) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonU64(invocation, "\"ticks\":", &OEL_SIL_BRIDGE_AppData.PendingClockTicks) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonU64(invocation, "\"tick_period_ns\":", &OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonU64(invocation, "\"reset_counter\":", &reset) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonString(invocation, "\"clock_id\":\"", OEL_SIL_BRIDGE_AppData.ClockId,
                                      sizeof(OEL_SIL_BRIDGE_AppData.ClockId)) != 0 ||
        OEL_SIL_BRIDGE_ReadJsonString(satellite, "\"satellite_id\":\"", OEL_SIL_BRIDGE_AppData.SatelliteId,
                                      sizeof(OEL_SIL_BRIDGE_AppData.SatelliteId)) != 0)
    {
        return -3;
    }
    OEL_SIL_BRIDGE_AppData.PendingClockReset = (uint32)reset;
    sensor->ValidFlags = 0x02U;
    for (i = 0; i < 3; ++i)
    {
        sensor->PosEciKm[i] = 0.0;
        sensor->VelEciKmS[i] = 0.0;
        sensor->SunDirEci[i] = 0.0;
    }
    sensor->MassKg = 0.0;
    sensor->DensityKgM3 = 0.0;
    return 0;
}

static CFE_Status_t OEL_SIL_BRIDGE_OpenSocket(void)
{
    struct sockaddr_in bind_address;
    int                flags;

    OEL_SIL_BRIDGE_AppData.SocketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (OEL_SIL_BRIDGE_AppData.SocketFd < 0)
    {
        return CFE_STATUS_EXTERNAL_RESOURCE_FAIL;
    }

    flags = fcntl(OEL_SIL_BRIDGE_AppData.SocketFd, F_GETFL, 0);
    if (flags < 0 || fcntl(OEL_SIL_BRIDGE_AppData.SocketFd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        close(OEL_SIL_BRIDGE_AppData.SocketFd);
        OEL_SIL_BRIDGE_AppData.SocketFd = -1;
        return CFE_STATUS_EXTERNAL_RESOURCE_FAIL;
    }

    memset(&bind_address, 0, sizeof(bind_address));
    bind_address.sin_family = AF_INET;
    bind_address.sin_port   = htons(OEL_SIL_BRIDGE_BIND_PORT);
    if (inet_pton(AF_INET, OEL_SIL_BRIDGE_BIND_ADDRESS, &bind_address.sin_addr) != 1 ||
        bind(OEL_SIL_BRIDGE_AppData.SocketFd, (struct sockaddr *)&bind_address, sizeof(bind_address)) < 0)
    {
        close(OEL_SIL_BRIDGE_AppData.SocketFd);
        OEL_SIL_BRIDGE_AppData.SocketFd = -1;
        return CFE_STATUS_EXTERNAL_RESOURCE_FAIL;
    }

    memset(&OEL_SIL_BRIDGE_AppData.OelAddress, 0, sizeof(OEL_SIL_BRIDGE_AppData.OelAddress));
    OEL_SIL_BRIDGE_AppData.OelAddress.sin_family = AF_INET;
    OEL_SIL_BRIDGE_AppData.OelAddress.sin_port   = htons(OEL_SIL_BRIDGE_OEL_PORT);
    if (inet_pton(AF_INET, OEL_SIL_BRIDGE_OEL_ADDRESS, &OEL_SIL_BRIDGE_AppData.OelAddress.sin_addr) != 1)
    {
        close(OEL_SIL_BRIDGE_AppData.SocketFd);
        OEL_SIL_BRIDGE_AppData.SocketFd = -1;
        return CFE_STATUS_EXTERNAL_RESOURCE_FAIL;
    }
    return CFE_SUCCESS;
}

static void OEL_SIL_BRIDGE_ReceiveSensors(void)
{
    uint8_t                packet[OEL_SIL_BRIDGE_MAX_DATAGRAM];
    OEL_SIL_V2_PacketView_t view;
    OEL_SIL_SensorMsg_t     sensor;
    struct sockaddr_in      peer_address;
    socklen_t               peer_size;
    ssize_t                 received;
    int32                   status;

    while (true)
    {
        peer_size = sizeof(peer_address);
        received = recvfrom(OEL_SIL_BRIDGE_AppData.SocketFd, packet, sizeof(packet), 0,
                            (struct sockaddr *)&peer_address, &peer_size);
        if (received < 0)
        {
            if (errno != EAGAIN && errno != EWOULDBLOCK)
            {
                CFE_EVS_SendEvent(OEL_SIL_BRIDGE_SOCKET_ERR_EID, CFE_EVS_EventType_ERROR,
                                  "OEL SIL: recvfrom failed, errno=%d", errno);
            }
            return;
        }

        if (peer_address.sin_addr.s_addr != OEL_SIL_BRIDGE_AppData.OelAddress.sin_addr.s_addr ||
            peer_address.sin_port != OEL_SIL_BRIDGE_AppData.OelAddress.sin_port)
        {
            ++OEL_SIL_BRIDGE_AppData.DecodeErrorCount;
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_DECODE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: rejected datagram from unconfigured UDP source");
            continue;
        }

        status = OEL_SIL_V2_Decode(packet, (size_t)received, &view);
        if (status != 0)
        {
            ++OEL_SIL_BRIDGE_AppData.DecodeErrorCount;
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_DECODE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: rejected v2 datagram size=%ld status=%ld", (long)received, (long)status);
            continue;
        }
        if (view.kind != 3U && OEL_SIL_BRIDGE_AppData.HaveSensorSequence &&
            view.sequence != OEL_SIL_BRIDGE_AppData.LastSensorSequence + 1U)
        {
            ++OEL_SIL_BRIDGE_AppData.SequenceGapCount;
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_SEQUENCE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: request sequence discontinuity previous=%lu current=%lu",
                              (unsigned long)OEL_SIL_BRIDGE_AppData.LastSensorSequence,
                              (unsigned long)view.sequence);
            continue;
        }
        OEL_SIL_BRIDGE_AppData.HaveSensorSequence = true;
        OEL_SIL_BRIDGE_AppData.LastSensorSequence = view.sequence;
        if (view.kind == 3U || view.kind == 4U)
        {
            char json[OEL_SIL_BRIDGE_MAX_DATAGRAM + 1U];
            if (view.payload_size <= OEL_SIL_BRIDGE_MAX_DATAGRAM)
            {
                memcpy(json, view.payload, view.payload_size);
                json[view.payload_size] = '\0';
                if (view.kind == 3U)
                {
                    if (OEL_SIL_BRIDGE_ReadJsonString(json, "\"boot_id\":\"", OEL_SIL_BRIDGE_AppData.BootId,
                                                      sizeof(OEL_SIL_BRIDGE_AppData.BootId)) != 0)
                    {
                        ++OEL_SIL_BRIDGE_AppData.DecodeErrorCount;
                        continue;
                    }
                }
            }
            if (OEL_SIL_BRIDGE_SendV2(view.kind, view.sequence, "{}") != 0)
            {
                ++OEL_SIL_BRIDGE_AppData.SendErrorCount;
            }
            continue;
        }
        if (view.kind != 1U || OEL_SIL_BRIDGE_AppData.HavePendingInvocation)
        {
            ++OEL_SIL_BRIDGE_AppData.DecodeErrorCount;
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_DECODE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: unsupported or overlapping v2 request kind=%u", (unsigned int)view.kind);
            continue;
        }
        memset(&sensor, 0, sizeof(sensor));
        CFE_MSG_Init(CFE_MSG_PTR(sensor.TelemetryHeader), CFE_SB_ValueToMsgId(OEL_SIL_SENSOR_MID), sizeof(sensor));
        status = OEL_SIL_BRIDGE_DecodeInputBatch(&view, &sensor);
        if (status != 0)
        {
            ++OEL_SIL_BRIDGE_AppData.DecodeErrorCount;
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_DECODE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: invalid v2 input batch status=%ld", (long)status);
            continue;
        }
        OEL_SIL_BRIDGE_AppData.PendingWireSequence = view.sequence;
        OEL_SIL_BRIDGE_AppData.HavePendingInvocation = true;
        OEL_SIL_BRIDGE_AppData.LastSimTimeNs =
            OEL_SIL_BRIDGE_AppData.PendingClockTicks * OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs;
        status = CFE_SB_TransmitMsg(CFE_MSG_PTR(sensor.TelemetryHeader), true);
        if (status == CFE_SUCCESS)
        {
            ++OEL_SIL_BRIDGE_AppData.SensorPacketCount;
        }
        else
        {
            CFE_EVS_SendEvent(OEL_SIL_BRIDGE_MESSAGE_ERR_EID, CFE_EVS_EventType_ERROR,
                              "OEL SIL: sensor Software Bus transmit failed RC=0x%08lX", (unsigned long)status);
        }
    }
}

static void OEL_SIL_BRIDGE_SendActuator(const CFE_SB_Buffer_t *buffer)
{
    const OEL_SIL_ActuatorMsg_t *actuator = (const OEL_SIL_ActuatorMsg_t *)buffer;
    CFE_MSG_Size_t               message_size = 0;
    char                         payload[4096];
    int                          payload_size;
    uint64                       validity_ticks;

    if (CFE_MSG_GetSize(&buffer->Msg, &message_size) != CFE_SUCCESS || message_size != sizeof(*actuator))
    {
        ++OEL_SIL_BRIDGE_AppData.SendErrorCount;
        CFE_EVS_SendEvent(OEL_SIL_BRIDGE_MESSAGE_ERR_EID, CFE_EVS_EventType_ERROR,
                          "OEL SIL: rejected actuator SB message size=%lu expected=%lu",
                          (unsigned long)message_size, (unsigned long)sizeof(*actuator));
        return;
    }
    if (!OEL_SIL_BRIDGE_AppData.HavePendingInvocation || OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs == 0U)
    {
        ++OEL_SIL_BRIDGE_AppData.SendErrorCount;
        CFE_EVS_SendEvent(OEL_SIL_BRIDGE_MESSAGE_ERR_EID, CFE_EVS_EventType_ERROR,
                          "OEL SIL: actuator message has no pending v2 invocation");
        return;
    }
    validity_ticks = ((uint64)actuator->ValidTimeoutMs * 1000000ULL +
                      OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs - 1ULL) /
                     OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs;
    if (validity_ticks < 1U)
    {
        validity_ticks = 1U;
    }
    payload_size = snprintf(
        payload, sizeof(payload),
        "{\"commands\":[{\"actuator_id\":\"wrench\",\"command_id\":{\"boot_id\":\"%s\","
        "\"sequence\":%llu,\"source_id\":\"cfs/wrench\"},\"contract_version\":\"1.0\","
        "\"frame\":{\"name\":\"OEL/ACTUATOR/%s/wrench\",\"registry_version\":\"frames-v1\"},"
        "\"issued_at\":{\"clock_id\":\"%s\",\"reset_counter\":%lu,\"scale\":\"onboard\","
        "\"tick_period_ns\":%llu,\"ticks\":%llu,\"validity\":\"valid\"},"
        "\"payload\":{\"force_n\":[0.0,0.0,0.0],\"schema\":\"ideal_wrench.v1\","
        "\"torque_n_m\":[%.17g,%.17g,%.17g]},\"satellite_id\":\"%s\","
        "\"schema\":\"oel.flight_software.actuator_command.v1\",\"validity\":{"
        "\"expires_at\":{\"clock_id\":\"%s\",\"reset_counter\":%lu,\"scale\":\"onboard\","
        "\"tick_period_ns\":%llu,\"ticks\":%llu,\"validity\":\"valid\"},"
        "\"not_before\":{\"clock_id\":\"%s\",\"reset_counter\":%lu,\"scale\":\"onboard\","
        "\"tick_period_ns\":%llu,\"ticks\":%llu,\"validity\":\"valid\"}}}],"
        "\"contract_version\":\"1.0\",\"invocation_id\":%llu,\"requested_next_invocations\":[],"
        "\"satellite_id\":\"%s\",\"schema\":\"oel.flight_software.output.v1\",\"telemetry\":[]}",
        OEL_SIL_BRIDGE_AppData.BootId,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.CommandSequence,
        OEL_SIL_BRIDGE_AppData.SatelliteId,
        OEL_SIL_BRIDGE_AppData.ClockId,
        (unsigned long)OEL_SIL_BRIDGE_AppData.PendingClockReset,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingClockTicks,
        actuator->TorqueBodyNm[0], actuator->TorqueBodyNm[1], actuator->TorqueBodyNm[2],
        OEL_SIL_BRIDGE_AppData.SatelliteId,
        OEL_SIL_BRIDGE_AppData.ClockId,
        (unsigned long)OEL_SIL_BRIDGE_AppData.PendingClockReset,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs,
        (unsigned long long)(OEL_SIL_BRIDGE_AppData.PendingClockTicks + validity_ticks),
        OEL_SIL_BRIDGE_AppData.ClockId,
        (unsigned long)OEL_SIL_BRIDGE_AppData.PendingClockReset,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingTickPeriodNs,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingClockTicks,
        (unsigned long long)OEL_SIL_BRIDGE_AppData.PendingInvocationId,
        OEL_SIL_BRIDGE_AppData.SatelliteId);
    if (payload_size < 0 || (size_t)payload_size >= sizeof(payload) ||
        OEL_SIL_BRIDGE_SendV2(2U, OEL_SIL_BRIDGE_AppData.PendingWireSequence, payload) != 0)
    {
        ++OEL_SIL_BRIDGE_AppData.SendErrorCount;
        CFE_EVS_SendEvent(OEL_SIL_BRIDGE_SEND_ERR_EID, CFE_EVS_EventType_ERROR,
                          "OEL SIL: v2 actuator response failed errno=%d", errno);
        return;
    }
    OEL_SIL_BRIDGE_AppData.HavePendingInvocation = false;
    ++OEL_SIL_BRIDGE_AppData.CommandSequence;
    ++OEL_SIL_BRIDGE_AppData.ActuatorSequence;
    ++OEL_SIL_BRIDGE_AppData.ActuatorPacketCount;
}

void OEL_SIL_BRIDGE_Main(void)
{
    CFE_Status_t     status;
    CFE_SB_Buffer_t *buffer;

    status = OEL_SIL_BRIDGE_Init();
    if (status != CFE_SUCCESS)
    {
        OEL_SIL_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_ERROR;
    }

    while (CFE_ES_RunLoop(&OEL_SIL_BRIDGE_AppData.RunStatus))
    {
        OEL_SIL_BRIDGE_ReceiveSensors();
        while (CFE_SB_ReceiveBuffer(&buffer, OEL_SIL_BRIDGE_AppData.ActuatorPipe, CFE_SB_POLL) == CFE_SUCCESS)
        {
            OEL_SIL_BRIDGE_SendActuator(buffer);
        }
        OS_TaskDelay(OEL_SIL_BRIDGE_POLL_DELAY_MS);
    }

    OEL_SIL_BRIDGE_Cleanup();
    CFE_ES_ExitApp(OEL_SIL_BRIDGE_AppData.RunStatus);
}

CFE_Status_t OEL_SIL_BRIDGE_Init(void)
{
    CFE_Status_t status;

    memset(&OEL_SIL_BRIDGE_AppData, 0, sizeof(OEL_SIL_BRIDGE_AppData));
    OEL_SIL_BRIDGE_AppData.SocketFd = -1;
    OEL_SIL_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_RUN;
    strcpy(OEL_SIL_BRIDGE_AppData.SatelliteId, "satellite");
    strcpy(OEL_SIL_BRIDGE_AppData.BootId, "boot-0");
    strcpy(OEL_SIL_BRIDGE_AppData.ClockId, "satellite/onboard");

    status = CFE_EVS_Register(NULL, 0, CFE_EVS_EventFilter_BINARY);
    if (status == CFE_SUCCESS)
    {
        status = CFE_SB_CreatePipe(&OEL_SIL_BRIDGE_AppData.ActuatorPipe, OEL_SIL_BRIDGE_PIPE_DEPTH,
                                   OEL_SIL_BRIDGE_PIPE_NAME);
    }
    if (status == CFE_SUCCESS)
    {
        status = CFE_SB_Subscribe(CFE_SB_ValueToMsgId(OEL_SIL_ACTUATOR_MID), OEL_SIL_BRIDGE_AppData.ActuatorPipe);
    }
    if (status == CFE_SUCCESS)
    {
        status = OEL_SIL_BRIDGE_OpenSocket();
    }
    if (status == CFE_SUCCESS)
    {
        CFE_EVS_SendEvent(OEL_SIL_BRIDGE_INIT_INF_EID, CFE_EVS_EventType_INFORMATION,
                          "OEL SIL bridge initialized UDP %s:%u -> %s:%u",
                          OEL_SIL_BRIDGE_BIND_ADDRESS, OEL_SIL_BRIDGE_BIND_PORT,
                          OEL_SIL_BRIDGE_OEL_ADDRESS, OEL_SIL_BRIDGE_OEL_PORT);
    }
    else
    {
        CFE_ES_WriteToSysLog("OEL SIL bridge initialization failed RC=0x%08lX errno=%d\n",
                             (unsigned long)status, errno);
        OEL_SIL_BRIDGE_Cleanup();
    }
    return status;
}

void OEL_SIL_BRIDGE_Cleanup(void)
{
    if (OEL_SIL_BRIDGE_AppData.SocketFd >= 0)
    {
        close(OEL_SIL_BRIDGE_AppData.SocketFd);
        OEL_SIL_BRIDGE_AppData.SocketFd = -1;
    }
}
