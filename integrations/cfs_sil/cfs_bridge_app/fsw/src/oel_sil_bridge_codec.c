#include "oel_sil_bridge_codec.h"
#include <string.h>

static uint16_t OEL_SIL_ReadU16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t OEL_SIL_ReadU32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t OEL_SIL_ReadU64(const uint8_t *p)
{
    uint64_t value = 0;
    int      i;
    for (i = 7; i >= 0; --i)
    {
        value = (value << 8) | p[i];
    }
    return value;
}

static double OEL_SIL_ReadF64(const uint8_t *p)
{
    uint64_t bits = OEL_SIL_ReadU64(p);
    double   value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void OEL_SIL_WriteU16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void OEL_SIL_WriteU32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void OEL_SIL_WriteU64(uint8_t *p, uint64_t value)
{
    int i;
    for (i = 0; i < 8; ++i)
    {
        p[i] = (uint8_t)(value >> (8 * i));
    }
}

static void OEL_SIL_WriteF64(uint8_t *p, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    OEL_SIL_WriteU64(p, bits);
}

static int32 OEL_SIL_DecodeHeader(const uint8_t *packet, size_t packet_size, uint16_t expected_id,
                                  size_t expected_size, OEL_SIL_WireHeader_t *header)
{
    if (packet == NULL || header == NULL || packet_size != expected_size)
    {
        return -1;
    }
    if (packet[0] != OEL_SIL_WIRE_MAGIC_0 || packet[1] != OEL_SIL_WIRE_MAGIC_1 ||
        packet[2] != OEL_SIL_WIRE_MAGIC_2 || packet[3] != OEL_SIL_WIRE_MAGIC_3)
    {
        return -2;
    }

    header->Version   = OEL_SIL_ReadU16(packet + 4);
    header->MessageId = OEL_SIL_ReadU16(packet + 6);
    header->Sequence  = OEL_SIL_ReadU32(packet + 8);
    header->SimTimeNs = OEL_SIL_ReadU64(packet + 12);
    if (header->Version != OEL_SIL_WIRE_VERSION || header->MessageId != expected_id)
    {
        return -3;
    }
    return 0;
}

int32 OEL_SIL_DecodeSensorPacket(const uint8_t *packet, size_t packet_size, OEL_SIL_WireHeader_t *header,
                                 OEL_SIL_SensorMsg_t *sensor)
{
    const uint8_t *p;
    int            i;
    int32          status;

    if (sensor == NULL)
    {
        return -1;
    }
    status = OEL_SIL_DecodeHeader(packet, packet_size, OEL_SIL_SENSOR_WIRE_ID, OEL_SIL_SENSOR_PACKET_SIZE, header);
    if (status != 0)
    {
        return status;
    }

    p                  = packet + OEL_SIL_HEADER_WIRE_SIZE;
    sensor->ValidFlags = *p++;
    for (i = 0; i < 3; ++i, p += 8) sensor->PosEciKm[i] = OEL_SIL_ReadF64(p);
    for (i = 0; i < 3; ++i, p += 8) sensor->VelEciKmS[i] = OEL_SIL_ReadF64(p);
    for (i = 0; i < 4; ++i, p += 8) sensor->QuatBn[i] = OEL_SIL_ReadF64(p);
    for (i = 0; i < 3; ++i, p += 8) sensor->OmegaBodyRadS[i] = OEL_SIL_ReadF64(p);
    sensor->MassKg = OEL_SIL_ReadF64(p);
    p += 8;
    for (i = 0; i < 3; ++i, p += 8) sensor->SunDirEci[i] = OEL_SIL_ReadF64(p);
    sensor->DensityKgM3 = OEL_SIL_ReadF64(p);
    return 0;
}

int32 OEL_SIL_EncodeActuatorPacket(uint8_t *packet, size_t packet_capacity, uint32_t sequence,
                                  uint64_t sim_time_ns, const OEL_SIL_ActuatorMsg_t *actuator)
{
    uint8_t *p;
    int      i;

    if (packet == NULL || actuator == NULL || packet_capacity < OEL_SIL_ACTUATOR_PACKET_SIZE)
    {
        return -1;
    }
    packet[0] = OEL_SIL_WIRE_MAGIC_0;
    packet[1] = OEL_SIL_WIRE_MAGIC_1;
    packet[2] = OEL_SIL_WIRE_MAGIC_2;
    packet[3] = OEL_SIL_WIRE_MAGIC_3;
    OEL_SIL_WriteU16(packet + 4, OEL_SIL_WIRE_VERSION);
    OEL_SIL_WriteU16(packet + 6, OEL_SIL_ACTUATOR_WIRE_ID);
    OEL_SIL_WriteU32(packet + 8, sequence);
    OEL_SIL_WriteU64(packet + 12, sim_time_ns);

    p    = packet + OEL_SIL_HEADER_WIRE_SIZE;
    *p++ = actuator->Mode;
    for (i = 0; i < 3; ++i, p += 8) OEL_SIL_WriteF64(p, actuator->ThrustEciKmS2[i]);
    for (i = 0; i < 3; ++i, p += 8) OEL_SIL_WriteF64(p, actuator->TorqueBodyNm[i]);
    for (i = 0; i < 3; ++i, p += 8) OEL_SIL_WriteF64(p, actuator->WheelTorqueNm[i]);
    OEL_SIL_WriteU32(p, actuator->ValidTimeoutMs);
    return (int32)OEL_SIL_ACTUATOR_PACKET_SIZE;
}
