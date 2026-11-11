#ifndef OEL_SIL_BRIDGE_CODEC_H
#define OEL_SIL_BRIDGE_CODEC_H

#include "oel_sil_bridge_msg.h"
#include <stddef.h>
#include <stdint.h>

#define OEL_SIL_WIRE_MAGIC_0 'R'
#define OEL_SIL_WIRE_MAGIC_1 'P'
#define OEL_SIL_WIRE_MAGIC_2 'O'
#define OEL_SIL_WIRE_MAGIC_3 '1'
#define OEL_SIL_WIRE_VERSION 1U

#define OEL_SIL_SENSOR_WIRE_ID   0x1900U
#define OEL_SIL_ACTUATOR_WIRE_ID 0x1901U

#define OEL_SIL_HEADER_WIRE_SIZE   20U
#define OEL_SIL_SENSOR_PAYLOAD_SIZE 145U
#define OEL_SIL_SENSOR_PACKET_SIZE  165U
#define OEL_SIL_ACTUATOR_PAYLOAD_SIZE 77U
#define OEL_SIL_ACTUATOR_PACKET_SIZE  97U

typedef struct
{
    uint16_t Version;
    uint16_t MessageId;
    uint32_t Sequence;
    uint64_t SimTimeNs;
} OEL_SIL_WireHeader_t;

int32 OEL_SIL_DecodeSensorPacket(const uint8_t *packet, size_t packet_size, OEL_SIL_WireHeader_t *header,
                                 OEL_SIL_SensorMsg_t *sensor);
int32 OEL_SIL_EncodeActuatorPacket(uint8_t *packet, size_t packet_capacity, uint32_t sequence,
                                  uint64_t sim_time_ns, const OEL_SIL_ActuatorMsg_t *actuator);

#endif
