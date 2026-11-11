#ifndef OEL_SIL_V2_CODEC_H
#define OEL_SIL_V2_CODEC_H

#include <stddef.h>
#include <stdint.h>

#define OEL_SIL_V2_HEADER_SIZE 16u
#define OEL_SIL_V2_TRAILER_SIZE 4u
#define OEL_SIL_V2_VERSION 2u

typedef struct
{
    uint8_t kind;
    uint32_t sequence;
    const uint8_t *payload;
    uint32_t payload_size;
} OEL_SIL_V2_PacketView_t;

int32_t OEL_SIL_V2_Encode(uint8_t *packet, size_t capacity, uint8_t kind,
                          uint32_t sequence, const uint8_t *payload,
                          uint32_t payload_size);

int32_t OEL_SIL_V2_Decode(const uint8_t *packet, size_t packet_size,
                          OEL_SIL_V2_PacketView_t *view);

#endif
