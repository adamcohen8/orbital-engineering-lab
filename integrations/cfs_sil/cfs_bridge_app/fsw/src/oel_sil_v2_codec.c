#include "oel_sil_v2_codec.h"

static void put_u32(uint8_t *out, uint32_t value)
{
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}

static uint32_t get_u32(const uint8_t *input)
{
    return ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) |
           ((uint32_t)input[2] << 8) | (uint32_t)input[3];
}

static uint32_t crc32_bytes(const uint8_t *data, uint32_t size)
{
    uint32_t crc = 0xFFFFFFFFu;
    uint32_t i;
    for (i = 0; i < size; ++i) {
        uint32_t byte = data[i];
        uint32_t bit;
        crc ^= byte;
        for (bit = 0; bit < 8u; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

int32_t OEL_SIL_V2_Encode(uint8_t *packet, size_t capacity, uint8_t kind,
                          uint32_t sequence, const uint8_t *payload,
                          uint32_t payload_size)
{
    uint32_t i;
    uint32_t total = OEL_SIL_V2_HEADER_SIZE + payload_size + OEL_SIL_V2_TRAILER_SIZE;
    if (packet == NULL || payload == NULL || capacity < total) return -1;
    packet[0] = 'O'; packet[1] = 'E'; packet[2] = 'L'; packet[3] = '2';
    packet[4] = OEL_SIL_V2_VERSION;
    packet[5] = kind;
    packet[6] = 0u; packet[7] = 0u;
    put_u32(&packet[8], sequence);
    put_u32(&packet[12], payload_size);
    for (i = 0; i < payload_size; ++i) packet[OEL_SIL_V2_HEADER_SIZE + i] = payload[i];
    put_u32(&packet[OEL_SIL_V2_HEADER_SIZE + payload_size], crc32_bytes(payload, payload_size));
    return (int32_t)total;
}

int32_t OEL_SIL_V2_Decode(const uint8_t *packet, size_t packet_size,
                          OEL_SIL_V2_PacketView_t *view)
{
    uint32_t payload_size;
    uint32_t expected_crc;
    uint32_t actual_crc;
    size_t expected_size;
    if (packet == NULL || view == NULL || packet_size < OEL_SIL_V2_HEADER_SIZE + OEL_SIL_V2_TRAILER_SIZE)
        return -1;
    if (packet[0] != 'O' || packet[1] != 'E' || packet[2] != 'L' || packet[3] != '2' ||
        packet[4] != OEL_SIL_V2_VERSION || packet[6] != 0u || packet[7] != 0u)
        return -2;
    payload_size = get_u32(&packet[12]);
    expected_size = (size_t)OEL_SIL_V2_HEADER_SIZE + payload_size + OEL_SIL_V2_TRAILER_SIZE;
    if (packet_size != expected_size)
        return -3;
    expected_crc = get_u32(&packet[OEL_SIL_V2_HEADER_SIZE + payload_size]);
    actual_crc = crc32_bytes(&packet[OEL_SIL_V2_HEADER_SIZE], payload_size);
    if (actual_crc != expected_crc)
        return -4;
    view->kind = packet[5];
    view->sequence = get_u32(&packet[8]);
    view->payload = &packet[OEL_SIL_V2_HEADER_SIZE];
    view->payload_size = payload_size;
    return 0;
}
