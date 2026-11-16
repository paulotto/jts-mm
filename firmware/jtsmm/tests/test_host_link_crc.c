// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#include "jtsmm/runtime/host_link.h"

#include "check.h"

#include <string.h>

/*
 * Pin the frame CRC to CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no
 * reflection, no final XOR) -- the variant the host decoders implement
 * (software/jtsmm/jtsmm/protocol/crc.py and the native module). The old test
 * only checked crc != 0, which any CRC variant (or a broken loop) passes.
 */
int main(void) {
    const char *check_str = "123456789";
    CHECK_EQ_U(host_link_crc16((const uint8_t *)check_str, strlen(check_str)), 0x29B1u);
    CHECK_EQ_U(host_link_crc16(NULL, 0), 0xFFFFu); /* init value, nothing mixed in */

    /* Body of a CMD_ACK frame (type, len, t_master_us LE, payload): what the
     * CRC covers, i.e. everything after the two sync bytes. */
    const uint8_t body[] = {0xE0, 0x01, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x01};
    CHECK_EQ_U(host_link_crc16(body, sizeof(body)), 0x8F26u);

    CHECK_DONE("test_host_link_crc");
}
