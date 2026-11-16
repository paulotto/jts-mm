// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

#ifdef HOST_TEST
#define __dmb() __sync_synchronize()
#else
#define __dmb() __asm__ volatile("dmb" ::: "memory")
#endif
#define __compiler_memory_barrier() __asm__ volatile("" ::: "memory")

static inline uint32_t save_and_disable_interrupts(void) {
    return 0u;
}
static inline void restore_interrupts(uint32_t status) {
    (void)status;
}
