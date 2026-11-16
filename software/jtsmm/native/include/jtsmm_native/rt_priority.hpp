// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file rt_priority.hpp
 * @brief Best-effort SCHED_FIFO for capture threads (Linux).
 *
 * @details Promptly draining the USB vendor endpoint keeps the MCU EMG ring
 *          from overflowing under host load. Requires CAP_SYS_NICE or
 *          RLIMIT_RTPRIO; failures leave the default scheduling class.
 */
#pragma once

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace jtsmm {

#if defined(__linux__)
/**
 * @brief Try to set SCHED_FIFO for the calling thread.
 * @param priority FIFO priority (typically 1–49 for unprivileged RTPRIO).
 * @note Failures are ignored on purpose.
 */
inline void try_realtime_priority(int priority) {
    sched_param param{};
    param.sched_priority = priority;
    (void)pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
}
#else
/** @brief No-op on non-Linux hosts. */
inline void try_realtime_priority(int /*priority*/) {}
#endif

}  // namespace jtsmm
