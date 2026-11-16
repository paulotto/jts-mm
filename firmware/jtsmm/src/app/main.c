// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file main.c
 * @brief Firmware entry: init then enter comms_core_main.
 */

#include "jtsmm/app/sensor_core.h"
#include "jtsmm/app/comms_core.h"
#include "jtsmm/app/fault.h"
#include "jtsmm/runtime/usb_service.h"

#include "pico/stdlib.h"

/*
 * main() runs on physical core 0 (the Pico SDK boot core) and becomes the comms
 * core. The sensor core (sensor_core.c) is launched later on physical core 1 by
 * the comms core (comms_core_main), once the platform buses/DMA are initialized
 * by the first config apply. This avoids a race where the sensor core runs while
 * the comms core is still bringing the platform up.
 */
int main(void) {
    usb_service_init();
    fault_handler_init();
    sensor_core_init();
    comms_core_main();
    return 0;
}
