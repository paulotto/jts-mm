# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.13)

set(PICO_PLATFORM rp2350 CACHE STRING "Pico platform" FORCE)
set(PICO_BOARD pico2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(pico_sdk_import.cmake)

project(jtsmm C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

set(JTSMM_SOURCES
    src/app/main.c
    src/app/sensor_core.c
    src/app/comms_core.c
    src/app/fault.c
    src/app/selftest.c
    src/runtime/timebase.c
    src/runtime/ringbuf.c
    src/runtime/host_link.c
    src/runtime/config_apply.c
    src/runtime/config_flash.c
    src/runtime/config_json.c
    src/runtime/usb_service.c
    src/runtime/cmd_parser.c
    src/runtime/capture_session.c
    src/platform/usb/usb_descriptors.c
    src/platform/platform_config.c
    src/platform/gpio_irq.c
    src/platform/dma_irq.c
    src/platform/spi_dma.c
    src/platform/i2c_dma.c
    src/drivers/ads1299.c
    src/drivers/bno085.c
    src/drivers/imu_bus.c
    src/drivers/trigger_pio.c
    src/drivers/trigger_timing.c
    src/drivers/led_flash_pio.c
    src/drivers/panel.c
    src/domain/config_defaults.c
    src/domain/emg_config.c
    src/domain/imu_config.c
    src/domain/imu_timing.c
    src/domain/bno085_reports.c
    src/domain/trigger_config.c
)

add_executable(jtsmm ${JTSMM_SOURCES})

# Warnings for our own sources only: pico-sdk INTERFACE libraries compile into
# this target too, and cJSON is third-party, so target-wide flags would drown
# our warnings in theirs. No -Werror yet: some driver files still warn.
set_source_files_properties(${JTSMM_SOURCES} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra")

pico_generate_pio_header(jtsmm ${CMAKE_CURRENT_LIST_DIR}/pio/ws2812.pio)
pico_generate_pio_header(jtsmm ${CMAKE_CURRENT_LIST_DIR}/pio/trigger_gen.pio)
pico_generate_pio_header(jtsmm ${CMAKE_CURRENT_LIST_DIR}/pio/led_flash.pio)

# Core 0 runs the command parser, cJSON and printf-heavy reporters: 4 KB stack
# (the SDK default 2 KB was exceeded by two 1.5 KB dump buffers in one frame).
# Core 1 inherits the same size. Each fills its 4 KB scratch bank exactly.
# cJSON recursion is bounded well below its default 1000 levels.
target_compile_definitions(jtsmm PRIVATE
    PICO_STACK_SIZE=0x1000
    CJSON_NESTING_LIMIT=32
)

# Hardware watchdog timeout in ms; 0 = disabled (default: not yet validated on
# the PCB). Enable with -DJTSMM_WATCHDOG_MS=4000; see include/jtsmm/app/fault.h.
set(JTSMM_WATCHDOG_MS 0 CACHE STRING "Hardware watchdog timeout (ms), 0 = off")
target_compile_definitions(jtsmm PRIVATE JTSMM_WATCHDOG_MS=${JTSMM_WATCHDOG_MS})

# IMU sample timestamps (include/jtsmm/domain/imu_timing.h). Set either to 0 to
# fall back to the previous behaviour without code edits: INT_EDGE=0 stamps a
# read with the poll time instead of the INT falling edge; SH2_DELAY=0 gives
# every report of a packet that time instead of its SH-2 sample time.
set(JTSMM_IMU_TS_USE_INT_EDGE 1 CACHE STRING "IMU t_int from the INT edge IRQ (1) or poll time (0)")
set(JTSMM_IMU_TS_USE_SH2_DELAY 1 CACHE STRING "IMU per-report SH-2 delay correction (1/0)")
target_compile_definitions(jtsmm PRIVATE
    IMU_TS_USE_INT_EDGE=${JTSMM_IMU_TS_USE_INT_EDGE}
    IMU_TS_USE_SH2_DELAY=${JTSMM_IMU_TS_USE_SH2_DELAY}
)

# Shared IRQ handler slots (SDK default 4) are all taken now: USB (TinyUSB),
# the core-0 GPIO callback, the DMA dispatcher and the core-1 IMU INT raw
# handler. The SDK hard-asserts when the table is full; leave headroom.
target_compile_definitions(jtsmm PRIVATE PICO_MAX_SHARED_IRQ_HANDLERS=8)

target_include_directories(jtsmm PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/third_party/cjson
)

target_sources(jtsmm PRIVATE
    third_party/cjson/cJSON.c
)

target_link_libraries(jtsmm
    pico_stdlib
    pico_multicore
    pico_flash
    hardware_spi
    hardware_i2c
    hardware_gpio
    hardware_pio
    hardware_dma
    hardware_irq
    hardware_timer
    hardware_sync
    hardware_flash
    hardware_watchdog
    pico_unique_id
    tinyusb_device
)

pico_enable_stdio_uart(jtsmm 0)

pico_add_extra_outputs(jtsmm)
