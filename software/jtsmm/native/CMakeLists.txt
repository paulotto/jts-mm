# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(jtsmm_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# HDF5 session writer (M8b) is compiled in when its source is present so the
# module can be built incrementally.
set(JTSMM_NATIVE_SOURCES
  src/bulk_reader.cpp
  src/mcu_capture.cpp
  src/record_decode.cpp
  src/module.cpp
)
set(JTSMM_HDF5_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/hdf5_session.cpp")
if(EXISTS "${JTSMM_HDF5_SOURCE}")
  list(APPEND JTSMM_NATIVE_SOURCES src/hdf5_session.cpp)
  find_package(HDF5 REQUIRED COMPONENTS C)
  set(JTSMM_HAVE_HDF5 ON)
endif()

pybind11_add_module(_jtsmm_native ${JTSMM_NATIVE_SOURCES})

target_include_directories(_jtsmm_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LIBUSB_INCLUDE_DIRS}
)
target_link_libraries(_jtsmm_native PRIVATE ${LIBUSB_LINK_LIBRARIES})
target_link_directories(_jtsmm_native PRIVATE ${LIBUSB_LIBRARY_DIRS})

if(JTSMM_HAVE_HDF5)
  target_compile_definitions(_jtsmm_native PRIVATE JTSMM_HAVE_HDF5=1)
  target_include_directories(_jtsmm_native PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(_jtsmm_native PRIVATE ${HDF5_C_LIBRARIES})
endif()

# Drop the built extension next to the Python package so `import jtsmm.native`
# finds it without an install step.
set(JTSMM_PKG_NATIVE "${CMAKE_CURRENT_SOURCE_DIR}/../jtsmm/native")
set_target_properties(_jtsmm_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${JTSMM_PKG_NATIVE}"
)
