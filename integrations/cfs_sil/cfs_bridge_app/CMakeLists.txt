project(OEL_SIL_BRIDGE C)

add_cfe_app(oel_sil_bridge
  fsw/src/oel_sil_bridge_app.c
  fsw/src/oel_sil_bridge_codec.c
  fsw/src/oel_sil_v2_codec.c
)

target_include_directories(oel_sil_bridge PUBLIC
  fsw/inc
  config
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "oel_sil_bridge currently supports the native Linux cFS target only")
endif()
