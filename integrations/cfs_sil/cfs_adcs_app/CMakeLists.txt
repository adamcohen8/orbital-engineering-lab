project(OEL_ADCS_APP C)

add_cfe_app(oel_adcs_app fsw/src/oel_adcs_app.c)

target_include_directories(oel_adcs_app PUBLIC
  fsw/inc
  ../oel_sil_bridge/fsw/inc
  ../oel_sil_bridge/config
)

target_link_libraries(oel_adcs_app m)
