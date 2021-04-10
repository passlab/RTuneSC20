cmake_minimum_required(VERSION 3.2)
project(rtune)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE Debug)

set(OMPT_SUPPORT TRUE)
if (OMPT_SUPPORT)
    set(OMP_LIB_PATH ${OMP_STANDALONE_INSTALL}/lib)
    set(OMP_INCLUDE ${OMP_STANDALONE_INSTALL}/include)
endif()
#set(PAPI_INSTALL /usr) # on carina

#######################################################################################################################
################################ Feature config #######################################################################
#######################################################################################################################
# For tracing
#set(OMPT_TRACING_SUPPORT TRUE) # Tracing all the event and save them as record
#set(OMPT_ONLINE_TRACING_PRINT TRUE) # Print the traces online, not used anymore

set(RTUNE_MEASUREMENT_SUPPORT TRUE) # Enable measurement of timing. For enabling energy and PAPI counters, see below
set(RTUNE_AUTOTUNING TRUE)
#set(CPUFREQ_SUPPORT TRUE)
# For additional measurement, PE (power and energy) measurement needs root
#set(PAPI_MEASUREMENT_SUPPORT TRUE) # PAPI measurement
set(PE_MEASUREMENT_SUPPORT FALSE) # Power and energy measurement

# For optimization
#set(PE_OPTIMIZATION_SUPPORT TRUE)
#set(PE_OPTIMIZATION_DVFS TRUE)
#######################################################################################################################

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rtune_config.h.cmake "${CMAKE_CURRENT_BINARY_DIR}/rtune_config.h" @ONLY)

set(SOURCE_FILES
    rtune_runtime.h
    rtune_runtime.c
    measurement.c
    measurement.h
)


add_library(rtune SHARED ${SOURCE_FILES})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(rtune m)
#target_link_libraries(rtune unwind)

if (CPUFREQ_SUPPORT)
    target_link_libraries(rtune cpufreq)
endif()

set(rtune_targets rtune)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rtune_config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/rtune.h
        DESTINATION include)

install(TARGETS ${rtune_targets}
        LIBRARY DESTINATION lib
        )
