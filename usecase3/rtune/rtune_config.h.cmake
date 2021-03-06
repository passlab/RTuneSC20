#ifndef RTUNE_CONFIG_H
#define RTUNE_CONFIG_H

#define RTUNE_VERSION_MAJOR    @RTUNE_VERSION_MAJOR@
#define RTUNE_VERSION_MINOR    @RTUNE_VERSION_MINOR@
#define RTUNE_VERSION          @RTUNE_VERSION@

// cmakedefine01 MACRO will define MACRO as either 0 or 1
// cmakedefine MACRO 1 will define MACRO as 1 or leave undefined

#cmakedefine OMPT_SUPPORT
#cmakedefine OMPT_TRACING_SUPPORT
#cmakedefine OMPT_TRACING_GRAPHML_DUMP
#cmakedefine OMPT_ONLINE_TRACING_PRINT
#cmakedefine RTUNE_MEASUREMENT_SUPPORT
#cmakedefine PAPI_MEASUREMENT_SUPPORT
#cmakedefine CPUFREQ_SUPPORT
#cmakedefine PE_MEASUREMENT_SUPPORT
#cmakedefine PE_OPTIMIZATION_SUPPORT
#cmakedefine PE_OPTIMIZATION_DVFS
#cmakedefine RTUNE_AUTOTUNING
#ifdef RTUNE_AUTOTUNING
#ifndef RTUNE_MEASUREMENT_SUPPORT
#define RTUNE_MEASUREMENT_SUPPORT 1
#endif
#endif

#endif /* RTUNE_CONFIG_H */