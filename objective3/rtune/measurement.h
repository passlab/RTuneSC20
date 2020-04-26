#ifndef __MEASUREMENT_H__
#define __MEASUREMENT_H__

#ifdef PAPI_MEASUREMENT_SUPPORT
#include <papi.h>
#define CORE_DEFAULT_NUM_PAPI_EVENTS 2
#define UNCORE_DEFAULT_NUM_PAPI_EVENTS 1
#endif

/*
 * The measurement is based on PAPI hardware counter and MSR/RAPL
 * core counters are per thread
 * uncore counters are per package
 *
 * package energy is per package
 * pp0 is per package
 * pp1 is per package
 * dram energy is per system.
 *
 * Thus we need to identify which thread to perform hardware counter read:
 * Each thread will read per-core counter (PAPI core counter)
 * One thread will read system-related counter (dram)
 * One thread per package will read uncore and per-package related counters (uncore and package/pp0/pp1)
 */

//used to indicate which measurement to perform
#define MEASURE_TIME        0x0001
#define MEASURE_TIME_MASK   0x1110
#define MEASURE_PAPI        0x0010
#define MEASURE_PAPI_MASK   0x1101
#define MEASURE_ENERGY      0x0100
#define MEASURE_ENERGY_MASK 0x1011
#define MEASURE_ALL (MEASURE_TIME|MEASURE_PAPI|MEASURE_ENERGY)

typedef struct ompt_measurement {
    double time_stamp;
    /* the configuration, e.g. number of threads, core frequency, etc when measuring */
    int requested_team_size;
    int team_size;

#ifdef CPUFREQ_SUPPORT
    struct cpufreq_available_frequencies * frequency; /* the current frequency used */
#endif

#ifdef PAPI_MEASUREMENT_SUPPORT
    /* uncore events is only meaningful for the master thread/cpu of the package, the same as pe measurement */
    int num_PAPI_events;
    int num_core_PAPI_events;
    int PAPI_eventSet;
    int *PAPI_Events;
    /* TODO: space optimization needed since we have one measurement per record */
    long long PAPI_counter[CORE_DEFAULT_NUM_PAPI_EVENTS+UNCORE_DEFAULT_NUM_PAPI_EVENTS];

#endif

#ifdef PE_MEASUREMENT_SUPPORT
    long long pe_package[MAX_PACKAGES]; /* in uj */
    long long pe_dram[MAX_PACKAGES];
    long long pe_pp0[MAX_PACKAGES]; /* PP0 is core energy */
    long long pe_pp1[MAX_PACKAGES]; /* PP1 is uncore energy */
    double edp;
#endif
} ompt_measurement_t;


#ifdef  __cplusplus
extern "C" {
#endif

extern ompt_measurement_t total_consumed;

extern void ompt_measure_global_init( );
extern void ompt_measure_global_fini( );
extern void ompt_measure_thread_init( );
extern void ompt_measure_thread_fini( );
extern void ompt_measure_init(ompt_measurement_t * me);
extern void ompt_measure_reset(ompt_measurement_t *me);
extern void ompt_measure(ompt_measurement_t *me, int which);
extern void ompt_measure_consume(ompt_measurement_t *me, int which);
extern void ompt_measure_diff(ompt_measurement_t *consumed, ompt_measurement_t *begin_me, ompt_measurement_t *end_me, int which);
extern void ompt_measure_accu(ompt_measurement_t *accu, ompt_measurement_t *me, int which);
extern int ompt_measure_compare(ompt_measurement_t *best, ompt_measurement_t *current, int which);
extern void omp_measure_init_max(ompt_measurement_t *me);
//extern void ompt_measure_print(ompt_measurement_t * me);
//extern void ompt_measure_print_header(ompt_measurement_t * me);
extern void ompt_measure_print_header_csv(ompt_measurement_t * me, FILE *csv_file);
extern void ompt_measure_print_csv(ompt_measurement_t * me, FILE* csvfile);
extern void ompt_measure_print(ompt_measurement_t * me);
extern void ompt_measure_print_header(ompt_measurement_t * me);

extern double read_timer();
extern double read_timer_ms();

#ifdef  __cplusplus
};
#endif
#endif
