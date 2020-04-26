#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <limits.h>
#include <float.h>
#include <sys/time.h>
#include "rtune_runtime.h"

double read_timer() {
    struct timeval t;
    double time;
    gettimeofday(&t, NULL);
    time = t.tv_sec + 1.0e-6*t.tv_usec;
    return time;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeval t;
    double time;
    gettimeofday(&t, NULL);
    time = t.tv_sec*1000.0 + 1.0e-3*t.tv_usec;
    return time;
}

ompt_measurement_t total_consumed;

#ifdef PAPI_MEASUREMENT_SUPPORT

//int CORE_PAPI_Events[CORE_DEFAULT_NUM_PAPI_EVENTS]={PAPI_TOT_INS, PAPI_TOT_CYC};
char * CORE_PAPI_EVENT_NAMES[] = {"CPU_CLK_UNHALTED","CYCLE_ACTIVITY:STALLS_TOTAL","RESOURCE_STALLS:SB"};
//char * CORE_PAPI_EVENT_NAMES[] = {"CPU_CLK_UNHALTED","CYCLE_ACTIVITY:STALLS_TOTAL"};
long CORE_PAPI_EVENTS_THRESHOLD[CORE_DEFAULT_NUM_PAPI_EVENTS]={2147483646L, 2147483646L};
//long CORE_PAPI_EVENTS_THRESHOLD[CORE_DEFAULT_NUM_PAPI_EVENTS]={83646L, 83646L};
int UNCORE_PAPI_Events[UNCORE_DEFAULT_NUM_PAPI_EVENTS]={PAPI_L1_DCM};
#endif

void ompt_measure_global_init() {
#ifdef PAPI_MEASUREMENT_SUPPORT
    /*papi event initilization*/
    PAPI_library_init(PAPI_VER_CURRENT);
    //PAPI_thread_init(get_global_thread_num);
    PAPI_thread_init(pthread_self);
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    rapl_sysfs_init();
#endif
}

/**
 * TODO
 */
void ompt_measure_global_fini() {
#ifdef PAPI_MEASUREMENT_SUPPORT
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    rapl_sysfs_fini();
#endif
}

#ifdef PAPI_MEASUREMENT_SUPPORT

static void PAPI_overflow_handler(int EventSet, void *address, long_long overflow_vector, void *context) {
    printf("Overflow at %p! bit=0x%llx \en", address, overflow_vector);
}
#endif

/**
 * Each thread should call this function to init the measurement (PAPI and PE)
 * @param emap
 */
void ompt_measure_thread_init() {
//    memset(emap, 0, sizeof(ompt_measurement_t));
#ifdef PAPI_MEASUREMENT_SUPPORT
    event_map.PAPI_eventSet = PAPI_NULL;

    PAPI_create_eventset(&event_map.PAPI_eventSet);

    event_map.num_core_PAPI_events = 0;
    int i;
    for (i=0; i<CORE_DEFAULT_NUM_PAPI_EVENTS; i++) {
        int eventcode;
        PAPI_event_name_to_code(CORE_PAPI_EVENT_NAMES[i], &eventcode);
        int num = PAPI_add_event(event_map.PAPI_eventSet, eventcode);
        if (num == PAPI_OK) {
            event_map.num_core_PAPI_events++;
            event_map.PAPI_Events[i] = eventcode;
#if 0
            int sw_overflow = PAPI_OVERFLOW_FORCE_SW; /* Software overflow allows for derived events, one setting per eventSet */
            sw_overflow = 0;
            int rtval = PAPI_overflow(event_map.PAPI_eventSet, CORE_PAPI_Events[i], CORE_PAPI_EVENTS_THRESHOLD[i], sw_overflow, PAPI_overflow_handler);
            if (rtval != PAPI_OK) printf("PAPI_overflow failed: %d\n", rtval);
#endif
        }
    }
    //printf("%d core events of total %d are added by thread %d.\n", event_map.num_core_PAPI_events, CORE_DEFAULT_NUM_PAPI_EVENTS, event_map.thread_id );

    event_map.num_PAPI_events = event_map.num_core_PAPI_events;
    if (is_master()) {
        for (i=0; i<UNCORE_DEFAULT_NUM_PAPI_EVENTS; i++) {
            int num = PAPI_add_event(event_map.PAPI_eventSet, UNCORE_PAPI_Events[i]);
            if (num == PAPI_OK) {
                event_map.num_PAPI_events++;
                event_map.PAPI_Events[i+event_map.num_core_PAPI_events] = UNCORE_PAPI_Events[i];
            }
        }
        //printf("%d uncore events of total %d are added by thread %d.\n", event_map.num_PAPI_events - event_map.num_core_PAPI_events, UNCORE_DEFAULT_NUM_PAPI_EVENTS, event_map.thread_id );
    }
    PAPI_start(event_map.PAPI_eventSet);
#endif

#ifdef PE_MEASUREMENT_SUPPORT
    if (is_master()) {
    }
#endif
}

/* init a measurement to have the same eventset as the thread event set */
void ompt_measure_init(ompt_measurement_t * me) {
#ifdef PAPI_MEASUREMENT_SUPPORT
    me->num_PAPI_events = event_map.num_PAPI_events;
    me->num_core_PAPI_events = event_map.num_core_PAPI_events;
    me->PAPI_eventSet = event_map.PAPI_eventSet;
    me->PAPI_Events = &event_map.PAPI_Events[0];
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (is_master()) {
    }
#endif
}

void ompt_measure_thread_fini() {
#ifdef PAPI_MEASUREMENT_SUPPORT
    PAPI_stop(event_map.PAPI_eventSet, NULL);
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (is_master()) {
    }
#endif
}

void ompt_measure_reset(ompt_measurement_t *me) {
    me->time_stamp = 0;
    int i;
#ifdef PAPI_MEASUREMENT_SUPPORT
    PAPI_reset(me->PAPI_eventSet);
    for (i=0; i<me->num_PAPI_events; i++)
        me->PAPI_counter[i] = 0;
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (is_master()) {
        for (i=0; i<MAX_PACKAGES; i++)
        me->pe_package[i] = 0;
        me->pe_dram[i] = 0;
    }
#endif
}
/**
 * @param me
 */
void ompt_measure(ompt_measurement_t *me, int which) {
    if (which & MEASURE_TIME) me->time_stamp = read_timer_ms();
#ifdef PAPI_MEASUREMENT_SUPPORT
    if (which & MEASURE_PAPI) {
        //PAPI_read_counters(me->core_PAPI_counter, CORE_DEFAULT_NUM_PAPI_EVENTS);
        PAPI_read(me->PAPI_eventSet, me->PAPI_counter);
    }
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (which & MEASURE_ENERGY) {
        if (is_master()) {
            rapl_sysfs_read_package_dram_energy(me->pe_package, me->pe_dram);
        }
    }
#endif
}

/**
 * perform current stop/measurement and store the difference between current measurement and the me and store the difference
 * in me
 * @param me
 */
void ompt_measure_consume(ompt_measurement_t *me, int which) {
    if (which & MEASURE_TIME) me->time_stamp = read_timer_ms() - me->time_stamp;
#ifdef PAPI_MEASUREMENT_SUPPORT
    if (which & MEASURE_PAPI) {
        int i;
        long long PAPI_counter[me->num_PAPI_events];
        PAPI_read(me->PAPI_eventSet, PAPI_counter);
        for (i=0; i<me->num_PAPI_events; i++)
            me->PAPI_counter[i] = PAPI_counter[i] - me->PAPI_counter[i];
    }
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (which & MEASURE_ENERGY) {
        if (is_master()) {
            ompt_measurement_t current;
            rapl_sysfs_read_package_dram_energy(current.pe_package, current.pe_dram);

            me->pe_package[0] = energy_consumed(me->pe_package, current.pe_package);
            me->pe_dram[0] = energy_consumed(me->pe_dram, current.pe_package);
            //me->pe_pp0[0] = energy_consumed(me->pe_pp0, current.pe_pp0);
            //me->pe_pp1[0] = energy_consumed(me->pe_pp1, current.pe_pp1);
        }
    }
#endif
}

void ompt_measure_diff(ompt_measurement_t *consumed, ompt_measurement_t *begin_me, ompt_measurement_t *end_me, int which) {
    if (which & MEASURE_TIME) consumed->time_stamp = end_me->time_stamp - begin_me->time_stamp;
#ifdef PAPI_MEASUREMENT_SUPPORT
    if (which & MEASURE_PAPI) {
        int i;
        for (i = 0; i < end_me->num_PAPI_events; i++)
            consumed->PAPI_counter[i] = end_me->PAPI_counter[i] - begin_me->PAPI_counter[i];
    }
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (which & MEASURE_ENERGY) {
        if (is_master()) {
            consumed->pe_package[0] = energy_consumed(begin_me->pe_package, end_me->pe_package);
            consumed->pe_dram[0] = energy_consumed(begin_me->pe_dram, end_me->pe_package);
//        consumed->pe_pp0[0] = energy_consumed(begin_me->pe_pp0, end_me->pe_pp0);
//        consumed->pe_pp1[0] = energy_consumed(begin_me->pe_pp1, end_me->pe_pp1);
        }
    }
#endif
}

/**
 * Compare two measurement according to either performance, energy or EDP.
 * The function returns the percentage of (best - current)/best in integer
 * @param best
 * @param current
 */
int ompt_measure_compare(ompt_measurement_t *best, ompt_measurement_t *current, int which) {
    if (which & MEASURE_TIME) {
        double diff = best->time_stamp - current->time_stamp;
        return (int) (100.0 * (diff / best->time_stamp));
    }
#ifdef PAPI_MEASUREMENT_SUPPORT
    if (which & MEASURE_PAPI) {
        int i = 0;
        double diff = best->PAPI_counter[i] - current->PAPI_counter[i];
        return (int) (100.0 * (diff / best->PAPI_counter[i]));
    }
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (which & MEASURE_ENERGY) {
        if (is_master()) {
            double diff = best->pe_package[0] - current->pe_package[0];
            return (int) (100.0 * (diff / best->pe_package[0]));
        }
    }
#endif
    return 0;
}

void ompt_measure_accu(ompt_measurement_t *accu, ompt_measurement_t *me, int which) {
    if (which & MEASURE_TIME) accu->time_stamp += me->time_stamp;
#ifdef PAPI_MEASUREMENT_SUPPORT
    if (which & MEASURE_PAPI) {
        int i;
        for (i = 0; i < accu->num_PAPI_events; i++)
            accu->PAPI_counter[i] += me->PAPI_counter[i];
    }
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (which & MEASURE_ENERGY) {
        if (is_master()) {
            accu->pe_package[0] += me->pe_package[0];
            accu->pe_dram[0] += me->pe_dram[0];
            // accu->pe_pp0[0] += me->pe_pp0[0];
            // accu->pe_pp1[0] += me->pe_pp1[0];
        }
    }
#endif
}

void omp_measure_init_max(ompt_measurement_t *me) {
    me->time_stamp = DBL_MAX;
#ifdef PAPI_MEASUREMENT_SUPPORT
    int i;
    for (i=0; i<me->num_PAPI_events; i++)
        me->PAPI_counter[i] = LLONG_MAX;
#endif
#ifdef PE_MEASUREMENT_SUPPORT
    if (is_master()) {
        me->pe_package[0] = LLONG_MAX;
        me->pe_dram[0] = LLONG_MAX;
    }
#endif
}

void ompt_measure_print_header_csv(ompt_measurement_t * me, FILE *csv_file) {
    fprintf(csv_file, "Time(ms)");
#ifdef PE_MEASUREMENT_SUPPORT
//    fprintf(csv_file, ",Total Energy (PKG+DRAM)(j),package,PP0,PP1,DRAM");
    fprintf(csv_file, ",Total Energy (PKG+DRAM)(j),package,DRAM");
#endif
#ifdef PAPI_MEASUREMENT_SUPPORT
    int i;
    int number = me->num_PAPI_events;
    int Events[number];
//    PAPI_list_events(me->eventSet, Events, &number);
//    printf("%d PAPI events\n", number);
    char EventName[PAPI_MAX_STR_LEN];
    for (i=0; i<number; i++) {
        //PAPI_event_code_to_name(Events[i], EventName);
        PAPI_event_code_to_name(me->PAPI_Events[i], EventName);
        fprintf(csv_file, ",%s", EventName);
    }

    /* for derived output */
#ifdef PAPI_CPI_PRINT
    fprintf(csv_file, ",PAPI_CPI");
#endif
#endif
}

void ompt_measure_print_csv(ompt_measurement_t * me, FILE* csvfile) {
    fprintf(csvfile, "%.2f", me->time_stamp);
#ifdef PE_MEASUREMENT_SUPPORT
    double package_energy = me->pe_package[0];
//    double pp0_energy = me->pe_pp0[0];
//    double pp1_energy = me->pe_pp1[0];
    double dram_energy = me->pe_dram[0];
    double total_energy = package_energy + dram_energy;
//    fprintf(csvfile, ",%.2f,%.2f,%.2f,%.2f,%.2f", total_energy, package_energy, pp1_energy, pp0_energy, dram_energy);
    fprintf(csvfile, ",%.2f,%.2f,%.2f", total_energy, package_energy, dram_energy);
#endif
#ifdef PAPI_MEASUREMENT_SUPPORT
    int i;
    for (i=0; i<me->num_PAPI_events; i++) {
        fprintf(csvfile, ",%lld", me->PAPI_counter[i]);
    }
#ifdef PAPI_CPI_PRINT
    fprintf(csvfile, ",%.3f", ((double)me->PAPI_counter[1])/((double)me->PAPI_counter[0]));
#endif
#endif
}

/**
 * print the info in certain format
 * @param me
 */
void ompt_measure_print(ompt_measurement_t * me) {
    printf("%.2f", me->time_stamp);
#ifdef PE_MEASUREMENT_SUPPORT
    double package_energy = me->pe_package[0];
    double dram_energy = me->pe_dram[0];
 //   double pp0_energy = me->pe_pp0[0];
 //   double pp1_energy = me->pe_pp1[0];
    double total_energy = package_energy + dram_energy;
  //  printf("\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t\t%.2f", total_energy, package_energy, dram_energy, pp1_energy, pp0_energy);
    printf("\t|\t%.2f\t\t%.2f\t\t%.2f", total_energy/1000000.0, package_energy/1000000.0,
           dram_energy/1000000.0);
#endif
#ifdef PAPI_MEASUREMENT_SUPPORT
    printf("\t|\t");
#ifdef PAPI_CPI_PRINT
    printf("%.3f\t\t", ((double)me->PAPI_counter[1])/((double)me->PAPI_counter[0]));
#endif
    int i;
    for (i=0; i<me->num_PAPI_events; i++) {
        printf("%lld\t\t", me->PAPI_counter[i]);
    }
#endif
    printf("\n");
}

void ompt_measure_print_header(ompt_measurement_t * me) {
    printf("Time(ms)");
#ifdef PE_MEASUREMENT_SUPPORT
//    printf("\tEnergy (j) total (PKG+DRAM): package\tPP0\t\t\tPP1\t\t\tDRAM");
    printf("\t|\tEnergy (j) total (PKG+DRAM): package\t\t\tDRAM");
#endif
#ifdef PAPI_MEASUREMENT_SUPPORT
    printf("\t|\t");
#ifdef PAPI_CPI_PRINT
    printf("PAPI_CPI\t");
#endif
    int i;
    int number = me->num_PAPI_events;
    int Events[number];
//    PAPI_list_events(me->eventSet, Events, &number);
//    printf("%d PAPI events\n", number);
    char EventName[PAPI_MAX_STR_LEN];
    for (i=0; i<number; i++) {
        //PAPI_event_code_to_name(Events[i], EventName);
        PAPI_event_code_to_name(me->PAPI_Events[i], EventName);
        printf("%s\t", EventName);
    }
#endif
    printf("\n");
}
