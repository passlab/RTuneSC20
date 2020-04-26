#include <stdint.h>
#include "rtune_config.h"
#include "measurement.h"
#include "rtune.h"
#ifdef CPUFREQ_SUPPORT
#include <cpufreq.h>
#endif

#define MAX_NUM_RECORDS 10000000
#define MAX_NUM_THREADS 128
#define MAX_NEST_DEPTH 16
#define MAX_HIST_PARALLEL 16
/* the max number of parallel regions in the original source code */
#define MAX_SRC_PARALLELS 64

typedef enum CPUFREQ_GOVERNOR {
    CPUFREQ_GOVERNOR_userspace,
    GPUFREQ_GOVERNOR_conservative,
    GPUFREQ_GOVERNOR_ondemand,
    GPUFREQ_GOVERNOR_powersave,
    GPUFREQ_GOVERNOR_performance,
} CPUFREQ_GOVERNOR_t;

/*
 * This struct is used for providing configuration for tuning a lexgion (user specified using RTune_begin and RTune_end,
 * a worksharing region or a parallel region. So each lexgion has an object of it and there is an global object of this
 * struct as well to provide the default configuration.
 *
 * Each field of this struct MUST be "int" type, which is designed to simplify parsing a configuration file to read
 * the configuration values. For each lexgion, the configuration can be viewed as key-value pairs and the configuration
 * file will be written in the format similar to, but simpler than INI or TOML format. The order of those fields in the
 * rtune_config_t MUST be in the same order as their keys in the config_key array, again, it is designed to simplify
 * parsing the config file. Thus to add or remove a configuration, rtune_config_t, NUM_CONFIG_KEYS and config_keys must
 * be modified and make sure they are consistent so we do not need to change the code for reading the config file.
 *
 * Check read_config function for details.
 */
typedef struct rtune_config {/* all the config fields MUST be from the beginning and
                              * in the same order as the keys in the config_keys array */
    int rtune_enabled; /* enable autotuning of both num_threads and frequency */
    int rtune_log; /* a flag to turn on/off rtune logging to file */
    int rtune_target; /* the target of tuning: performance (only), energy or EDP */
    int rtune_num_runs_per_tune; /* how many runs we need for the lexgion for measurement and tunning, default 1 */
    int rtune_start_run; /* the first record to start rtune, default 1 */

    /* config for number of threads */
    int rtune_enabled_num_threads; /* enable auto tuning of num_threads */
    int max_num_threads;
    int min_num_threads;
    int num_threads;
    int rtune_initial_num_threads;
    int threshold_best2fixed_num_threads; /* a threshold to fix the configuration of num_threads when the best
                                           * configuration is consecutively used that number of of times. */
    /* config for frequency */
    int rtune_enabled_frequency; /* enable autotuning of frequency */
    int cpufreq_governor; /* the governor, int index that points to the cpufreq_governor array */
    int rtune_perthread_frequency; /* a flag to indicate whether we tune each thread or all the threads */
    unsigned int threshold_memboundness; /* if the memory boundness is above this threshold, reduce to the next frequency state */
    int threshold_best2fixed_frequency;

    /* config for EDP */
    int threashold_edp; /* edp threshold */

    const void * codeptr; /* this MUST be after the config fields of this struct */
#ifdef CPUFREQ_SUPPORT
    /* a link list for all the available frequencies of the CPU starting from the max to the min */
    struct cpufreq_available_frequencies * cpufreq_available_freqs;
    struct cpufreq_available_frequencies * rtune_max_frequency;
    struct cpufreq_available_frequencies * rtune_min_frequency;
    struct cpufreq_available_frequencies * fixed_frequency;
    struct cpufreq_available_frequencies * rtune_initial_frequency; /* we may need long type, so far, 4Ghz is the max we can do with unsigned int type */
    struct cpufreq_available_frequencies * rtune_wait_frequency; /* this is the freq if we want to put a CPU to wait during a sync */
#endif
} rtune_config_t;

#define CONVERGING_DIRECTION_INCREMENT 0
#define CONVERGING_DIRECTION_DECREMENT 0

typedef enum user_parameter_datatype {
	INT_TYPE,
	SHORT_TYPE,
	FLOAT_TYPE,
	DOUBLE_TYPE,    
};

typedef struct rtune_user_parameter {
    void * ptr; /* the address of the user parameters */
    int size; /* the size in bytes of each parameter value */
    void * values; /* the values of the parameters along the simulation */
    long count; /* the number of values of the parameters */
    void * threashold; /* the threshold VALUE. The use of this value should be based on the parameter type */
    char converging_direction; /* INCREMENT or DECREMENT */
    char data_type;
};

extern rtune_config_t global_config; /* the default config for a lexgion if user does not provide config */
extern int num_user_configured_lexgions; /* number of lexgions that user provide configuration for in the config file */
extern rtune_config_t lexgion_rtune_config[]; /* configuration provided by users for the lexgions */

#define NUM_CONFIG_KEYS 16
// Must be in the same order as the rtune_config_init struct
static const char *config_keys[] = {
        "rtune_enabled",  //true or false: to enable rtune, which will enable rtune_enabled_num_threads
        "rtune_log",      //true or false: to enable logging of tuning process
        "rtune_target",   //useless
        "rtune_num_runs_per_tune", //int number: number of runs to collect and record
        "rtune_start_run",         //int number, the first runs to start tuning, default 1, so we can skip some runs at the beginning

        "rtune_enabled_num_threads", //true or false: useless or redundant now with rtune_enabled
        "max_num_threads",   //max number of threads
        "min_num_threads",
        "num_threads",
        "rtune_initial_num_threads", //initial thread to start with
        "threshold_best2fixed_num_threads", //useless

        //the following is useless right now, since cpufreq tuning is too tricky
        "rtune_enabled_frequency",
        "cpufreq_governor", /* because of cpufreq_governor is char * type, twice the size of an int */
        "rtune_perthread_frequency",
        "threshold_memboundness",
        "threshold_best2fixed_frequency",

        "threshold_edp",
};

/* The trace record struc contains every posibble information we want to store per event
 * though not all the fields are used for any events
 * For one million records, we will need about 72Mbytes of memory to store
 * the tracing for each thread.
 *
 * For each event (short or long), there are normally two record, begin record and end record, linked together through
 * the @match_record field. The @endpoint field tells whether it is a begin or end record.
 *
 * For measurement support enabled, the measured data (exe time, PAPI, power) info are stored in the end record.
 */
typedef struct ompt_trace_record ompt_trace_record_t;
#define CACHE_LINE_SIZE 64
/**
 * This struct is used for the master record to store the related records of all workers, e.g. parallel_begin record
 * to store implicit task records of all threads.
 */
struct ompt_worker_records {
    ompt_trace_record_t * implicit_tasks;
    ompt_trace_record_t * implicit_barrier_sync;
    ompt_trace_record_t * implicit_barrier_wait;
/* padding to eliminate false sharing in an array that will be accessed by multiple thread (one element per thread) */
    char padding[CACHE_LINE_SIZE];
};
struct ompt_trace_record {
    int thread_id;
    uint64_t thread_id_inteam;
    int requested_team_size; /* user setting of team size */
    int kind;  /* secondary event kind */
    int endpoint; /* begin or end */
    struct ompt_lexgion *lgp; /* the lexgion of this record */
    int type; /* the type of a lexgion: parallel, master, singer, barrier, task, section, etc. we use trace record event id for this type */
    struct ompt_trace_record *parent; /* the record for the lexgion that enclose the lexgion for this record */
    struct ompt_trace_record *parallel_record; /* The record for the innermost parellel lexgion that encloses this record */

    struct ompt_trace_record *next; /* the link of the link list for all the begin/start records of the same lexical region */

    int record_id;
    struct ompt_trace_record * match_record; /* index for the matching record. the match for begin_event is end and the match for end_event is begin */

    /* for the purpose of saving memory for tracing record, we should enable this macro only if we really do TRACING and
     * Measurement the same time, though it does not hurt even if we donot do */
#if defined(OMPT_TRACING_SUPPORT) && defined(RTUNE_MEASUREMENT_SUPPORT)
    ompt_measurement_t measurement;
#endif
    struct ompt_worker_records * worker_records; /* for master to store the records for all the workers, e.g. the implicit tasks records */
} ;

typedef enum lexgion_endpoint_t {
    lexgion_scope_begin                    = 1,
    lexgion_scope_end                      = 2
} lexgion_endpoint_t;

/**
 * A lexgion (lexical region) represent a region in the source code
 * storing the lexical parallel regions encountered in the runtime.
 * A lexgion should be identified by the codeptr_ra and the type field together. codeptr_ra is the binary address of the
 * lexgion and type is the type of region. The reasons we need type for identify a lexgion are:
 * 1). OpenMP combined and composite construct, e.g. parallel for
 * 2). implicit barrier, e.g. parallel.
 * Becasue of that, the events for those constructs may use the same codeptr_ra for the callback, thus we need further
 * check the type so we know whether we need to create two different lexgion objects
 * */
typedef struct ompt_lexgion {
    /* we use the binary address of the lexgion as key for each lexgion, assuming that only one
     * call path of the containing function. This is the limitation since a function may be called from different site,
     * but this seems ok so far
     */
    const void *codeptr_ra;
    const void *end_codeptr;
    /* For single construct, there are two end_codeptr-s for the region, one for the code containing
     * the single region, which is for the thread who win to execute the single; the other for the code that does not
     * contain the single region, which is for the threads who do not execute the single. Thus we need the
     * end_codeptr2 variable. It is important to note that only one of two codeptrs (end_codeptr and end_codeptr2) may
     * set. Both are set only if a thread encounters the same single region at least twice, and it executes once and skips
     * the other times.
     *
     * Other than single, so far we are not sure whether other directives has this behavior, e.g. sections
     */
    const void *end_codeptr2;
    ompt_trace_record_t * most_recent;
    int total_record; /* total number of records, i.e. totoal number of execution of the same parallel region */
    rtune_config_t * rtune_config;

    int num_threads_tuning_done; /* flag to indicate whether num_threads tuning is done or not */
    int frequency_tuning_done; /* flag to indicate whether frequency tuning is done or not */
    ompt_measurement_t total;
    ompt_measurement_t best;
    struct exe_time_thread {
        double exe_time;
        double derivative; //derivative % against best performance
        int count;
    } exe_time_threads[MAX_NUM_THREADS]; //max 128 threads supported
    int rtune_best_counter_num_threads; /* how many times the best configuration for num_threads so far has been consecutively seen */
    int rtune_best_counter_frequency; /* how many times the best configuration for num_threads so far has been consecutively seen */
    ompt_measurement_t rtune_accu; /* accumulated measurement for the past number of runs used for tuning*/
    ompt_measurement_t current;
    FILE * rtune_logfile;
} ompt_lexgion_t;

/* each thread has an object of thread_event_map that stores all the tracing record along 
 * during the execution.
 *
 * We assume NO thread migration happening in OS
 */
typedef struct thread_event_map {
    int cpu_id; /* the linux cpu id, i.e. /sys/devices/system/cpu/ */
    int package_id; /* the hardware package/socket/processor this thread belongs to, i.e. /sys/devices/system/cpu/cpuX/topology/physical_package_id */
    uint64_t *thread_data;
    int counter;
    /* the stack for storing the record indices of the lexgion events.
     * Considering nested region, this has to be stack
     */
    ompt_lexgion_t *lexgion_stack[MAX_NEST_DEPTH]; /* depth of lexgion_begin call, which could be of the same region */
    int lexgion_stack_top;
    ompt_trace_record_t * record_stack; /* the top of the record stack through out execution, linked through parent field */

    ompt_lexgion_t lexgions[MAX_SRC_PARALLELS];
    int lexgion_end; /* the last lexgion in the lexgions array */
    int lexgion_recent; /* the most-recently used lexgion in the lexgions array */

#ifdef PAPI_MEASUREMENT_SUPPORT
    /* the events that will be used by all the measurement of this thread */
    int num_PAPI_events;
    int num_core_PAPI_events;
    int PAPI_eventSet;
    int PAPI_Events[CORE_DEFAULT_NUM_PAPI_EVENTS+UNCORE_DEFAULT_NUM_PAPI_EVENTS];
#endif

    /* so far this is useless */
    ompt_measurement_t measurement;

    ompt_trace_record_t *records;
} thread_event_map_t;

#ifdef  __cplusplus
extern "C" {
#endif

/* this is the array for store all the event tracing records by all the threads */
extern __thread thread_event_map_t event_map;
extern __thread int global_thread_num;
extern __thread ompt_trace_record_t * current_parallel_record;
extern __thread ompt_lexgion_t * current_parallel_lexgion;

extern volatile int num_threads;

/* handy macro for get pointers to the event_map of a thread, or pointer to a trace record */
#define get_last_lexgion_record() (event_map.lexgion_stack[event_map.innermost_lexgion]->most_recent)

#define top_record() (event_map.record_stack)
#define top_lexgion() (event_map.lexgion_stack[event_map.lexgion_stack_top])
#define is_master() (global_thread_num == 0)

/* functions for init/fini event map */
extern void init_thread();
extern void fini_thread();

/** mark in the map that the execution enters into a region (parallel region, master, single, etc)
 * can only be called when the lexgion event is added to the record
 */
extern ompt_trace_record_t * top_record_type(int type);
extern ompt_lexgion_t *ompt_lexgion_begin(const void *codeptr_ra);
extern ompt_lexgion_t *ompt_lexgion_end(const void * codeptr_ra);
extern ompt_trace_record_t *add_trace_record_begin(int type, ompt_lexgion_t *lgp);
extern ompt_trace_record_t *add_trace_record_end(ompt_lexgion_t *lgp);
extern void print_lexgions(thread_event_map_t *emap);

/*
 * API for configure rtune based on config file
 */
extern void rtune_config_init();
extern void rtune_config_fini();
extern void print_config();
/**
 * runtime instrumentation API
 */
/* for autotuning performance and energy */
extern int rtune_master_begin_lexgion(ompt_lexgion_t *lgp);
extern void rtune_master_end_lexgion(ompt_lexgion_t *lgp, ompt_trace_record_t *record);

extern void print_all_lexgions_csv(thread_event_map_t * emap);
extern void print_all_lexgions(thread_event_map_t * emap);
extern void ompt_event_maps_to_graphml(thread_event_map_t* maps);

#ifdef  __cplusplus
};
#endif
