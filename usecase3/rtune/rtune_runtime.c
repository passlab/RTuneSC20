#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>
#include "rtune_runtime.h"
#include "rtune.h"

__thread thread_event_map_t event_map;
__thread int global_thread_num;
__thread ompt_trace_record_t * current_parallel_record;
__thread ompt_lexgion_t * current_parallel_lexgion;
volatile int num_threads = 0; /* this counter is not guaranteed to provide the
 * actual exact number of threads until after it become useless :-). So make it useful after being useless */

rtune_config_t global_config; /* the object that has the default config for rtune, each lexgion has its own config though*/
rtune_config_t lexgion_rtune_config[MAX_SRC_PARALLELS]; /* a global array that store the rtune configuration for each lexgion */
int num_user_configured_lexgions = 0;

extern int sched_getcpu(void); /* get rid of compiler warning */

extern char *ompt_event_names[]; /* initialized when ompt_callback is registered */

static void push_lexgion(ompt_lexgion_t * lgp) {
    event_map.lexgion_stack[++event_map.lexgion_stack_top] = lgp;
}

static ompt_lexgion_t * pop_lexgion() {
    ompt_lexgion_t * top = event_map.lexgion_stack[event_map.lexgion_stack_top--];
    return top;
}

static void push_record(ompt_trace_record_t * record) {
    record->parent = event_map.record_stack;
    event_map.record_stack = record;
}

static ompt_trace_record_t * pop_record() {
    ompt_trace_record_t * top = event_map.record_stack;
    event_map.record_stack = top->parent;
    return top;
}

ompt_trace_record_t * top_record_type(int type) {
    ompt_trace_record_t * top = event_map.record_stack;
    while (top != NULL && top->type != type) {
        top = top->parent;
    }
    return top;
}

/**
 * add a begin event trace record
 * @param emap
 * @param event_id
 * @param frame
 * @param lgp: the lexgion pointer
 * @param parallel_record: the trace record that launches the lexgion this record belongs to. If NULL, this record is the
 *                  launching record
 * @return
 */
ompt_trace_record_t *add_trace_record_begin(int type, ompt_lexgion_t *lgp) {
    int counter = event_map.counter;
    ompt_trace_record_t *rd = &event_map.records[counter];
    rd->record_id = counter;
    //if (global_thread_num == 0) printf("add begin trace record type %s, id %d, lgp: %p\n", ompt_event_names[type], counter, lgp->codeptr_ra);
    rd->lgp = lgp;
    rd->type = type;
    rd->parallel_record = current_parallel_record;
    rd->parent = top_record();
    rd->thread_id = global_thread_num;
    rd->match_record = NULL;
    rd->endpoint = lexgion_scope_begin;

    /* add record to the lexgion link-list */
    if (lgp->most_recent == NULL) { /* the first record */
        lgp->most_recent = rd;
        rd->next = NULL;
    } else {
        rd->next = lgp->most_recent;
        lgp->most_recent = rd;
    }

    /* push to the record stack */
    push_record(rd);
    event_map.counter++;

    return rd;
}

ompt_trace_record_t *add_trace_record_end(ompt_lexgion_t *lgp) {
    int counter = event_map.counter;
    ompt_trace_record_t *rd = &event_map.records[counter];
    rd->record_id = counter;
    rd->endpoint = lexgion_scope_end;

    ompt_trace_record_t * begin_record = pop_record();
    rd->lgp = lgp;

    /* link the begin and end record so we can easily find each other */
    begin_record->match_record = rd;
    rd->match_record = begin_record;

    //if (global_thread_num == 0) printf("add end trace record for type %s, id %d, lgp: %p\n", ompt_event_names[begin_record->type], counter, lgp->codeptr_ra);

    event_map.counter++;
    return rd;
}

/**
 * this can be called by multiple threads since it is a read-only search.
 * @param emap
 * @param codeptr_ra
 * @param index
 * @return
 */
static ompt_lexgion_t *ompt_find_lexgion(const void *codeptr_ra, int * index) {
    if (event_map.lexgion_recent < 0 || event_map.lexgion_end < 0) return NULL; /* play it safe for dealing with data race */
    int i;
    ompt_lexgion_t * lgp;
    /* search forward from the most recent one */
    for (i=event_map.lexgion_recent; i<=event_map.lexgion_end; i++) {
        lgp = &event_map.lexgions[i];
        if (lgp->codeptr_ra == codeptr_ra) {
            *index = i;
            return lgp;
        }
    }
    /* search from 0 to most recent one */
    for (i=0; i<event_map.lexgion_recent; i++) {
        lgp = &event_map.lexgions[i];
        if (lgp->codeptr_ra == codeptr_ra) {
            *index = i;
            return lgp;
        }
    }
    return NULL;
}

/**
 * Entering a lexgion and meaning a new record for this lexgion
 */
ompt_lexgion_t *ompt_lexgion_begin(const void *codeptr_ra) {
    int i;

    ompt_lexgion_t *lgp = ompt_find_lexgion(codeptr_ra, &i);
    if (lgp != NULL) {
        lgp->total_record++;
        //lgp->type = type; /* assertion check */
    } else {
        i = event_map.lexgion_end + 1;
        if (i == MAX_SRC_PARALLELS) {
            fprintf(stderr, "Max number of lex regions (%d) allowed in the source code reached\n",
                    MAX_SRC_PARALLELS);
        }
        lgp = &event_map.lexgions[i];
        lgp->codeptr_ra = codeptr_ra;
        //printf("%d: lexgion_begin(%d, %X): first time encountered %X\n", emap->thread_id, i, codeptr_ra, lgp);
        lgp->most_recent = NULL;
        lgp->end_codeptr = NULL;
        lgp->end_codeptr2 = NULL;
        lgp->total_record = 1;

        /* find the rtune_config_init object if user provide the config info for this lexgion, otherwise, use global_config */
        int j;
        for (j=0; j<num_user_configured_lexgions; j++) {
            if (lexgion_rtune_config[j].codeptr == codeptr_ra) {
	    	//printf("finding my config: %d, %p, %p, %p\n", j, &lexgion_rtune_config[j], lexgion_rtune_config[j].codeptr, codeptr_ra);
                lgp->rtune_config = &lexgion_rtune_config[j];
                break;
            }
        }
        if (j == num_user_configured_lexgions) { /* use global */
            lgp->rtune_config = &global_config;
        }

        for (j=0; j<MAX_NUM_THREADS; j++)
            lgp->exe_time_threads[j].count = 0;

        event_map.lexgion_end = i;
        if (lgp->rtune_config->rtune_enabled && lgp->rtune_config->rtune_log) {
            char filename[48];
            sprintf(filename, "rtune_log_%p_thread_%d.txt", codeptr_ra, global_thread_num);
            lgp->rtune_logfile = fopen(filename, "w");
        }
    }
    event_map.lexgion_recent = i; /* cache it for future search */
    push_lexgion(lgp);
    //printf("push lexgion: %x codeptr: %x\n", lgp, codeptr_ra);
    return lgp;
}

ompt_lexgion_t * ompt_lexgion_end(const void * codeptr_ra) {
    ompt_lexgion_t * lgp = pop_lexgion();
    if (lgp->end_codeptr == NULL) {
        lgp->end_codeptr = codeptr_ra;
    } else if (lgp->end_codeptr != codeptr_ra) {
        /* this is normal for single construct, i.e. the endptr_ra-s for the thread executing the single and those that do not
         * execute the single are different.
         */
        //fprintf(stderr, "Thread: %d, Second end_codeptr captured for the region: %p-->%p | %p (single or section)\n",
        //        global_thread_num, lgp->codeptr_ra, lgp->end_codeptr, codeptr_ra);
        if (lgp->end_codeptr2 == NULL) {
            lgp->end_codeptr2 = codeptr_ra;
        } else if (lgp->end_codeptr2 != codeptr_ra) {
        //    fprintf(stderr, "Thread: %d, Third end_codeptr captured for the region: %p-->%p | %p | %p (section region)\n",
        //            global_thread_num, lgp->codeptr_ra, lgp->end_codeptr, lgp->end_codeptr2, codeptr_ra);
        } else {}
    } else {}
    //printf("ppop lexgion: %x codeptr: %x\n", lgp, codeptr_ra);
    return lgp;
}

volatile static global_initialized = 0;
volatile static global_finalized = 1;
void rtune_global_init() {
    if (global_initialized && !global_finalized) return;
    int i;

    //for (i=0; i<MAX_PACKAGES; i++) package_master_thread[i] = -1;
    ompt_measure_global_init( );
#ifdef RTUNE_AUTOTUNING
    rtune_config_init(); /* read config from file or use system default */
    //print_config();
    omp_set_num_threads(global_config.max_num_threads);
#endif

#ifdef PE_OPTIMIZATION_SUPPORT
    /* check the system to find total number of hardware cores, total number of hw threads (kernel processors),
     * SMT way and mapping of hwthread id with core id
     */
    int coreid = sched_getcpu() % total_cores;
    int hwth2id = coreid + total_cores; /* NOTES: this only works for 2-way hyperthreading/SMT */
    for(i = 0; i < total_cores; i++)
    {
        if (coreid == i) { /* at the beginning, only this core is in HIGH_FREQ and all others are set in LOW_FREQ */
            HWTHREADS_FREQ[i] = CORE_HIGH_FREQ;
            cpufreq_set_frequency(i, CORE_HIGH_FREQ);
        } else {
            HWTHREADS_FREQ[i] = CORE_LOW_FREQ;
            cpufreq_set_frequency(i, CORE_LOW_FREQ);
        }
    }

    for (; i<TOTAL_NUM_HWTHREADS; i++) {
        if (hwth2id == i) {
            HWTHREADS_FREQ[i] = CORE_HIGH_FREQ;
        } else {
            HWTHREADS_FREQ[i] = CORE_LOW_FREQ;
        }
    }
#endif

    //New need this since now event_maps are now in global static area */
    //memset(event_maps, 0, sizeof(thread_event_map_t)*MAX_NUM_THREADS);

#ifdef GLOBAL_MEASUREMENT
    ompt_measure_init(&total_consumed);
    ompt_measure(&total_consumed);
#endif
    global_initialized = 1;
    global_finalized = 0;
}

void rtune_global_fini() {
    if (!global_initialized && global_finalized) return;
    global_finalized = 1;
    // on_ompt_event_runtime_shutdown();
#ifdef GLOBAL_MEASUREMENT
    ompt_measure_consume(&total_consumed);
#endif
    ompt_measure_global_fini( );
#ifdef GLOBAL_MEASUREMENT
    printf("==============================================================================================\n");
    printf("Total OpenMP Execution: | ");
    ompt_measure_print_header(&total_consumed);
    printf("                        | ");
    ompt_measure_print(&total_consumed);
    printf("==============================================================================================\n");
#endif
    /*
    void* callstack[128];
    int i, frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack, frames);
    for (i = 0; i < frames; ++i) {
        printf("%s\n", strs[i]);
    }
     */

//    print_all_lexgions(emap);
//    print_all_lexgions_csv(emap);

    rtune_config_fini( );
    global_initialized = 0;
}

void rtune_master_begin() {
    const void * begin_codeptr = __builtin_return_address(0);
    ompt_lexgion_t * lgp = ompt_lexgion_begin(begin_codeptr);
#ifdef OMPT_TRACING_SUPPORT
    ompt_trace_record_t *record = add_trace_record_begin(ompt_callback_control_tool, lgp);
#endif
    rtune_master_begin_lexgion(lgp);
    omp_set_num_threads(lgp->current.team_size);
}

void rtune_master_end() {
    const void * end_codeptr = __builtin_return_address(0);
    ompt_lexgion_t * lgp = ompt_lexgion_end(end_codeptr);
    ompt_trace_record_t *record = NULL;
    //printf("RTune_end lexgion: %p\n", lgp);
#ifdef OMPT_TRACING_SUPPORT
    record = add_trace_record_end(lgp);
#endif
    rtune_master_end_lexgion(lgp, record);
}

/**
 * This function should be called by a master thread in the sequential region and upon entering into parallel execution.
 * @param emap The thread event map
 * @param lgp the lexgion identified for the tuning and a lexgion has a begin and end lines that enclose it.
 * @param requested_team_size the requested number of threads
 * @param team_size the tuned number of threads
 * @param frequency the tuned CPU frequency (package frequency)
 *
 * The tuned configuration (team_size and frequency) are stored in the lgp->current measurement object.
 *
 * @return whether rtuning is performed (1) or not (0)
 */
int rtune_master_begin_lexgion(ompt_lexgion_t *lgp) {
#if 1
#ifdef RTUNE_MEASUREMENT_SUPPORT
    if (lgp->total_record < lgp->rtune_config->rtune_start_run) { /* the first record */
        //consider this is warm up
    } else if (lgp->total_record == lgp->rtune_config->rtune_start_run) { /* the first record */
        ompt_measure_init(&lgp->current);
        ompt_measure_init(&lgp->total);
        if (lgp->rtune_config->rtune_enabled_num_threads) {
            /* use the initial values provided by the config */
            lgp->current.team_size = lgp->rtune_config->rtune_initial_num_threads;
            ompt_measure_init(&lgp->rtune_accu);
            ompt_measure_reset(&lgp->rtune_accu);
            omp_measure_init_max(&lgp->best);
        }
        lgp->num_threads_tuning_done = !lgp->rtune_config->rtune_enabled_num_threads;
    } else {
        //printf("rtune enabled: %p, %d\n", lgp->rtune_config, lgp->rtune_config->rtune_enabled);
        /* greedy strategy for tuning num_threads */
        if (!lgp->num_threads_tuning_done) {
            if ((lgp->total_record - lgp->rtune_config->rtune_start_run) % lgp->rtune_config->rtune_num_runs_per_tune == 0) {
                int current_team_size = lgp->current.team_size;
                lgp->exe_time_threads[current_team_size].count++;
                lgp->exe_time_threads[current_team_size].exe_time = lgp->rtune_accu.time_stamp; //store exe time
                if (lgp->rtune_config->rtune_log) {
                    fprintf(lgp->rtune_logfile,
                            "------------------------------------------------------------------------------\n");
                    fprintf(lgp->rtune_logfile,
                            "Collecting for number %d-%d run: %d num_threads to deliver %.2fms perf\n",
                            lgp->total_record - lgp->rtune_config->rtune_num_runs_per_tune,
                            lgp->total_record, current_team_size, lgp->rtune_accu.time_stamp);
                }
                int min_num_threads = lgp->rtune_config->min_num_threads;
                int team_size;
                if (current_team_size > min_num_threads) {
                    team_size = current_team_size - 2; // new team size
                    if (team_size < min_num_threads) {
                        team_size = min_num_threads;
                    }
                } else { //performance trail is done, now we need to select the team_size that delivers the best perf
                    lgp->num_threads_tuning_done = 1;
                    //search the exe_time collected for each thread configuration
                    int i;
                    double best_perf = lgp->exe_time_threads[min_num_threads].exe_time;
                    team_size = min_num_threads;
                    if (lgp->exe_time_threads[min_num_threads+1].count) {
                        i = min_num_threads+1;
                    } else i = min_num_threads+2;
                    for (; i<=lgp->rtune_config->max_num_threads; i+=2) {
                        if (lgp->exe_time_threads[i].exe_time < best_perf) {
                            best_perf = lgp->exe_time_threads[i].exe_time;
                            team_size = i;
                        }
                    }
                    if (lgp->rtune_config->rtune_log) {
                        fprintf(lgp->rtune_logfile,
                            "================================================================================\n");
                        fprintf(lgp->rtune_logfile,
                            "Setting %d num_threads that delivers the best  %.2fms perf\n",
                                team_size, best_perf);
                    }
                    printf("Setting %d num_threads that delivers the best  %.2fms perf\n", team_size, best_perf);

                    /* use first derivative to find the sweetspot between performance and number of threads */
                    if (team_size > min_num_threads) {
                        double derivative_threshold = 0.02;
                        lgp->exe_time_threads[min_num_threads].derivative = (lgp->exe_time_threads[min_num_threads].exe_time - best_perf)/best_perf/(team_size-min_num_threads);
                        int team_size_derivative = min_num_threads;
                        double best_derivative = lgp->exe_time_threads[min_num_threads].derivative;

                        if (lgp->exe_time_threads[min_num_threads+1].count) {
                            i = min_num_threads+1;
                        } else i = min_num_threads+2;

                        /** this algorithm is considered to have optimal tradeoff between performance and # cores */
                        while (i < team_size) {
                            double i_derivative = (lgp->exe_time_threads[i].exe_time - best_perf)/best_perf/(team_size-i);
                            lgp->exe_time_threads[i].derivative = i_derivative;
                            if (lgp->rtune_config->rtune_log) {
                                fprintf(lgp->rtune_logfile,
                                        "------------------------------------------------------------------------------\n");
                                fprintf(lgp->rtune_logfile,
                                        "Calculating derivative of performance for %d num_threads that deliver %.2fms perf: %.2f\n",
                                        i, lgp->exe_time_threads[i].exe_time, i_derivative);
                                fprintf(lgp->rtune_logfile,
                                        "compared with %threads for the best %.2fms perf, derivative (per-thread differential) is %.2\n",
                                        team_size, best_perf, i_derivative);
                            }
#ifdef PERF_ENERGY_TRADEOFF
                            if (i_derivative < best_derivative) {
                                best_derivative = i_derivative;
                                team_size_derivative = i;
                                if (lgp->rtune_config->rtune_log) {
                                    fprintf(lgp->rtune_logfile, "================================================================================\n");
                                    fprintf(lgp->rtune_logfile,
                                        "Setting %d num_threads that delivers the best overal perf/energy tradeoff %.2fms perf.\n",
                                        i, lgp->exe_time_threads[i].exe_time);
                                }
                            }
#else
                            if (i_derivative < derivative_threshold) {
                                best_derivative = i_derivative;
                                team_size_derivative = i;
                                if (lgp->rtune_config->rtune_log) {
                                    fprintf(lgp->rtune_logfile, "================================================================================\n");
                                    fprintf(lgp->rtune_logfile,
                                        "Setting %d num_threads (%.2fms perf) that is within the derivative (per-thread differential) threshold (%2.f)\n",
                                        i, lgp->exe_time_threads[i].exe_time, derivative_threshold);
                                }
                                break;
                            }
#endif
                            i += 2;
                        }
                        team_size = team_size_derivative;
                    }
                }
                lgp->current.team_size = team_size;
                ompt_measure_reset(&lgp->rtune_accu); //reset measurement
            }
        }
    }
    /* if rtune disabled, measure; if rtune_enabled and rtune not complets, measure */
    if (!lgp->num_threads_tuning_done) {
        //ompt_measure_reset(&lgp->current);
        ompt_measure(&lgp->current, MEASURE_ALL);
//        printf("counter: %d, measure at begin: %f\n", lgp->total_record, lgp->current.time_stamp);
    }
#endif
#else
#ifdef RTUNE_MEASUREMENT_SUPPORT
    if (lgp->total_record == 1) { /* the first record */
        ompt_measure_init(&lgp->current);
        ompt_measure_init(&lgp->total);
        if (lgp->rtune_config->rtune_enabled_num_threads || lgp->rtune_config->rtune_enabled_frequency) {
            /* use the initial values provided by the config */
            lgp->current.team_size = lgp->rtune_config->rtune_initial_num_threads;
#ifdef CPUFREQ_SUPPORT
            lgp->current.frequency = lgp->rtune_config->rtune_initial_frequency;
#endif
            ompt_measure_init(&lgp->rtune_accu);
            ompt_measure_reset(&lgp->rtune_accu);
            omp_measure_init_max(&lgp->best);
        }
        lgp->num_threads_tuning_done = !lgp->rtune_config->rtune_enabled_num_threads;
        lgp->frequency_tuning_done = !lgp->rtune_config->rtune_enabled_frequency;
    } else {
        //printf("rtune enabled: %p, %d\n", lgp->rtune_config, lgp->rtune_config->rtune_enabled);
        /* greedy strategy for tuning num_threads */
        if (!lgp->num_threads_tuning_done && lgp->current.team_size >= lgp->rtune_config->min_num_threads) {
        //	printf("team size: %d\n", lgp->current.team_size);
            /* our assumption is that we are executing the lexgion of the same problem size */
            if (lgp->rtune_best_counter_num_threads < lgp->rtune_config->threshold_best2fixed_num_threads) { /* not yet stablize, keep tuning */
                if ((lgp->total_record - 1) % lgp->rtune_config->rtune_num_runs_per_tune == 0) {
                    int diff = ompt_measure_compare(&lgp->best, &lgp->rtune_accu, MEASURE_TIME);
                    if (lgp->rtune_config->rtune_log) {
                        fprintf(lgp->rtune_logfile,
                                "---------------------------------------------------------------------------------------------------------------------\n");
                        fprintf(lgp->rtune_logfile,
                                "Tuning from run %d: last config of %d num_threads delivered %d%% perf improvement (now at %.2fms) over the perf with %d threads\n",
                                lgp->total_record, lgp->current.team_size, diff, lgp->rtune_accu.time_stamp,
                                lgp->best.team_size);
                    }
                    if (diff >= 0) { /* better performance for 0%, we should tune */
                        memcpy(&lgp->best, &lgp->rtune_accu, sizeof(ompt_measurement_t));
                        lgp->rtune_best_counter_num_threads = 1;
                        lgp->best.team_size = lgp->current.team_size;
                        int team_size = lgp->current.team_size - 2; // new team size
                        int min_num_threads = lgp->rtune_config->min_num_threads;
                        //printf("\tMeasure and Tune for next team size: %d\n", team_size);
                        if (team_size < min_num_threads) {
                            team_size = min_num_threads;
                        }
                        lgp->current.team_size = team_size;
                    } else {
                        lgp->current.team_size = lgp->best.team_size;
                        lgp->rtune_best_counter_num_threads++;
                        //printf("\tRTune will use the previous team_size: %d, which has been used %d times\n", team_size, lgp->rtune_best_counter_num_threads++);
                    }
                    ompt_measure_reset(&lgp->rtune_accu);
                }
            } else {
                lgp->num_threads_tuning_done = 1;
            }
        }
#ifdef CPUFREQ_SUPPORT
        if (lgp->num_threads_tuning_done && !lgp->frequency_tuning_done &&
            lgp->current.frequency != lgp->rtune_config->rtune_min_frequency) {
            /** Tune for frequency using greedy strategies */
            if (lgp->rtune_best_counter_frequency < lgp->rtune_config->threshold_best2fixed_frequency) {
                if ((lgp->total_record - 1) % lgp->rtune_config->rtune_num_runs_per_tune == 0) {
                    int diff = ompt_measure_compare(&lgp->best, &lgp->rtune_accu, MEASURE_TIME|MEASURE_PAPI);
                    if (lgp->rtune_config->rtune_log) {
                        fprintf(lgp->rtune_logfile,
                                "---------------------------------------------------------------------------------------------------------------------\n");
                        fprintf(lgp->rtune_logfile,
                                "Tuning from run %d: last config of %d num_threads delivered %d%% perf improvement (now at %.2fms) over the perf with %d threads\n",
                                lgp->total_record, lgp->current.team_size, diff, lgp->rtune_accu.time_stamp,
                                lgp->best.team_size);
                    }
                    if (diff >= 0) { /* better performance for 0%, we should tune */
                        memcpy(&lgp->best, &lgp->rtune_accu, sizeof(ompt_measurement_t));
                        lgp->rtune_best_counter_frequency = 1;

                        /* TODO: selecting next frequency according to the memory boundness or EDP */
                        lgp->best.frequency = lgp->current.frequency;
                        if (lgp->current.frequency->next != NULL) {
                            lgp->current.frequency = lgp->current.frequency->next;
                        }
                        //printf("\tMeasure and Tune for next team size: %d\n", team_size);
                    } else {
                        lgp->current.frequency = lgp->best.frequency;
                        lgp->rtune_best_counter_frequency++;
                        //printf("\tRTune will use the previous team_size: %d, which has been used %d times\n", team_size, lgp->rtune_best_counter_num_threads++);
                    }
                    ompt_measure_reset(&lgp->rtune_accu);
                } else {
                    /* nothing to do other than just do the next run */
                }
            } else {/* tuning completes */
                lgp->frequency_tuning_done = 1;
            }
        }
#endif
    }
    /* if rtune disabled, measure; if rtune_enabled and rtune not complets, measure */
    if ((!lgp->rtune_config->rtune_enabled_num_threads && !lgp->rtune_config->rtune_enabled_frequency)
        || (lgp->rtune_config->rtune_enabled_num_threads && !lgp->num_threads_tuning_done)
        || (lgp->rtune_config->rtune_enabled_frequency && !lgp->frequency_tuning_done)) {
        //ompt_measure_reset(&lgp->current);
        ompt_measure(&lgp->current, MEASURE_ALL);
//        printf("counter: %d, measure at begin: %f\n", lgp->total_record, lgp->current.time_stamp);
    }
#endif
#endif
}

void rtune_master_end_lexgion(ompt_lexgion_t *lgp, ompt_trace_record_t *record) {
#if 1
#ifdef RTUNE_MEASUREMENT_SUPPORT
    if (!lgp->num_threads_tuning_done) {
        ompt_measure_consume(&lgp->current, MEASURE_ALL);
        ompt_measure_accu(&lgp->total, &lgp->current, MEASURE_ALL);
        ompt_measure_accu(&lgp->rtune_accu, &lgp->current, MEASURE_ALL); /* accumulate the measurment data */
        //           printf("counter: %d, measure at end: %f\n", lgp->total_record, lgp->current.time_stamp);
//	    printf("accu time at the end of record %d: %f\n", lgp->total_record, lgp->rtune_accu.time_stamp);
#ifdef OMPT_TRACING_SUPPORT
        record->measurement = lgp->current;
#endif
    }
#endif
#else
#ifdef RTUNE_MEASUREMENT_SUPPORT
    if ((!lgp->rtune_config->rtune_enabled_num_threads && !lgp->rtune_config->rtune_enabled_frequency)
        || (lgp->rtune_config->rtune_enabled_num_threads && !lgp->num_threads_tuning_done)
        || (lgp->rtune_config->rtune_enabled_frequency && !lgp->frequency_tuning_done)) {
        ompt_measure_consume(&lgp->current, MEASURE_ALL);
        ompt_measure_accu(&lgp->total, &lgp->current, MEASURE_ALL);
        ompt_measure_accu(&lgp->rtune_accu, &lgp->current, MEASURE_ALL); /* accumulate the measurment data */
        //           printf("counter: %d, measure at end: %f\n", lgp->total_record, lgp->current.time_stamp);
//	    printf("accu time at the end of record %d: %f\n", lgp->total_record, lgp->rtune_accu.time_stamp);
#ifdef OMPT_TRACING_SUPPORT
        record->measurement = lgp->current;
#endif
    }
#endif
#endif
}


/**
 * read in a config line, which is in the format of "key = value" into the config object. key must be a
 * valid identifier and value can only be "an integer, TRUE or FALSE"
 * @param config
 * @param lineBuffer
 */
static void read_config_line(rtune_config_t * config, char * lineBuffer) {
    char key[64];
    char value[64];
    /* a simple parser to split and grab the key and value */
    sscanf(lineBuffer, "%s = %s", key, value);

    int i;
    for (i=0; i<NUM_CONFIG_KEYS; i++) {
        if (strcasecmp(key, config_keys[i]) == 0) {
            if (strcasecmp(value, "TRUE") == 0) {
                ((int*)config)[i] = 1;
            } else if (strcasecmp(value, "FALSE") == 0) {
                ((int*)config)[i] = 0;
            } else {
                ((int*)config)[i] = atoi(value);
            }
            //printf("%s=%d\n", key, ((unsigned int*)config)[i]);
            break;
        }
    }
}

#ifdef CPUFREQ_SUPPORT
// The preexising policies before we run the code for the system. We need to store it since we are going to change
// the cpufreq policies and we need to restore it back. So far, we only handle max 128 CPUs //
static struct cpufreq_policy *preexist_policy[128];
#endif
/**
 * Read the config file and store the configuration into a rtune_config_t object
 */
void rtune_config_init() {
    /* init the global config */
    int total_cores = sysconf( _SC_NPROCESSORS_ONLN );
    global_config.rtune_enabled = 1;
    global_config.rtune_log = 1;
    global_config.rtune_target = RTUNE_PERFORMANCE;
    global_config.rtune_num_runs_per_tune = 1;
    global_config.rtune_start_run = 1;

    global_config.rtune_enabled_num_threads;
    global_config.max_num_threads = total_cores;
    global_config.min_num_threads = 1;
    global_config.num_threads = total_cores;
    global_config.rtune_initial_num_threads = total_cores;
    global_config.threshold_best2fixed_num_threads = UINT_MAX;

    int i;
#ifdef CPUFREQ_SUPPORT
    global_config.cpufreq_available_freqs = cpufreq_get_available_frequencies(0);

	struct cpufreq_available_frequencies * min_freq = global_config.cpufreq_available_freqs->first;
	while (min_freq->next != NULL) min_freq = min_freq->next;

    global_config.rtune_max_frequency = global_config.cpufreq_available_freqs->first;
    global_config.rtune_min_frequency = min_freq;
    global_config.fixed_frequency = NULL;
    global_config.rtune_initial_frequency = global_config.rtune_max_frequency;
    global_config.rtune_wait_frequency = global_config.rtune_min_frequency;

    global_config.rtune_enabled_frequency = 0;
    global_config.rtune_perthread_frequency = 0;
    global_config.threshold_memboundness = 20;
    global_config.cpufreq_governor = GPUFREQ_GOVERNOR_performance;
    global_config.threashold_edp = 0;
    global_config.threshold_best2fixed_frequency = UINT_MAX;

    //manually set userspace governor first */
    //store the existing policies
    for (i=0; i< total_cores; i++) {
        preexist_policy[i] = cpufreq_get_policy(i);
    }

    struct cpufreq_policy policy;
    policy.min = global_config.rtune_min_frequency->frequency;
    policy.max = global_config.rtune_max_frequency->frequency;
    policy.governor = "userspace";
    //set the userspace policies */
    for (i=0; i< total_cores; i++) {
        cpufreq_set_policy(i, &policy);
    }
#endif

    /* read in config from config file if it is provided via RTUNE_CONFIGFILE env */
    const char* configFileName = getenv("RTUNE_CONFIGFILE");
    FILE *configFile;
    if (configFileName != NULL) {
        configFile = fopen(configFileName, "r");
        if (configFile == NULL) {
            fprintf(stderr, "Cannot open config file %s, ignore. \n", configFileName);
            return;
        }
    } else return;

    /* Read the config file and store the configuration into a rtune_config_t object */
    rtune_config_t *config; /* The very first one should be initialized from the system */
    char lineBuffer[512];
    int max_char = 511;
    void * lexgion_codeptr;
    int read_global_config = 1; /* a flag */
    while (fgets(lineBuffer, max_char, configFile) != NULL) {
        char * ptr = lineBuffer;
        /* ignore comment line which starts with #, and blank line. */
        while(*ptr==' ' || *ptr=='\t') ptr++; // skip leading whitespaces
        if (*ptr == '#') continue; /* comment line, continue */
        if(*ptr=='\r' || *ptr=='\n') continue;

        if (*ptr == '[') {  /* we just finish read the global config,
                             * which is listed before the first lexgion */
            read_global_config = 0; /* the first time scanner sees a '[', reading global config is done */
            sscanf(ptr, "[ lexgion.0x%p ]", &lexgion_codeptr);
            /* copy in the default config values */
            memcpy(&lexgion_rtune_config[num_user_configured_lexgions], &global_config, sizeof(rtune_config_t));
            lexgion_rtune_config[num_user_configured_lexgions].codeptr = (void*)lexgion_codeptr;
            config = &lexgion_rtune_config[num_user_configured_lexgions];
            //printf("Config for lexgion.0x%p, %p, global config: %p\n", lexgion_codeptr, config, &global_config);
            num_user_configured_lexgions ++;
        } else if (read_global_config) { /* first read the global config */
            read_config_line(&global_config, ptr);
        } else { /* read the config for a specified lexgion */
            read_config_line(config, ptr);
        }
    }
    fclose(configFile);
    /* turn rtune_enabled into rtune_enabled_num_threads and rtune_enabled_frequency */
    if (global_config.rtune_enabled) {
        global_config.rtune_enabled_num_threads = 1;
        global_config.rtune_enabled_frequency = 0;
    }
    for (i=0; i<num_user_configured_lexgions; i++) {
        rtune_config_t * config = &lexgion_rtune_config[i];
        if (config->rtune_enabled) {
            config->rtune_enabled_num_threads = 1;
            config->rtune_enabled_frequency = 0;
        }
    }
}

void rtune_config_fini( ) {
#ifdef CPUFREQ_SUPPORT
    //deallocate memory allocated by cpufreq lib
    cpufreq_put_available_frequencies(global_config.cpufreq_available_freqs->first);
    int i;
    //revert back to the original cpufreq policies before the program.
    for (i=0; i< total_cores; i++) {
        cpufreq_set_policy(i, preexist_policy[i]);
        cpufreq_put_policy(preexist_policy[i]);
    }
#endif
}
void print_config() {
    int i;
    printf("\n========================= RTune Configuration ==========================================\n");
    printf("[Default Config]\n");
    for (i=0; i<NUM_CONFIG_KEYS; i++) {
        printf("\t%s = %d\n", config_keys[i], ((int*)&global_config)[i]);
    }

    int j;
    for (j=0; j<num_user_configured_lexgions; j++) {
        rtune_config_t * config = &lexgion_rtune_config[j];
        printf("[lexgion.%p]\n", config->codeptr);
        for (i=0; i<NUM_CONFIG_KEYS; i++) {
            printf("\t%s = %d\n", config_keys[i], ((int*)config)[i]);
        }
    }
    printf("==========================================================================================\n\n");
}


static void print_lexgion_csv(thread_event_map_t * emap, ompt_lexgion_t * lgp) {
#if defined(OMPT_TRACING_SUPPORT) && defined(RTUNE_MEASUREMENT_SUPPORT)
    char csv_filename[64];
    sprintf(csv_filename, "lexgion_%p_%d.csv", lgp->codeptr_ra, lgp->total_record);
    FILE * csv_file = fopen(csv_filename, "w+");
    fprintf(csv_file, "Record ID,team size,");
    ompt_measure_print_header_csv(&lgp->total, csv_file);
    fprintf(csv_file, "\n");
    ompt_trace_record_t * record = lgp->most_recent;
    int count = 0;
    while (record != NULL) {
        fprintf(csv_file, "%d,", record->record_id);
        /* measurement data is stored in the end_record, which can be accessed through the match_record index */
        ompt_trace_record_t * end_record = record->match_record;
        ompt_measure_print_csv(&end_record->measurement, csv_file);
        fprintf(csv_file, "\n");
        record = record->next;
        count++;
    }
    fclose(csv_file);
#endif

}

void print_all_lexgions_csv(thread_event_map_t * emap) {
    if (emap->lexgion_end < 0) return;
    int i;
    char accu_filename[128];
    char aver_filename[128];
    time_t rawtime;
    struct tm *info;
    time( &rawtime );
    info = localtime( &rawtime );

    char time_stamp[32];
    strftime(time_stamp,32,"%Y-%m-%d-%H:%M:%S%z", info);

    sprintf(accu_filename, "lexgion_accu_summary_%d_%s.csv", emap->lexgion_end, time_stamp);
    sprintf(aver_filename, "lexgion_aver_summary_%d_%s.csv", emap->lexgion_end, time_stamp);
    FILE * accu_csv_fid = fopen(accu_filename, "w+");
    FILE * aver_csv_fid = fopen(aver_filename, "w+");
    ompt_lexgion_t * lgp = &emap->lexgions[0];
    fprintf(accu_csv_fid, "address, #exe, ");
    fprintf(aver_csv_fid, "address, #exe, ");
    ompt_measure_print_header_csv(&lgp->total, accu_csv_fid);
    ompt_measure_print_header_csv(&lgp->total, aver_csv_fid);
    fprintf(accu_csv_fid, "\n");
    fprintf(aver_csv_fid, "\n");
    for (i=0; i<=emap->lexgion_end; i++) {
        ompt_lexgion_t * lgp = &emap->lexgions[i];
        fprintf(accu_csv_fid, "%p, %d, ", lgp->codeptr_ra, lgp->total_record);
        fprintf(aver_csv_fid, "%p, %d, ", lgp->codeptr_ra, lgp->total_record);
        ompt_measure_print_csv(&lgp->total, accu_csv_fid);
        fprintf(accu_csv_fid, "\n");
        ompt_measure_print_csv(&lgp->total, aver_csv_fid);
        fprintf(aver_csv_fid, "\n");
        print_lexgion_csv(emap, lgp);
        //if (lgp->most_recent->event == ompt_callback_parallel_begin)
    }
}

static void print_lexgion(int count, ompt_lexgion_t * lgp) {
    if (lgp->end_codeptr2 == NULL) {
        printf("====================== #%d Lexgion %p-->%p, total %d executions: ===============\n",
               count, lgp->codeptr_ra, lgp->end_codeptr, lgp->total_record);
    } else {
        printf("====================== #%d Lexgion %p-->%p | %p, total %d executions: =====\n",
               count, lgp->codeptr_ra, lgp->end_codeptr, lgp->end_codeptr2, lgp->total_record);
    }
    printf("Accumulated Stats: | ");
    ompt_measure_print_header(&lgp->total);
    printf("                   | ");
    ompt_measure_print(&lgp->total);

#if defined(OMPT_TRACING_SUPPORT) && defined(RTUNE_MEASUREMENT_SUPPORT)
    printf("---------------------------------- Execution Records -----------------------------------\n");
    printf("#: name, id\t| ");
    ompt_measure_print_header(&lgp->total);
    ompt_trace_record_t * record = lgp->most_recent;
    count = 1;
    while (record != NULL) {
        printf("#%d: %s, %d\t| ", count, ompt_event_names[record->type], record->record_id);
        /* measurement data is stored in the end_record, which can be accessed through the match_record index */
        ompt_trace_record_t * end_record = record->match_record;
        ompt_measure_print(&end_record->measurement);
        record = record->next;
        count++;
    }
    printf("------------------------------------------------------------------------------------------\n");
#endif
    printf("==========================================================================================\n");
}

void print_lexgions(thread_event_map_t *emap) {
    if (emap->lexgion_end < 0) return;
    int i;
    printf("==============================================================================================\n");
    printf("========================= Lexgions: (Total: %d, Listed from the most recent):==================\n", emap->lexgion_end+1);
    printf("==============================================================================================\n");

    int counter = 0;
    /* search forward from the most recent one */
    for (i=emap->lexgion_recent; i<=emap->lexgion_end; i++) {
        ompt_lexgion_t * lgp = &emap->lexgions[i];
//        printf("event: %d\n", lgp->most_recent->event);
        //if (lgp->most_recent->event == ompt_callback_parallel_begin)
            print_lexgion(++counter, lgp);
    }
    /* search from 0 to most recent one */
    for (i=0; i<emap->lexgion_recent; i++) {
        ompt_lexgion_t * lgp = &emap->lexgions[i];
        //printf("event: %d\n", lgp->most_recent->event);
        //if (lgp->most_recent->event == ompt_callback_parallel_begin)
            print_lexgion(++counter, lgp);
    }
}

/**
 * obsolete and not working anymore.
 * To make it work, all the event map needs to be deeply copied to an array and pass the array
 * to this function
 */
#ifdef OMPT_TRACING_GRAPHML_DUMP
typedef struct graphml_node {
    char * Name;
    char * Shape;
    char * Color;

    char * BorderColor;
    char * BorderType;
    char * BorderWidth;
} graphml_node_graphics_t;

graphml_node_graphics_t graphml_event_node_graphics[64];

#define SET_EVENT_NODE_GRAPHICS(event, shape, color, borderColor, borderType, borderWidth) \
    graphml_event_node_graphics[event].Name = #event; \
    graphml_event_node_graphics[event].Shape = #shape; \
    graphml_event_node_graphics[event].Color = #color; \
    graphml_event_node_graphics[event].BorderColor = #borderColor; \
    graphml_event_node_graphics[event].BorderType = #borderType; \
    graphml_event_node_graphics[event].BorderWidth = #borderWidth;

#define SET_EVENT_NODE_GRAPHICS_DEFAULT_BORDER(event, shape, color) \
    graphml_event_node_graphics[event].Name = #event; \
    graphml_event_node_graphics[event].Shape = shape; \
    graphml_event_node_graphics[event].Color = color; \
    graphml_event_node_graphics[event].BorderColor = "#000000"; \
    graphml_event_node_graphics[event].BorderType = "line"; \
    graphml_event_node_graphics[event].BorderWidth = "1.0";

#define EVENT_NODE_GRAPHICS_LABELNAME(event) \
    graphml_event_node_graphics[event].Name

#define EVENT_NODE_GRAPHICS_SHAPE(event) \
    graphml_event_node_graphics[event].Shape

#define EVENT_NODE_GRAPHICS_COLOR(event) \
    graphml_event_node_graphics[event].Color

#define EVENT_NODE_GRAPHICS_BORDERCOLOR(event) \
    graphml_event_node_graphics[event].BorderColor

#define EVENT_NODE_GRAPHICS_BORDERTYPE(event) \
    graphml_event_node_graphics[event].BorderType

#define EVENT_NODE_GRAPHICS_BORDERWIDTH(event) \
    graphml_event_node_graphics[event].BorderWidth


void ompt_event_maps_to_graphml(thread_event_map_t* maps) {

    SET_EVENT_NODE_GRAPHICS(ompt_callback_thread_begin,     ellipse,          #99CCFF, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_thread_end,       ellipse,          #99CCFF, #000000, line, 4.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_parallel_begin,   rectangle,        #00FF00, #000000, line, 6.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_parallel_end,     rectangle,        #00FF00, #000000, line, 6.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_task_create,      roundrectangle,   #00CC11, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_implicit_task,    roundrectangle,   #00CC11, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_master,           roundrectangle,   #99CCFF, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_work,             roundrectangle,   #99CC11, #000000, line, 1.0);
    //SET_EVENT_NODE_GRAPHICS(ompt_callback_idle,           roundrectangle,   #FFFF00, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_control_tool,     roundrectangle,   #FFFF00, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_sync_region,      roundrectangle,   #FF6347, #000000, line, 1.0);
    SET_EVENT_NODE_GRAPHICS(ompt_callback_sync_region_wait, roundrectangle,   #FF0000, #000000, line, 1.0);

    const char graphml_filename[] = "OMPTrace.graphml";
    FILE *graphml_file = fopen(graphml_filename, "w+");
    /* graphml format //
    <?xml version="1.0" encoding="UTF-8"?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
            http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">

            <graph id="G" edgedefault="directed">
                <node id="n0"/>
                <edge source="n0" target="n2"/>
                <node id="n1"/>
                <node id="n2"/>
            </graph>
        </graphml>
     */
 //   fprintf(graphml_file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
 //   fprintf(graphml_file, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n");
 //   fprintf(graphml_file, "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
 //   fprintf(graphml_file, "xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");

    char * indent = "\t";
    fprintf(graphml_file,"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
    fprintf(graphml_file,"<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" "
            "xmlns:java=\"http://www.yworks.com/xml/yfiles-common/1.0/java\" "
            "xmlns:sys=\"http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0\" "
            "xmlns:x=\"http://www.yworks.com/xml/yfiles-common/markup/2.0\" "
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
            "xmlns:y=\"http://www.yworks.com/xml/graphml\" xmlns:yed=\"http://www.yworks.com/xml/yed/3\" "
            "xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns "
            "http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd\">\n");
    fprintf(graphml_file, "\t<key for=\"port\" id=\"d0\" yfiles.type=\"portgraphics\"/>\n");
    fprintf(graphml_file, "\t<key for=\"port\" id=\"d1\" yfiles.type=\"portgeometry\"/>\n");
    fprintf(graphml_file, "\t<key for=\"port\" id=\"d2\" yfiles.type=\"portuserdata\"/>\n");
    fprintf(graphml_file, "\t<key attr.name=\"color\" attr.type=\"string\" for=\"node\" id=\"d3\">\n");
    fprintf(graphml_file, "\t\t<default><![CDATA[yellow]]></default>\n");
    fprintf(graphml_file, "\t</key>\n");
    fprintf(graphml_file, "\t<key attr.name=\"url\" attr.type=\"string\" for=\"node\" id=\"d4\"/>\n");
    fprintf(graphml_file, "\t<key attr.name=\"description\" attr.type=\"string\" for=\"node\" id=\"d5\"/>\n");
    fprintf(graphml_file, "\t<key for=\"node\" id=\"d6\" yfiles.type=\"nodegraphics\"/>\n");
    fprintf(graphml_file, "\t<key for=\"graphml\" id=\"d7\" yfiles.type=\"resources\"/>\n");
    fprintf(graphml_file, "\t<key attr.name=\"weight\" attr.type=\"double\" for=\"edge\" id=\"d8\"/>\n");
    fprintf(graphml_file, "\t<key attr.name=\"url\" attr.type=\"string\" for=\"edge\" id=\"d9\"/>\n");
    fprintf(graphml_file, "\t<key attr.name=\"description\" attr.type=\"string\" for=\"edge\" id=\"d10\"/>\n");
    fprintf(graphml_file, "\t<key for=\"edge\" id=\"d11\" yfiles.type=\"edgegraphics\"/>\n");
    fprintf(graphml_file, "\t<graph id=\"G\" edgedefault=\"directed\">\n");

    int i;
    for (i=0; i<MAX_NUM_THREADS; i++) {
        thread_event_map_t * emap = get_event_map(i);
        if (i != 0 && emap->thread_id == 0) {/* this is the unused map after the last used one */
            num_threads = i;
            break;
        }

        int j = 0;
        for (j=0; j<emap->counter; j++) {
            ompt_trace_record_t * record = get_trace_record_from_emap(emap, j);
            /* create a graph node for each trace record, we need to create a unqiue node id, set node shape/size and color */
            fprintf(graphml_file, "%s\t<node id=\"%d-%d\">\n", indent, i, j); /* record_id should be asserted to be equal to j */

            if (i != record->thread_id || j != record->record_id) {
                printf("record mismatch with thread_id and record_id\n");
            }
            fprintf(graphml_file, "%s\t\t<data key=\"d6\">\n", indent);
            //fprintf(graphml_file, "%s\t\t\t<y:GenericNode configuration=\"ShinyPlateNode3\">\n", indent);
            fprintf(graphml_file, "%s\t\t\t<y:ShapeNode>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t<y:Geometry height=\"25.0\" width=\"50.0\" x=\"659.0\" y=\"233.0\"/>\n", indent);

            fprintf(graphml_file, "%s\t\t\t\t<y:Fill color=\"%s\" transparent=\"false\"/>\n", indent,
                    EVENT_NODE_GRAPHICS_COLOR(record->event));
            fprintf(graphml_file, "%s\t\t\t\t<y:BorderStyle color=\"%s\" type=\"%s\" width=\"%s\"/>\n", indent,
                    EVENT_NODE_GRAPHICS_BORDERCOLOR(record->event),
                    EVENT_NODE_GRAPHICS_BORDERTYPE(record->event),
                    EVENT_NODE_GRAPHICS_BORDERWIDTH(record->event));
            /* for the label */
            fprintf(graphml_file, "%s\t\t\t\t<y:NodeLabel "
                    "alignment=\"center\" autoSizePolicy=\"content\" "
                    "fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" "
                    "hasBackgroundColor=\"false\" hasLineColor=\"false\" "
                    "height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" "
                    "textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" "
                    "width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\">\n", indent);
            /* the label text itself */
            fprintf(graphml_file, "%s\t\t\t\t\t%s", indent,
                    EVENT_NODE_GRAPHICS_LABELNAME(record->event));
            if (record->endpoint == lexgion_scope_begin)
                fprintf(graphml_file, ":begin");
            else if (record->endpoint == lexgion_scope_end)
                fprintf(graphml_file, ":end");

            if (record->codeptr_ra != NULL)
                fprintf(graphml_file, ":%p", record->codeptr_ra);
            fprintf(graphml_file, ":[%d-%d]\n", i, j);

            fprintf(graphml_file, "%s\t\t\t\t\t<y:LabelModel>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t\t\t<y:SmartNodeLabelModel distance=\"4.0\"/>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t\t</y:LabelModel>\n", indent);

            fprintf(graphml_file, "%s\t\t\t\t\t<y:ModelParameter>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t\t\t<y:SmartNodeLabelModelParameter "
                    "labelRatioX=\"0.0\" labelRatioY=\"0.0\" "
                    "nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" "
                    "offsetX=\"0.0\" offsetY=\"0.0\" "
                    "upX=\"0.0\" upY=\"-1.0\"/>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t\t</y:ModelParameter>\n", indent);

            fprintf(graphml_file, "%s\t\t\t\t</y:NodeLabel>\n", indent);
//            fprintf(graphml_file, "%s\t\t\t\t<y:Shape type=\"rectangle\"/>\n", indent);
            fprintf(graphml_file, "%s\t\t\t\t<y:Shape type=\"%s\"/>\n", indent,
                    EVENT_NODE_GRAPHICS_SHAPE(record->event));

            fprintf(graphml_file, "%s\t\t\t</y:ShapeNode>\n", indent);
            //fprintf(graphml_file, "%s\t\t\t</y:GenericNode>\n", indent);
            fprintf(graphml_file, "%s\t\t</data>\n", indent);
            fprintf(graphml_file, "%s\t</node>\n", indent);


            if (j > 0) { /* create the direct edge between two consecutive trace record */
                fprintf(graphml_file, "%s\t<edge source=\"%d-%d\" target=\"%d-%d\"/>\n", indent, i, j-1, i, j); /* record_id should be asserted to be equal to j */
            }

            if (record->event == ompt_callback_parallel_begin) {
                int k;
                for (k=0; k<record->team_size; k++) {
                    /* k could start with 1, since for the master thread, the link between parallel_begin to the implicit task and the implicit_end to parallel_end
                     * are already set */
                    ompt_trace_record_t * implicit_task_record = record->worker_records[k].implicit_tasks;
                    fprintf(graphml_file, "%s\t<edge source=\"%d-%d\" target=\"%d-%d\"/>\n", indent, i, j, implicit_task_record->thread_id, implicit_task_record->record_id);
                    fprintf(graphml_file, "%s\t<edge source=\"%d-%d\" target=\"%d-%d\"/>\n", indent, implicit_task_record->thread_id, implicit_task_record->match_record, i, record->match_record);
                }
            }
        }
    }

    fprintf(graphml_file, "</graph>");
    fprintf(graphml_file, "</graphml>");
    fclose(graphml_file);

}
#endif
