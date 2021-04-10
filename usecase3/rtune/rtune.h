//
// Created by Yonghong Yan on 9/6/18.
//

#ifndef RTUNE_RTUNE_H
#define RTUNE_RTUNE_H

#include "rtune_config.h"

/* user level API to use RTune library */
/**
 * rtune_policy_t is a float as we will accept a ratio (tradeoff) between performance and energy
 * TBD: what does the ratio represent?
 */
typedef float rtune_policy_t;
typedef enum rtune_const_policy {
    RTUNE_PE_TRADEOFF, /* performance and energy tradeoff */
    RTUNE_PERFORMANCE = 1,
    RTUNE_ENERGY = 2,
    RTUNE_EDP = 3, /* Energy Delay Product */
} rtune_const_policy_t;

#ifdef  __cplusplus
extern "C" {
#endif

void rtune_master_begin();
void rtune_master_end();

extern void rtune_global_init();
extern void rtune_global_fini();
#ifdef  __cplusplus
};
#endif
#endif //RTUNE_RTUNE_H
