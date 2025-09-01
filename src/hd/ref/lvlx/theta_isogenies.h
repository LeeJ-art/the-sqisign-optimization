/** @file
 *
 * @authors Antonin Leroux
 *
 * @brief the theta isogeny header
 */

#ifndef THETA_ISOGENY_H
#define THETA_ISOGENY_H

#include <sqisign_namespace.h>
#include <ec.h>
#include <fp2.h>
#include "theta_structure.h"
#include <hd.h>
#include <hd_splitting_transforms.h>

/*New vectorization*/
void choose_small(theta_point_t* a, theta_point_t* b);
void copy_structure(theta_structure_t *out, theta_structure_t *A);
void reduce_q(uint32x4_t* a);
void power_to_num(uint32x4_t* a, uint64_t x);
#endif
