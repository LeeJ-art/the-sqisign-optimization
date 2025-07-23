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


void to_squared_theta_batched(uint32x4_t* out, uint32x4_t *a);
//new
void fp_mul_batched(uint32x2_t *out, uint32x4_t *a, uint32x4_t *b);
void prop_2(uint32x4_t *n);
uint32x4_t div5(uint32x4_t* in);
void fp2_mul_batched(uint32x4_t *out, uint32x4_t *a, uint32x4_t *b);
void fp2_sqr_batched(uint32x4_t* b, uint32x4_t *a);
void theta_montback(theta_point_t* a, fp_t* mb);
void choose_small(theta_point_t* a, theta_point_t* b);
void copy_structure(theta_structure_t *out, theta_structure_t *A);
#endif
