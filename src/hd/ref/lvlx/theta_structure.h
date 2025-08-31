/** @file
 *
 * @authors Antonin Leroux
 *
 * @brief the theta structure header
 */

#ifndef THETA_STRUCTURE_H
#define THETA_STRUCTURE_H

#include <ec.h>
#include <fp2.h>
#include <hd.h>

#include <arm_neon.h>
#include "theta_isogenies.h"

/** @internal
 * @ingroup hd_module
 * @defgroup hd_theta Functions for theta structures
 * @{
 */

/**
 * @brief Perform the hadamard transform on a theta point
 *
 * @param out Output: the theta_point
 * @param in a theta point*
 * in = (x,y,z,t)
 * out = (x+y+z+t, x-y+z-t, x+y-z-t, x-y-z+t)
 *
 */

static inline void
hadamard(theta_point_t *out, const theta_point_t *in)
{
    fp2_t t1, t2, t3, t4;

    // t1 = x + y
    fp2_add(&t1, &in->x, &in->y);
    // t2 = x - y
    fp2_sub(&t2, &in->x, &in->y);
    // t3 = z + t
    fp2_add(&t3, &in->z, &in->t);
    // t4 = z - t
    fp2_sub(&t4, &in->z, &in->t);

    fp2_add(&out->x, &t1, &t3);
    fp2_add(&out->y, &t2, &t4);
    fp2_sub(&out->z, &t1, &t3);
    fp2_sub(&out->t, &t2, &t4);
}


/* new batched funcs */
static inline void hadamard_vec(uint32x4_t*out, uint32x4_t* in){
    uint32x4_t tmp[18], q2[9];
    uint32x4_t reCarry, imCarry;

    for(int i = 0;i<8;i++) q2[i] = vdupq_n_u32(0x3FFFFFFE);
    q2[8] = vdupq_n_u32(0x9FFFE);

    for(int i = 0;i<18;i++){
      tmp[0][0] = in[i][0] + in[i][1];
      tmp[0][1] = (in[i][0] + q2[i%9][0]) - in[i][1];
      tmp[0][2] = in[i][2] + in[i][3];
      tmp[0][3] = (in[i][2] + q2[i%9][0]) - in[i][3];

      out[i][0] = tmp[0][0] + tmp[0][2];
      out[i][1] = tmp[0][1] + tmp[0][3];
      out[i][2] = tmp[0][0] + (q2[i%9][0] - tmp[0][2]);
      out[i][3] = tmp[0][1] + (q2[i%9][0] - tmp[0][3]);
    }

    prop_2(out);
    prop_2(out+9);
    reCarry = div5(out+8);
    imCarry = div5(out+17);
    out[0] = vaddq_u32(out[0], reCarry);
    out[9] = vaddq_u32(out[9], imCarry);
}

static inline 
void hadamard_transpose(uint32x4_t *Out, theta_point_t In){
    // hadamard
    theta_point_t tmp;
    hadamard(&tmp, &In);
    transpose(Out, tmp);
}


static inline
void hadamard_itranspose(theta_point_t *Out, uint32x4_t *In){
    theta_point_t tmp;    
    itranspose(&tmp, In);
    hadamard(Out, &tmp);
}

//data structure transform
void structure2point(theta_point_t* out, theta_structure_t *A);
void structure2point_reindex(theta_point_t* out, theta_structure_t *A);
void point2structure(theta_structure_t* A, theta_point_t *out);

/**
 * @brief Square the coordinates of a theta point
 * @param out Output: the theta_point
 * @param in a theta point*
 * in = (x,y,z,t)
 * out = (x^2, y^2, z^2, t^2)
 *
 */
static inline void
pointwise_square(theta_point_t *out, const theta_point_t *in)
{
    fp2_sqr(&out->x, &in->x);
    fp2_sqr(&out->y, &in->y);
    fp2_sqr(&out->z, &in->z);
    fp2_sqr(&out->t, &in->t);
}

/**
 * @brief Square the coordinates and then perform the hadamard transform
 *
 * @param out Output: the theta_point
 * @param in a theta point*
 * in = (x,y,z,t)
 * out = (x^2+y^2+z^2+t^2, x^2-y^2+z^2-t^2, x^2+y^2-z^2-t^2, x^2-y^2-z^2+t^2)
 *
 */
static inline void
to_squared_theta(theta_point_t *out, const theta_point_t *in)
{
    pointwise_square(out, in);
    hadamard(out, out);
}

/**
 * @brief Perform the theta structure precomputation
 *
 * @param A Output: the theta_structure
 *
 * if A.null_point = (x,y,z,t)
 * if (xx,yy,zz,tt) = to_squared_theta(A.null_point)
 * Computes y0,z0,t0,Y0,Z0,T0 = x/y,x/z,x/t,XX/YY,XX/ZZ,XX/TT
 *
 */
void theta_precomputation(theta_structure_t *A);
bool theta_precomputation_vec(uint32x4_t* a0, uint32x4_t* a1, uint32x4_t* A_null);
void transpose_theta_precomputation_vec(theta_structure_t *A);

/**
 * @brief Compute the double of the theta point in on the theta struc A
 *
 * @param out Output: the theta_point
 * @param A a theta structure
 * @param in a theta point in the theta structure A
 * in = (x,y,z,t)
 * out = [2] (x,y,z,t)
 * /!\ assumes that no coordinates is zero and that the precomputation of A has been done
 *
 */
void double_point(theta_point_t *out, theta_structure_t *A, const theta_point_t *in);
void double_point_vec(theta_point_t *out, theta_structure_t *A, const theta_point_t *in);

/**
 * @brief Compute the iterated double of the theta point in on the theta struc A
 *
 * @param out Output: the theta_point
 * @param A a theta structure
 * @param in a theta point in the theta structure A
 * @param exp the exponent
 * in = (x,y,z,t)
 * out = [2^2] (x,y,z,t)
 * /!\ assumes that no coordinates is zero and that the precomputation of A has been done
 *
 */
void double_iter(theta_point_t *out, theta_structure_t *A, const theta_point_t *in, int exp);
void double_iter_vec(theta_point_t *out, theta_structure_t *A, const theta_point_t *in, int exp);
void double_iter_vec_randomized(uint32x4_t *out, theta_structure_t *A, uint32x4_t *in, int exp);
/*
 * @brief Check if a theta point is a product theta point
 *
 * @param P a theta point
 * @return 0xFFFFFFFF if true, zero otherwise
 */
uint32_t is_product_theta_point(const theta_point_t *P);

// end hd_theta
/**
 * @}
 */

#endif
