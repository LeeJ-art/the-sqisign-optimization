/** @file
 *
 * @authors Antonin Leroux
 *
 * @brief The HD-isogenies algorithm required by the signature
 *
 */

#ifndef HD_H
#define HD_H

#include <sqisign_namespace.h>
#include <ec.h>
#include <stdio.h>
#include <arm_neon.h>

/** @defgroup hd_module Abelian surfaces and their isogenies
 * @{
 */

#define HD_extra_torsion 2

/** @defgroup hd_struct Data structures for dimension 2
 * @{
 */

/** @brief Type for couple point with XZ coordinates
 * @typedef theta_couple_point_t
 *
 * @struct theta_couple_point
 *
 * Structure for the couple point on an elliptic product
 * using XZ coordinates
 */
typedef struct theta_couple_point
{
    ec_point_t P1;
    ec_point_t P2;
} theta_couple_point_t;

/** @brief Type for three couple points T1, T2, T1-T2 with XZ coordinates
 * @typedef theta_kernel_couple_points_t
 *
 * @struct theta_kernel_couple_points
 *
 * Structure for a triple of theta couple points T1, T2 and T1 - T2
 */
typedef struct theta_kernel_couple_points
{
    theta_couple_point_t T1;
    theta_couple_point_t T2;
    theta_couple_point_t T1m2;
} theta_kernel_couple_points_t;

/** @brief Type for couple point with XYZ coordinates
 * @typedef theta_couple_jac_point_t
 *
 * @struct theta_couple_jac_point
 *
 * Structure for the couple point on an elliptic product
 * using XYZ coordinates
 */
typedef struct theta_couple_jac_point
{
    jac_point_t P1;
    jac_point_t P2;
} theta_couple_jac_point_t;

/** @brief Type for couple curve *
 * @typedef theta_couple_curve_t
 *
 * @struct theta_couple_curve
 *
 * the  theta_couple_curve structure
 */
typedef struct theta_couple_curve
{
    ec_curve_t E1;
    ec_curve_t E2;
} theta_couple_curve_t;

/** @brief Type for a product E1 x E2 with corresponding bases
 * @typedef theta_couple_curve_with_basis_t
 *
 * @struct theta_couple_curve_with_basis
 *
 * tType for a product E1 x E2 with corresponding bases Ei[2^n]
 */
typedef struct theta_couple_curve_with_basis
{
    ec_curve_t E1;
    ec_curve_t E2;
    ec_basis_t B1;
    ec_basis_t B2;
} theta_couple_curve_with_basis_t;

/** @brief Type for theta point *
 * @typedef theta_point_t
 *
 * @struct theta_point
 *
 * the  theta_point structure used
 */
typedef struct theta_point
{
    fp2_t x;
    fp2_t y;
    fp2_t z;
    fp2_t t;
} theta_point_t;

/** @brief Type for theta point with repeating components
 * @typedef theta_point_compact_t
 *
 * @struct theta_point_compact
 *
 * the  theta_point structure used for points with repeated components
 */
typedef struct theta_point_compact
{
    fp2_t x;
    fp2_t y;
} theta_point_compact_t;

/** @brief Type for theta structure *
 * @typedef theta_structure_t
 *
 * @struct theta_structure
 *
 * the  theta_structure structure used
 */
typedef struct theta_structure
{
    theta_point_t null_point;
    bool precomputation;

    // Eight precomputed values used for doubling and
    // (2,2)-isogenies.
    fp2_t XYZ0;
    fp2_t YZT0;
    fp2_t XZT0;
    fp2_t XYT0;

    fp2_t xyz0;
    fp2_t yzt0;
    fp2_t xzt0;
    fp2_t xyt0;
} theta_structure_t;

/** @brief A 2x2 matrix used for action by translation
 * @typedef translation_matrix_t
 *
 * @struct translation_matrix
 *
 * Structure to hold 4 fp2_t elements representing a 2x2 matrix used when computing
 * a compatible theta structure during gluing.
 */
typedef struct translation_matrix
{
    fp2_t g00;
    fp2_t g01;
    fp2_t g10;
    fp2_t g11;
} translation_matrix_t;

/** @brief A 4x4 matrix used for basis changes
 * @typedef basis_change_matrix_t
 *
 * @struct basis_change_matrix
 *
 * Structure to hold 16 elements representing a 4x4 matrix used for changing
 * the basis of a theta point.
 */
typedef struct basis_change_matrix
{
    fp2_t m[4][4];
} basis_change_matrix_t;

/** @brief Type for gluing (2,2) theta isogeny *
 * @typedef theta_gluing_t
 *
 * @struct theta_gluing
 *
 * the  theta_gluing structure
 */
typedef struct theta_gluing
{

    theta_couple_curve_t domain;
    theta_couple_jac_point_t xyK1_8;
    theta_point_compact_t imageK1_8;
    basis_change_matrix_t M;
    theta_point_t precomputation;
    theta_point_t codomain;

} theta_gluing_t;

/** @brief Type for standard (2,2) theta isogeny *
 * @typedef theta_isogeny_t
 *
 * @struct theta_isogeny
 *
 * the  theta_isogeny structure
 */
typedef struct theta_isogeny
{
    theta_point_t T1_8;
    theta_point_t T2_8;
    bool hadamard_bool_1;
    bool hadamard_bool_2;
    theta_structure_t domain;
    theta_point_t precomputation;
    theta_structure_t codomain;
} theta_isogeny_t;

/** @brief Type for splitting isomorphism *
 * @typedef theta_splitting_t
 *
 * @struct theta_splitting
 *
 * the theta_splitting structure
 */
typedef struct theta_splitting
{
    basis_change_matrix_t M;
    theta_structure_t B;

} theta_splitting_t;

// end of hd_struct
/**
 * @}
 */

/** @defgroup hd_functions Functions for dimension 2
 * @{
 */

/**
 * @brief Compute the double of the theta couple point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_point
 * @param in the theta couple point in the elliptic product
 * @param E1E2 an elliptic product
 * in = (P1,P2)
 * out = [2] (P1,P2)
 *
 */
void double_couple_point(theta_couple_point_t *out, const theta_couple_point_t *in, const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the iterated double of the theta couple point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_point
 * @param n : the number of iteration
 * @param E1E2 an elliptic product
 * @param in the theta couple point in the elliptic product
 * in = (P1,P2)
 * out = [2^n] (P1,P2)
 *
 */
void double_couple_point_iter(theta_couple_point_t *out,
                              unsigned n,
                              const theta_couple_point_t *in,
                              const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the addition of two points in (X : Y : Z) coordinates on the elliptic product E12
 *
 * @param out Output: the theta_couple_jac_point
 * @param T1 the theta couple jac point in the elliptic product
 * @param T2 the theta couple jac point in the elliptic product
 * @param E1E2 an elliptic product
 * in  = (P1, P2), (Q1, Q2)
 * out = (P1 + Q1, P2 + Q2)
 *
 **/
void add_couple_jac_points(theta_couple_jac_point_t *out,
                           const theta_couple_jac_point_t *T1,
                           const theta_couple_jac_point_t *T2,
                           const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the double of the theta couple point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_point
 * @param in the theta couple point in the elliptic product
 * @param E1E2 an elliptic product
 * in = (P1,P2)
 * out = [2] (P1,P2)
 *
 */
void double_couple_jac_point(theta_couple_jac_point_t *out,
                             const theta_couple_jac_point_t *in,
                             const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the iterated double of the theta couple jac point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_jac_point
 * @param n : the number of iteration
 * @param in the theta couple jac point in the elliptic product
 * @param E1E2 an elliptic product
 * in  = (P1,P2)
 * out = [2^n] (P1,P2)
 *
 */
void double_couple_jac_point_iter(theta_couple_jac_point_t *out,
                                  unsigned n,
                                  const theta_couple_jac_point_t *in,
                                  const theta_couple_curve_t *E1E2);

void double_couple_jac_point_iter_vec(theta_couple_jac_point_t *out,
                                  unsigned n,
                                  const theta_couple_jac_point_t *in,
                                  const theta_couple_curve_t *E1E2);

/**
 * @brief A forgetful function which returns (X : Z) points given a pair of (X : Y : Z) points
 *
 * @param P Output: the theta_couple_point
 * @param xyP : the theta_couple_jac_point
 **/
void couple_jac_to_xz(theta_couple_point_t *P, const theta_couple_jac_point_t *xyP);

/**
 * @brief Compute a (2,2) isogeny chain in dimension 2 between elliptic
 * products in the theta_model and evaluate at a list of points of the form
 * (P1,0) or (0,P2). Returns 0 if the codomain fails to split (or there is
 * an error during the computation) and 1 otherwise.
 *
 * @param n : the length of the isogeny chain
 * @param E12 an elliptic curve product
 * @param ker T1, T2 and T1-T2. couple points on E12[2^(n+2)]
 * @param extra_torsion boolean indicating if we give the points in E12[2^n] or
 * E12[2^(n+HD_extra_torsion)]
 * @param E34 Output: the codomain curve
 * @param P12 Input/Output: pointer to points to be pushed through the isogeny (in-place)
 * @param numP: length of the list of points given in P12 (can be zero)
 * @returns 1 on success 0 on failure
 *
 */
int theta_chain_compute_and_eval(unsigned n,
                                 /*const*/ theta_couple_curve_t *E12,
                                 const theta_kernel_couple_points_t *ker,
                                 bool extra_torsion,
                                 theta_couple_curve_t *E34,
                                 theta_couple_point_t *P12,
                                 size_t numP);

/**
 * @brief Compute a (2,2) isogeny chain in dimension 2 between elliptic
 * products in the theta_model and evaluate at a list of points of the form
 * (P1,0) or (0,P2). Returns 0 if the codomain fails to split (or there is
 * an error during the computation) and 1 otherwise.
 * Compared to theta_chain_compute_and_eval, it does extra isotropy
 * checks on the kernel.
 *
 * @param n : the length of the isogeny chain
 * @param E12 an elliptic curve product
 * @param ker T1, T2 and T1-T2. couple points on E12[2^(n+2)]
 * @param extra_torsion boolean indicating if we give the points in E12[2^n] or
 * E12[2^(n+HD_extra_torsion)]
 * @param E34 Output: the codomain curve
 * @param P12 Input/Output: pointer to points to be pushed through the isogeny (in-place)
 * @param numP: length of the list of points given in P12 (can be zero)
 * @returns 1 on success 0 on failure
 *
 */
int theta_chain_compute_and_eval_verify(unsigned n,
                                        /*const*/ theta_couple_curve_t *E12,
                                        const theta_kernel_couple_points_t *ker,
                                        bool extra_torsion,
                                        theta_couple_curve_t *E34,
                                        theta_couple_point_t *P12,
                                        size_t numP);

/**
 * @brief Compute a (2,2) isogeny chain in dimension 2 between elliptic
 * products in the theta_model and evaluate at a list of points of the form
 * (P1,0) or (0,P2). Returns 0 if the codomain fails to split (or there is
 * an error during the computation) and 1 otherwise.
 * Compared to theta_chain_compute_and_eval, it selects a random Montgomery
 * model of the codomain.
 *
 * @param n : the length of the isogeny chain
 * @param E12 an elliptic curve product
 * @param ker T1, T2 and T1-T2. couple points on E12[2^(n+2)]
 * @param extra_torsion boolean indicating if we give the points in E12[2^n] or
 * E12[2^(n+HD_extra_torsion)]
 * @param E34 Output: the codomain curve
 * @param P12 Input/Output: pointer to points to be pushed through the isogeny (in-place)
 * @param numP: length of the list of points given in P12 (can be zero)
 * @returns 1 on success, 0 on failure
 *
 */
int theta_chain_compute_and_eval_randomized(unsigned n,
                                            /*const*/ theta_couple_curve_t *E12,
                                            const theta_kernel_couple_points_t *ker,
                                            bool extra_torsion,
                                            theta_couple_curve_t *E34,
                                            theta_couple_point_t *P12,
                                            size_t numP);

/**
 * @brief Given a bases B1 on E1 and B2 on E2 copies this to create a kernel
 *         on E1 x E2 as couple points T1, T2 and T1 - T2
 *
 * @param ker Output: a kernel for dim_two_isogenies (T1, T2, T1-T2)
 * @param B1 Input basis on E1
 * @param B2 Input basis on E2
 **/
void copy_bases_to_kernel(theta_kernel_couple_points_t *ker, const ec_basis_t *B1, const ec_basis_t *B2);

/**
 * @brief Given a couple of points (P1, P2) on a couple of curves (E1, E2)
 * this function tests if both points are of order exactly 2^t
 *
 * @param T: couple point (P1, P2)
 * @param E: a couple of curves (E1, E2)
 * @param t: an integer
 * @returns 0xFFFFFFFF on success, 0 on failure
 */
static int
test_couple_point_order_twof(const theta_couple_point_t *T, const theta_couple_curve_t *E, int t)
{
    int check_P1 = test_point_order_twof(&T->P1, &E->E1, t);
    int check_P2 = test_point_order_twof(&T->P2, &E->E2, t);

    return check_P1 & check_P2;
}

/*New vectorization*/
static inline void theta_montback(theta_point_t* a, fp_t* mb){
    fp_mul(&(a[0].x.re), &(a[0].x.re), mb);
    fp_mul(&(a[0].x.im), &(a[0].x.im), mb);
    fp_mul(&(a[0].y.re), &(a[0].y.re), mb);
    fp_mul(&(a[0].y.im), &(a[0].y.im), mb);
    fp_mul(&(a[0].z.re), &(a[0].z.re), mb);
    fp_mul(&(a[0].z.im), &(a[0].z.im), mb);
    fp_mul(&(a[0].t.re), &(a[0].t.re), mb);
    fp_mul(&(a[0].t.im), &(a[0].t.im), mb);
}

static inline void u32_montback(uint32x4_t* a, uint32x4_t* mb){
    fp_mul_batched((uint32x2_t*)a, a, mb);
    fp_mul_batched((uint32x2_t*)(a+9), a+9, mb);
}

static 
inline void transpose(uint32x4_t *Out, theta_point_t In){
    uint64_t mask = ((uint64_t)1<<29)-1;
    uint32_t in32[4][18] = {0};

    // re-cast
    in32[0][0] = (uint32_t)(mask & In.x.re[0]);
    in32[1][0] = (uint32_t)(mask & In.y.re[0]);
    in32[2][0] = (uint32_t)(mask & In.z.re[0]);
    in32[3][0] = (uint32_t)(mask & In.t.re[0]);

    in32[0][1] = (uint32_t)(mask & ((In.x.re[0]>>29) + (In.x.re[1]<<22)));
    in32[1][1] = (uint32_t)(mask & ((In.y.re[0]>>29) + (In.y.re[1]<<22)));
    in32[2][1] = (uint32_t)(mask & ((In.z.re[0]>>29) + (In.z.re[1]<<22)));
    in32[3][1] = (uint32_t)(mask & ((In.t.re[0]>>29) + (In.t.re[1]<<22)));

    in32[0][2] = (uint32_t)(mask & (In.x.re[1]>>7));
    in32[1][2] = (uint32_t)(mask & (In.y.re[1]>>7));
    in32[2][2] = (uint32_t)(mask & (In.z.re[1]>>7));
    in32[3][2] = (uint32_t)(mask & (In.t.re[1]>>7));

    in32[0][3] = (uint32_t)(mask & ((In.x.re[1]>>36) + (In.x.re[2]<<15)));
    in32[1][3] = (uint32_t)(mask & ((In.y.re[1]>>36) + (In.y.re[2]<<15)));
    in32[2][3] = (uint32_t)(mask & ((In.z.re[1]>>36) + (In.z.re[2]<<15)));
    in32[3][3] = (uint32_t)(mask & ((In.t.re[1]>>36) + (In.t.re[2]<<15)));

    in32[0][4] = (uint32_t)(mask & (In.x.re[2]>>14));
    in32[1][4] = (uint32_t)(mask & (In.y.re[2]>>14));
    in32[2][4] = (uint32_t)(mask & (In.z.re[2]>>14));
    in32[3][4] = (uint32_t)(mask & (In.t.re[2]>>14));

    in32[0][5] = (uint32_t)(mask & ((In.x.re[2]>>43) + (In.x.re[3]<<8)));
    in32[1][5] = (uint32_t)(mask & ((In.y.re[2]>>43) + (In.y.re[3]<<8)));
    in32[2][5] = (uint32_t)(mask & ((In.z.re[2]>>43) + (In.z.re[3]<<8)));
    in32[3][5] = (uint32_t)(mask & ((In.t.re[2]>>43) + (In.t.re[3]<<8)));

    in32[0][6] = (uint32_t)(mask & (In.x.re[3]>>21));
    in32[1][6] = (uint32_t)(mask & (In.y.re[3]>>21));
    in32[2][6] = (uint32_t)(mask & (In.z.re[3]>>21));
    in32[3][6] = (uint32_t)(mask & (In.t.re[3]>>21));

    in32[0][7] = (uint32_t)(mask & ((In.x.re[3]>>50) + (In.x.re[4]<<1)));
    in32[1][7] = (uint32_t)(mask & ((In.y.re[3]>>50) + (In.y.re[4]<<1)));
    in32[2][7] = (uint32_t)(mask & ((In.z.re[3]>>50) + (In.z.re[4]<<1)));
    in32[3][7] = (uint32_t)(mask & ((In.t.re[3]>>50) + (In.t.re[4]<<1)));

    in32[0][8] = (uint32_t)(mask & (In.x.re[4]>>28));
    in32[1][8] = (uint32_t)(mask & (In.y.re[4]>>28));
    in32[2][8] = (uint32_t)(mask & (In.z.re[4]>>28));
    in32[3][8] = (uint32_t)(mask & (In.t.re[4]>>28));

    in32[0][9] = (uint32_t)(mask & (In.x.im[0]));
    in32[1][9] = (uint32_t)(mask & (In.y.im[0]));
    in32[2][9] = (uint32_t)(mask & (In.z.im[0]));
    in32[3][9] = (uint32_t)(mask & (In.t.im[0]));

    in32[0][10] = (uint32_t)(mask & ((In.x.im[0]>>29) + (In.x.im[1]<<22)));
    in32[1][10] = (uint32_t)(mask & ((In.y.im[0]>>29) + (In.y.im[1]<<22)));
    in32[2][10] = (uint32_t)(mask & ((In.z.im[0]>>29) + (In.z.im[1]<<22)));
    in32[3][10] = (uint32_t)(mask & ((In.t.im[0]>>29) + (In.t.im[1]<<22)));

    in32[0][11] = (uint32_t)(mask & (In.x.im[1]>>7));
    in32[1][11] = (uint32_t)(mask & (In.y.im[1]>>7));
    in32[2][11] = (uint32_t)(mask & (In.z.im[1]>>7));
    in32[3][11] = (uint32_t)(mask & (In.t.im[1]>>7));

    in32[0][12] = (uint32_t)(mask & ((In.x.im[1]>>36) + (In.x.im[2]<<15)));
    in32[1][12] = (uint32_t)(mask & ((In.y.im[1]>>36) + (In.y.im[2]<<15)));
    in32[2][12] = (uint32_t)(mask & ((In.z.im[1]>>36) + (In.z.im[2]<<15)));
    in32[3][12] = (uint32_t)(mask & ((In.t.im[1]>>36) + (In.t.im[2]<<15)));

    in32[0][13] = (uint32_t)(mask & (In.x.im[2]>>14));
    in32[1][13] = (uint32_t)(mask & (In.y.im[2]>>14));
    in32[2][13] = (uint32_t)(mask & (In.z.im[2]>>14));
    in32[3][13] = (uint32_t)(mask & (In.t.im[2]>>14));

    in32[0][14] = (uint32_t)(mask & ((In.x.im[2]>>43) + (In.x.im[3]<<8)));
    in32[1][14] = (uint32_t)(mask & ((In.y.im[2]>>43) + (In.y.im[3]<<8)));
    in32[2][14] = (uint32_t)(mask & ((In.z.im[2]>>43) + (In.z.im[3]<<8)));
    in32[3][14] = (uint32_t)(mask & ((In.t.im[2]>>43) + (In.t.im[3]<<8)));

    in32[0][15] = (uint32_t)(mask & (In.x.im[3]>>21));
    in32[1][15] = (uint32_t)(mask & (In.y.im[3]>>21));
    in32[2][15] = (uint32_t)(mask & (In.z.im[3]>>21));
    in32[3][15] = (uint32_t)(mask & (In.t.im[3]>>21));

    in32[0][16] = (uint32_t)(mask & ((In.x.im[3]>>50) + (In.x.im[4]<<1)));
    in32[1][16] = (uint32_t)(mask & ((In.y.im[3]>>50) + (In.y.im[4]<<1)));
    in32[2][16] = (uint32_t)(mask & ((In.z.im[3]>>50) + (In.z.im[4]<<1)));
    in32[3][16] = (uint32_t)(mask & ((In.t.im[3]>>50) + (In.t.im[4]<<1)));

    in32[0][17] = (uint32_t)(mask & (In.x.im[4]>>28));
    in32[1][17] = (uint32_t)(mask & (In.y.im[4]>>28));
    in32[2][17] = (uint32_t)(mask & (In.z.im[4]>>28));
    in32[3][17] = (uint32_t)(mask & (In.t.im[4]>>28));

    // transpose
    for(int i = 0;i<18;i++){
        uint32x4_t tmp = {in32[0][i], in32[1][i], in32[2][i], in32[3][i]};
        Out[i] = tmp;
    }
}

static inline
void itranspose(theta_point_t *Out, uint32x4_t *In){
    uint32_t in32[4][18];

    // itranspose
    for(int i = 0;i<4;i++){
        for(int j = 0;j<18;j++){
            in32[i][j] = In[j][i];
        }
    }

    // re-cast
    uint64_t mask = ((uint64_t)1<<51)-1;
    Out->x.re[0] = mask & (in32[0][0] + ((uint64_t)in32[0][1]<<29));
    Out->x.im[0] = mask & (in32[0][9] + ((uint64_t)in32[0][10]<<29));
    Out->y.re[0] = mask & (in32[1][0] + ((uint64_t)in32[1][1]<<29));
    Out->y.im[0] = mask & (in32[1][9] + ((uint64_t)in32[1][10]<<29));
    Out->z.re[0] = mask & (in32[2][0] + ((uint64_t)in32[2][1]<<29));
    Out->z.im[0] = mask & (in32[2][9] + ((uint64_t)in32[2][10]<<29));
    Out->t.re[0] = mask & (in32[3][0] + ((uint64_t)in32[3][1]<<29));
    Out->t.im[0] = mask & (in32[3][9] + ((uint64_t)in32[3][10]<<29));

    Out->x.re[1] = mask & (((uint64_t)in32[0][1]>>22) + ((uint64_t)in32[0][2]<<7) + ((uint64_t)in32[0][3]<<36));
    Out->x.im[1] = mask & (((uint64_t)in32[0][10]>>22) + ((uint64_t)in32[0][11]<<7) + ((uint64_t)in32[0][12]<<36));
    Out->y.re[1] = mask & (((uint64_t)in32[1][1]>>22) + ((uint64_t)in32[1][2]<<7) + ((uint64_t)in32[1][3]<<36));
    Out->y.im[1] = mask & (((uint64_t)in32[1][10]>>22) + ((uint64_t)in32[1][11]<<7) + ((uint64_t)in32[1][12]<<36));
    Out->z.re[1] = mask & (((uint64_t)in32[2][1]>>22) + ((uint64_t)in32[2][2]<<7) + ((uint64_t)in32[2][3]<<36));
    Out->z.im[1] = mask & (((uint64_t)in32[2][10]>>22) + ((uint64_t)in32[2][11]<<7) + ((uint64_t)in32[2][12]<<36));
    Out->t.re[1] = mask & (((uint64_t)in32[3][1]>>22) + ((uint64_t)in32[3][2]<<7) + ((uint64_t)in32[3][3]<<36));
    Out->t.im[1] = mask & (((uint64_t)in32[3][10]>>22) + ((uint64_t)in32[3][11]<<7) + ((uint64_t)in32[3][12]<<36));

    Out->x.re[2] = mask & (((uint64_t)in32[0][3]>>15) + ((uint64_t)in32[0][4]<<14) + ((uint64_t)in32[0][5]<<43));
    Out->x.im[2] = mask & (((uint64_t)in32[0][12]>>15) + ((uint64_t)in32[0][13]<<14) + ((uint64_t)in32[0][14]<<43));
    Out->y.re[2] = mask & (((uint64_t)in32[1][3]>>15) + ((uint64_t)in32[1][4]<<14) + ((uint64_t)in32[1][5]<<43));
    Out->y.im[2] = mask & (((uint64_t)in32[1][12]>>15) + ((uint64_t)in32[1][13]<<14) + ((uint64_t)in32[1][14]<<43));
    Out->z.re[2] = mask & (((uint64_t)in32[2][3]>>15) + ((uint64_t)in32[2][4]<<14) + ((uint64_t)in32[2][5]<<43));
    Out->z.im[2] = mask & (((uint64_t)in32[2][12]>>15) + ((uint64_t)in32[2][13]<<14) + ((uint64_t)in32[2][14]<<43));
    Out->t.re[2] = mask & (((uint64_t)in32[3][3]>>15) + ((uint64_t)in32[3][4]<<14) + ((uint64_t)in32[3][5]<<43));
    Out->t.im[2] = mask & (((uint64_t)in32[3][12]>>15) + ((uint64_t)in32[3][13]<<14) + ((uint64_t)in32[3][14]<<43));

    Out->x.re[3] = mask & (((uint64_t)in32[0][5]>>8) + ((uint64_t)in32[0][6]<<21) + ((uint64_t)in32[0][7]<<50));
    Out->x.im[3] = mask & (((uint64_t)in32[0][14]>>8) + ((uint64_t)in32[0][15]<<21) + ((uint64_t)in32[0][16]<<50));
    Out->y.re[3] = mask & (((uint64_t)in32[1][5]>>8) + ((uint64_t)in32[1][6]<<21) + ((uint64_t)in32[1][7]<<50));
    Out->y.im[3] = mask & (((uint64_t)in32[1][14]>>8) + ((uint64_t)in32[1][15]<<21) + ((uint64_t)in32[1][16]<<50));
    Out->z.re[3] = mask & (((uint64_t)in32[2][5]>>8) + ((uint64_t)in32[2][6]<<21) + ((uint64_t)in32[2][7]<<50));
    Out->z.im[3] = mask & (((uint64_t)in32[2][14]>>8) + ((uint64_t)in32[2][15]<<21) + ((uint64_t)in32[2][16]<<50));
    Out->t.re[3] = mask & (((uint64_t)in32[3][5]>>8) + ((uint64_t)in32[3][6]<<21) + ((uint64_t)in32[3][7]<<50));
    Out->t.im[3] = mask & (((uint64_t)in32[3][14]>>8) + ((uint64_t)in32[3][15]<<21) + ((uint64_t)in32[3][16]<<50));

    Out->x.re[4] = mask & (((uint64_t)in32[0][7]>>1) + ((uint64_t)in32[0][8]<<28));
    Out->x.im[4] = mask & (((uint64_t)in32[0][16]>>1) + ((uint64_t)in32[0][17]<<28));
    Out->y.re[4] = mask & (((uint64_t)in32[1][7]>>1) + ((uint64_t)in32[1][8]<<28));
    Out->y.im[4] = mask & (((uint64_t)in32[1][16]>>1) + ((uint64_t)in32[1][17]<<28));
    Out->z.re[4] = mask & (((uint64_t)in32[2][7]>>1) + ((uint64_t)in32[2][8]<<28));
    Out->z.im[4] = mask & (((uint64_t)in32[2][16]>>1) + ((uint64_t)in32[2][17]<<28));
    Out->t.re[4] = mask & (((uint64_t)in32[3][7]>>1) + ((uint64_t)in32[3][8]<<28));
    Out->t.im[4] = mask & (((uint64_t)in32[3][16]>>1) + ((uint64_t)in32[3][17]<<28));
}

static inline void 
transpose_matrix(uint32x4_t (*Out)[18], const basis_change_matrix_t* In){
    theta_point_t tp;

    tp.x = In->m[0][0];
    tp.y = In->m[1][0];
    tp.z = In->m[2][0];
    tp.t = In->m[3][0];
    transpose(Out[0], tp);

    tp.x = In->m[0][1];
    tp.y = In->m[1][1];
    tp.z = In->m[2][1];
    tp.t = In->m[3][1];
    transpose(Out[1], tp);

    tp.x = In->m[0][2];
    tp.y = In->m[1][2];
    tp.z = In->m[2][2];
    tp.t = In->m[3][2];
    transpose(Out[2], tp);

    tp.x = In->m[0][3];
    tp.y = In->m[1][3];
    tp.z = In->m[2][3];
    tp.t = In->m[3][3];
    transpose(Out[3], tp);
}

// end of hd_functions
/**
 * @}
 */
// end of hd_module
/**
 * @}
 */
#endif
