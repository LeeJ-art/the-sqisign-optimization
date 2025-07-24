#include "theta_structure.h"
#include <assert.h>
#include <arm_neon.h>

void
theta_precomputation(theta_structure_t *A)
{

    if (A->precomputation) {
        return;
    }

    theta_point_t A_dual;
    to_squared_theta(&A_dual, &A->null_point);

    fp2_t t1, t2;
    fp2_mul(&t1, &A_dual.x, &A_dual.y);
    fp2_mul(&t2, &A_dual.z, &A_dual.t);
    fp2_mul(&A->XYZ0, &t1, &A_dual.z);
    fp2_mul(&A->XYT0, &t1, &A_dual.t);
    fp2_mul(&A->YZT0, &t2, &A_dual.y);
    fp2_mul(&A->XZT0, &t2, &A_dual.x);

    fp2_mul(&t1, &A->null_point.x, &A->null_point.y);
    fp2_mul(&t2, &A->null_point.z, &A->null_point.t);
    fp2_mul(&A->xyz0, &t1, &A->null_point.z);
    fp2_mul(&A->xyt0, &t1, &A->null_point.t);
    fp2_mul(&A->yzt0, &t2, &A->null_point.y);
    fp2_mul(&A->xzt0, &t2, &A->null_point.x);

    A->precomputation = true;
}

void structure2point(theta_point_t* out, theta_structure_t *A){
    out[0].x = A->XYZ0;
    out[0].y = A->YZT0;
    out[0].z = A->XZT0;
    out[0].t = A->XYT0;
    out[1].x = A->xyz0;
    out[1].y = A->yzt0;
    out[1].z = A->xzt0;
    out[1].t = A->xyt0;
}

void structure2point_reindex(theta_point_t* out, theta_structure_t *A){
    out[0].x = A->YZT0;
    out[0].y = A->XZT0;
    out[0].z = A->XYT0;
    out[0].t = A->XYZ0;
    out[1].x = A->yzt0;
    out[1].y = A->xzt0;
    out[1].z = A->xyt0;
    out[1].t = A->xyz0;
}

void point2structure(theta_structure_t* A, theta_point_t *out){
    A->YZT0 = out[0].x;
    A->XZT0 = out[0].y;
    A->XYT0 = out[0].z;
    A->XYZ0 = out[0].t;

    A->yzt0 = out[1].x;
    A->xzt0 = out[1].y;
    A->xyt0 = out[1].z;
    A->xyz0 = out[1].t;
}

// a0: 5x A_null, a1: 2x A_null
bool theta_precomputation_vec(uint32x4_t* a0, uint32x4_t* a1, uint32x4_t* A_null)
{
    // theta_point_t A_dual;
    uint32x4_t A_dual[18];
    // to_squared_theta(&A_dual, &A->null_point);
    to_squared_theta_batched(A_dual, A_null); // dual: 1x

    // fp2_t t1, t2;
    uint32x4_t t1[18], t2[18], t3[18];
    // reindex
    for(int i = 0;i<18;i++){
        t1[i][0] = A_dual[i][0];
        t1[i][1] = A_dual[i][2];
        t1[i][2] = A_null[i][0];
        t1[i][3] = A_null[i][2];
        t2[i][0] = A_dual[i][1];
        t2[i][1] = A_dual[i][3];
        t2[i][2] = A_null[i][1];
        t2[i][3] = A_null[i][3];
    }
    fp2_mul_batched(t3, t1, t2); // t3 = XY ZT xy zt
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][1];
        t1[i][1] = t3[i][1];
        t1[i][2] = t3[i][0];
        t1[i][3] = t3[i][0];
        t2[i][0] = A_dual[i][1];
        t2[i][1] = A_dual[i][0];
        t2[i][2] = A_dual[i][3];
        t2[i][3] = A_dual[i][2];
    }
    fp2_mul_batched(a0, t1, t2);
    
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][3];
        t1[i][1] = t3[i][3];
        t1[i][2] = t3[i][2];
        t1[i][3] = t3[i][2];
        t2[i][0] = A_null[i][1];
        t2[i][1] = A_null[i][0];
        t2[i][2] = A_null[i][3];
        t2[i][3] = A_null[i][2];
    }
    fp2_mul_batched(a1, t1, t2);
    return true;
}

void transpose_theta_precomputation_vec(theta_structure_t *A)
{

    if (A->precomputation) {
        return;
    }

    // theta_point_t A_dual;
    uint32x4_t A_null[18], A_dual[18], a0[18], a1[18];
    theta_point_t B[2];
    transpose(A_null, A->null_point);
    //reduce
    prop_2(A_null);
    prop_2(A_null+9);
    uint32x4_t reCarry = div5(A_null+8), imCarry = div5(A_null+17);
    A_null[0] = vaddq_u32(A_null[0], reCarry);
    A_null[9] = vaddq_u32(A_null[9], imCarry);
    prop_2(A_null);
    prop_2(A_null+9);
    reduce_q(A_null);

    // to_squared_theta(&A_dual, &A->null_point);
    to_squared_theta_batched(A_dual, A_null);

    /*
    */
    // fp2_t t1, t2;
    uint32x4_t t1[18], t2[18], t3[18];
    // reindex
    for(int i = 0;i<18;i++){
        t1[i][0] = A_dual[i][0];
        t1[i][1] = A_dual[i][2];
        t1[i][2] = A_null[i][0];
        t1[i][3] = A_null[i][2];
        t2[i][0] = A_dual[i][1];
        t2[i][1] = A_dual[i][3];
        t2[i][2] = A_null[i][1];
        t2[i][3] = A_null[i][3];
    }
    fp2_mul_batched(t3, t1, t2); // t3 = XY ZT xy zt
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][1];
        t1[i][1] = t3[i][1];
        t1[i][2] = t3[i][0];
        t1[i][3] = t3[i][0];
        t2[i][0] = A_dual[i][1];
        t2[i][1] = A_dual[i][0];
        t2[i][2] = A_dual[i][3];
        t2[i][3] = A_dual[i][2];
    }

    // fp2_mul(&t1, &A_dual.x, &A_dual.y);
    // fp2_mul(&t2, &A_dual.z, &A_dual.t);
    // fp2_mul(&A->YZT0, &t2, &A_dual.y);
    // fp2_mul(&A->XZT0, &t2, &A_dual.x);
    // fp2_mul(&A->XYT0, &t1, &A_dual.t);
    // fp2_mul(&A->XYZ0, &t1, &A_dual.z);
    
    fp2_mul_batched(a0, t1, t2);

    // reduce
    prop_2(a0);
    prop_2(a0+9);
    reCarry = div5(a0+8), imCarry = div5(a0+17);
    a0[0] = vaddq_u32(a0[0], reCarry);
    a0[9] = vaddq_u32(a0[9], imCarry);
    prop_2(a0);
    prop_2(a0+9);

    itranspose(&(B[0]), a0);
    
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][3];
        t1[i][1] = t3[i][3];
        t1[i][2] = t3[i][2];
        t1[i][3] = t3[i][2];
        t2[i][0] = A_null[i][1];
        t2[i][1] = A_null[i][0];
        t2[i][2] = A_null[i][3];
        t2[i][3] = A_null[i][2];
    }

    // fp2_mul(&t1, &A->null_point.x, &A->null_point.y);
    // fp2_mul(&t2, &A->null_point.z, &A->null_point.t);
    // fp2_mul(&A->yzt0, &t2, &A->null_point.y);
    // fp2_mul(&A->xzt0, &t2, &A->null_point.x);
    // fp2_mul(&A->xyt0, &t1, &A->null_point.t);
    // fp2_mul(&A->xyz0, &t1, &A->null_point.z);
    fp2_mul_batched(a1, t1, t2);

    //reduce
    prop_2(a1);
    prop_2(a1+9);
    reCarry = div5(a1+8), imCarry = div5(a1+17);
    a1[0] = vaddq_u32(a1[0], reCarry);
    a1[9] = vaddq_u32(a1[9], imCarry);
    prop_2(a1);
    prop_2(a1+9);

    itranspose(&(B[1]), a1);

    point2structure(A, B);
    
    A->precomputation = true;
}

void
double_point(theta_point_t *out, theta_structure_t *A, const theta_point_t *in)
{
    to_squared_theta(out, in);
    fp2_sqr(&out->x, &out->x);
    fp2_sqr(&out->y, &out->y);
    fp2_sqr(&out->z, &out->z);
    fp2_sqr(&out->t, &out->t);

    if (!A->precomputation) {
        theta_precomputation(A);
    }
    fp2_mul(&out->x, &out->x, &A->YZT0);
    fp2_mul(&out->y, &out->y, &A->XZT0);
    fp2_mul(&out->z, &out->z, &A->XYT0);
    fp2_mul(&out->t, &out->t, &A->XYZ0);

    hadamard(out, out);

    fp2_mul(&out->x, &out->x, &A->yzt0);
    fp2_mul(&out->y, &out->y, &A->xzt0);
    fp2_mul(&out->z, &out->z, &A->xyt0);
    fp2_mul(&out->t, &out->t, &A->xyz0);
}

// (12+4in)x
void double_point_vec(theta_point_t *out, theta_structure_t *A, const theta_point_t *in)
{
    uint32x4_t out_transpose[18], q[9];
    transpose(out_transpose, in[0]);

    // reduce
    prop_2(out_transpose);
    prop_2(out_transpose+9);

    uint32x4_t reCarry = div5(out_transpose+8), imCarry = div5(out_transpose+17);

    out_transpose[0] = vaddq_u32(out_transpose[0], reCarry);
    out_transpose[9] = vaddq_u32(out_transpose[9], imCarry);

    prop_2(out_transpose);
    prop_2(out_transpose+9);
    
    // to_squared_theta(out, in);
    // 1x + 2xin
    // (in+in+1)x
    to_squared_theta_batched(out_transpose, out_transpose);

    // reduce
    prop_2(out_transpose);
    prop_2(out_transpose+9);

    reCarry = div5(out_transpose+8), imCarry = div5(out_transpose+17);

    out_transpose[0] = vaddq_u32(out_transpose[0], reCarry);
    out_transpose[9] = vaddq_u32(out_transpose[9], imCarry);

    prop_2(out_transpose);
    prop_2(out_transpose+9);

    // 3x + 4xin
    // ((in+in+1) + (in+in+1) + 1)x
    fp2_sqr_batched(out_transpose, out_transpose);

    // // transpose A
    uint32x4_t A_null[18], a0[18], a1[18];
    theta_point_t A_theta[2];
    transpose(A_null, A->null_point);
    structure2point_reindex(A_theta, A);
    transpose(a0, A_theta[0]);
    transpose(a1, A_theta[1]);

    // Here keep A's points in a0, a1 with mul-friendly sequence{YZT, XZT, XYT, XYZ}
    if (!A->precomputation) {
        // theta_precomputation(A);
        A->precomputation = theta_precomputation_vec(a0, a1, A_null); // a0: 5x, a1:2x
        itranspose(A_theta, a0);
        itranspose(A_theta+1, a1);
        point2structure(A, A_theta);
    }

    // fp2_mul(&out->x, &out->x, &A->YZT0);
    // fp2_mul(&out->y, &out->y, &A->XZT0);
    // fp2_mul(&out->z, &out->z, &A->XYT0);
    // fp2_mul(&out->t, &out->t, &A->XYZ0);
    // out: 9x + 4xin >1: 4x + 4xin
    // (((in+in+1) + (in+in+1) + 1) + 5 + 1)x  
    fp2_mul_batched(out_transpose, out_transpose, a0);

    // reduce
    prop_2(out_transpose);
    prop_2(out_transpose+9);

    reCarry = div5(out_transpose+8);
    imCarry = div5(out_transpose+17);

    out_transpose[0] = vaddq_u32(out_transpose[0], reCarry);
    out_transpose[9] = vaddq_u32(out_transpose[9], imCarry);

    prop_2(out_transpose);
    prop_2(out_transpose+9);

    //hadamard(out, out);
    for(int i = 0;i<8;i++) q[i] = vdupq_n_u32(0x1fffffff);
    q[8] = vdupq_n_u32(0x4ffff);
    uint32_t q2[9] = {1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 655358};

    for(int i = 0;i<18;i++){
      a0[0][0] = out_transpose[i][0] + out_transpose[i][1];
      a0[0][1] = (out_transpose[i][0] + q[i%9][0]) - out_transpose[i][1];
      a0[0][2] = out_transpose[i][2] + out_transpose[i][3];
      a0[0][3] = (out_transpose[i][2] + q[i%9][0]) - out_transpose[i][3];

      out_transpose[i][0] = a0[0][0] + a0[0][2];
      out_transpose[i][1] = a0[0][1] + a0[0][3];
      out_transpose[i][2] = a0[0][0] + (q2[i%9] - a0[0][2]);
      out_transpose[i][3] = a0[0][1] + (q2[i%9] - a0[0][3]);
    }

    // reduce
    prop_2(out_transpose);
    prop_2(out_transpose+9);

    reCarry = div5(out_transpose+8);
    imCarry = div5(out_transpose+17);

    out_transpose[0] = vaddq_u32(out_transpose[0], reCarry);
    out_transpose[9] = vaddq_u32(out_transpose[9], imCarry);

    prop_2(out_transpose);
    prop_2(out_transpose+9);

    // fp2_mul(&out->x, &out->x, &A->yzt0);
    // fp2_mul(&out->y, &out->y, &A->xzt0);
    // fp2_mul(&out->z, &out->z, &A->xyt0);
    // fp2_mul(&out->t, &out->t, &A->xyz0);        
    // 12x + 4xin
    // ((((in+in+1) + (in+in+1) + 1) + a0 + 1) + a1 + 1)x
    fp2_mul_batched(out_transpose, out_transpose, a1);

    // reduce
    prop_2(out_transpose);
    prop_2(out_transpose+9);

    reCarry = div5(out_transpose+8);
    imCarry = div5(out_transpose+17);

    out_transpose[0] = vaddq_u32(out_transpose[0], reCarry);
    out_transpose[9] = vaddq_u32(out_transpose[9], imCarry);

    prop_2(out_transpose);
    prop_2(out_transpose+9);

    itranspose(out, out_transpose);
}

void
double_iter(theta_point_t *out, theta_structure_t *A, const theta_point_t *in, int exp)
{
    if (exp == 0) {
        *out = *in;
    } else {
        double_point(out, A, in);
        for (int i = 1; i < exp; i++) {
            double_point(out, A, out);
        }
    }
}

void
double_iter_vec(theta_point_t *out, theta_structure_t *A, const theta_point_t *in, int exp)
{
    //fp_t mb12 = {450359962737049, 53687091, 0, 0, 52776558133248};
    //theta_point_t ref;
    //theta_structure_t B;
    //copy_structure(&B, A);
    fp_t mb5 = {27487790694, 0, 0, 0, 35184372088832}; // 2^(261*5-255*4)

    if (exp == 0) {
        *out = *in;
    } else {
        double_point_vec(out, A, in);
        //double_point(out, A, in);
        theta_montback(out, &mb5);
        //choose_small(&ref, out);
        for (int i = 1; i < exp; i++) {
            double_point_vec(out, A, out);
            //double_point(out, A, out);
            theta_montback(out, &mb5);
            //choose_small(&ref, out);
        }
    }
}

uint32_t
is_product_theta_point(const theta_point_t *P)
{
    fp2_t t1, t2;
    fp2_mul(&t1, &P->x, &P->t);
    fp2_mul(&t2, &P->y, &P->z);
    return fp2_is_equal(&t1, &t2);
}
