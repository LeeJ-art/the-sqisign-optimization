#include "theta_structure.h"
#include <assert.h>

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

void theta_precomputation_vec(theta_structure_t *A)
{

    if (A->precomputation) {
        return;
    }

    // theta_point_t A_dual;
    uint32x4_t A_null[18], A_dual[18], a0[18], a1[18];
    theta_point_t B[2];
    transpose(A_null, A->null_point);
    // to_squared_theta(&A_dual, &A->null_point);
    to_squared_theta_batched(A_dual, A_null);

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
    fp2_mul_batched(t3, t1, t2); // t3 = t1 t2 t3 t4
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][0];
        t1[i][1] = t3[i][1];
        t1[i][2] = t3[i][1];
        t1[i][3] = t3[i][0];
        t2[i][0] = A_dual[i][2];
        t2[i][1] = A_dual[i][1];
        t2[i][2] = A_dual[i][0];
        t2[i][3] = A_dual[i][3];
    }
    fp2_mul_batched(a0, t1, t2);
    itranspose(&(B[0]), a0);
    
    for(int i = 0;i<18;i++){
        t1[i][0] = t3[i][2];
        t1[i][1] = t3[i][3];
        t1[i][2] = t3[i][3];
        t1[i][3] = t3[i][2];
        t2[i][0] = A_null[i][2];
        t2[i][1] = A_null[i][1];
        t2[i][2] = A_null[i][0];
        t2[i][3] = A_null[i][3];
    }
    fp2_mul_batched(a1, t1, t2);
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

void double_point_vec(theta_point_t *out, theta_structure_t *A, const theta_point_t *in)
{
    uint32x4_t out_transpose[18], al[18], ah[18], q[9];
    theta_point_t tmp[2];
    transpose(out_transpose, in[0]);
    structure2point(tmp, A);
    transpose(ah, tmp[0]);
    transpose(al, tmp[1]);
    
    // to_squared_theta(out, in);
    to_square_theta_batched(out_transpose); // x1

    fp2_sqr_batched(out_transpose); // x2

    if (!A->precomputation) {
        theta_precomputation(A);
    }

    fp2_mul_batched(out_transpose, out_transpose, ah);

    //hadamard(out, out);
    for(int i = 0;i<8;i++) q[i] = vdupq_n_u32(0x1fffffff);
    q[8] = vdupq_n_u32(0x4ffff);
    uint32_t q2[9] = {1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 1073741822, 655358};

    for(int i = 0;i<18;i++){
      ah[0][0] = out_transpose[i][0] + out_transpose[i][1];
      ah[0][1] = (out_transpose[i][0] + q[i%9][0]) - out_transpose[i][1];
      ah[0][2] = out_transpose[i][2] + out_transpose[i][3];
      ah[0][3] = (out_transpose[i][2] + q[i%9][0]) - out_transpose[i][3];

      out_transpose[i][0] = ah[0][0] + ah[0][2];
      out_transpose[i][1] = ah[0][1] + ah[0][3];
      out_transpose[i][2] = ah[0][0] + (q2[i%9] - ah[0][2]);
      out_transpose[i][3] = ah[0][1] + (q2[i%9] - ah[0][3]);
    }

    prop_2(out_transpose);
    prop_2(out_transpose+9);
    fp2_mul_batched(out_transpose, out_transpose, al);

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
    if (exp == 0) {
        *out = *in;
    } else {
        double_point_vec(out, A, in);
        for (int i = 1; i < exp; i++) {
            double_point_vec(out, A, out);
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
