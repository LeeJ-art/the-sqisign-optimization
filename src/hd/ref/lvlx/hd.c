#include <hd.h>
#include <mp.h>
#include <assert.h>
#include <stdio.h>
#include <bench.h>
#include <arm_neon.h>


static __inline__ uint64_t rdtsc(void)
{
    return (uint64_t)cpucycles();
}


void
double_couple_point(theta_couple_point_t *out, const theta_couple_point_t *in, const theta_couple_curve_t *E1E2)
{
    ec_dbl(&out->P1, &in->P1, &E1E2->E1);
    ec_dbl(&out->P2, &in->P2, &E1E2->E2);
}

void
double_couple_point_iter(theta_couple_point_t *out,
                         unsigned n,
                         const theta_couple_point_t *in,
                         const theta_couple_curve_t *E1E2)
{
    if (n == 0) {
        memmove(out, in, sizeof(theta_couple_point_t));
    } else {
        double_couple_point(out, in, E1E2);
        for (unsigned i = 0; i < n - 1; i++) {
            double_couple_point(out, out, E1E2);
        }
    }
}

void
add_couple_jac_points(theta_couple_jac_point_t *out,
                      const theta_couple_jac_point_t *T1,
                      const theta_couple_jac_point_t *T2,
                      const theta_couple_curve_t *E1E2)
{
    ADD(&out->P1, &T1->P1, &T2->P1, &E1E2->E1);
    ADD(&out->P2, &T1->P2, &T2->P2, &E1E2->E2);
}

void
double_couple_jac_point(theta_couple_jac_point_t *out,
                        const theta_couple_jac_point_t *in,
                        const theta_couple_curve_t *E1E2)
{
    DBL(&out->P1, &in->P1, &E1E2->E1);
    DBL(&out->P2, &in->P2, &E1E2->E2);
}

void
double_couple_jac_point_iter(theta_couple_jac_point_t *out,
                             unsigned n,
                             const theta_couple_jac_point_t *in,
                             const theta_couple_curve_t *E1E2)
{
    //uint64_t time;
    if (n == 0) {
        *out = *in;
    } else if (n == 1) {
        double_couple_jac_point(out, in, E1E2);
    } else {
        fp2_t a1, a2, t1, t2;

        //time = rdtsc();
        jac_to_ws(&out->P1, &t1, &a1, &in->P1, &E1E2->E1);
        jac_to_ws(&out->P2, &t2, &a2, &in->P2, &E1E2->E2);
        //printf("- jac_to_ws: %lu\n", rdtsc()-time);

        //time = rdtsc();
        DBLW(&out->P1, &t1, &out->P1, &t1);
        DBLW(&out->P2, &t2, &out->P2, &t2);
        for (unsigned i = 0; i < n - 1; i++) {
            DBLW(&out->P1, &t1, &out->P1, &t1);
            DBLW(&out->P2, &t2, &out->P2, &t2);
        }
        //printf("- DBLW: %lu\n", rdtsc()-time);

        //time = rdtsc();
        jac_from_ws(&out->P1, &out->P1, &a1, &E1E2->E1);
        jac_from_ws(&out->P2, &out->P2, &a2, &E1E2->E2);
        //printf("- jac_from_ws: %lu\n", rdtsc()-time);
    }
}


void
jac_from_ws_vec(uint32x4_t *out, const fp2_t *ao31, const fp2_t *ao32, const ec_curve_t *curve1, const ec_curve_t *curve2)
{
    // Cost of 1M + 1S when A != 0.
    /* X = U - (A*W^2)/3, Y = V, Z = W. */
    theta_point_t tp;
    uint32x4_t a32[18], b32[18];
    tp.x = ao31[0];
    tp.z = ao32[0];
    transpose(b32, tp);
    if (!fp2_is_zero(&(curve1->A))) {
        // fp2_sqr(&t, &P->z);
        // fp2_mul(&t, &t, ao3);
        // fp2_sub(&Q->x, &P->x, &t);
        for (int i=0; i<18; i++){
            a32[i][0] = out[18+i][1];
        }
        fp2_sqr_batched(a32, a32);
        fp2_mul_batched(a32, a32, b32);
        fp2_sub_batched(a32, out, a32);
        for (int i=0; i<18; i++){
            out[i][0] = a32[i][0];
        }
    }

    if (!fp2_is_zero(&(curve2->A))) {
        // fp2_sqr(&t, &P->z);
        // fp2_mul(&t, &t, ao3);
        // fp2_sub(&Q->x, &P->x, &t);
        for (int i=0; i<18; i++){
            a32[i][2] = out[18+i][3];
        }
        fp2_sqr_batched(a32, a32);
        fp2_mul_batched(a32, a32, b32);
        fp2_sub_batched(a32, out, a32);
        for (int i=0; i<18; i++){
            out[i][2] = a32[i][2];
        }
    }
}

void
double_couple_jac_point_iter_vec(theta_couple_jac_point_t *out,
                             unsigned n,
                             const theta_couple_jac_point_t *in,
                             const theta_couple_curve_t *E1E2)
{
    //uint64_t time;
    //printf("n: %u\n", n);
    if (n == 0) {
        *out = *in;
    } else if (n == 1) {
        double_couple_jac_point(out, in, E1E2);
    } else {
        fp2_t a1, a2, t1, t2;
        jac_to_ws(&out->P1, &t1, &a1, &in->P1, &E1E2->E1);
        jac_to_ws(&out->P2, &t2, &a2, &in->P2, &E1E2->E2);
        
        fp_t mb  = {0, 0, 0, 0, 35184372088832}; // 2**(510-261)
        fp_t mf = {1638, 0, 0, 0, 35184372088832}; //2**(261)
        uint32x4_t out32[36], reCarry, imCarry;
        theta_point_t tp;
        tp.x = out->P1.x;
        tp.y = out->P1.y;
        tp.z = out->P2.x;
        tp.t = out->P2.y;
        theta_montback(&tp, &mf);
        transpose(out32, tp);
        prop_2(out32);
        prop_2(out32+9);
        reCarry = div5(out32+8), imCarry = div5(out32+17);
        out32[0] = vaddq_u32(out32[0], reCarry);
        out32[9] = vaddq_u32(out32[9], imCarry);

        tp.x = t1;
        tp.y = out->P1.z;
        tp.z = t2;
        tp.t = out->P2.z;
        theta_montback(&tp, &mf);
        transpose(out32+18, tp);
        prop_2(out32+18);
        prop_2(out32+27);
        reCarry = div5(out32+26), imCarry = div5(out32+35);
        out32[18] = vaddq_u32(out32[18], reCarry);
        out32[27] = vaddq_u32(out32[27], imCarry);
        DBLW_vec(out32);
        for (unsigned i = 0; i < n - 1; i++) {
            DBLW_vec(out32);
        }
        itranspose(&tp, out32);
        theta_montback(&tp, &mb);
        out->P1.x = tp.x;
        out->P1.y = tp.y;
        out->P2.x = tp.z;
        out->P2.y = tp.t;
        itranspose(&tp, out32+18);
        theta_montback(&tp, &mb);
        t1 = tp.x;
        out->P1.z = tp.y;
        t2 = tp.z;
        out->P2.z = tp.t;

        //without montback => R2
        jac_from_ws(&out->P1, &out->P1, &a1, &E1E2->E1);
        jac_from_ws(&out->P2, &out->P2, &a2, &E1E2->E2);
        // jac_from_ws_vec(out32, &a1, &a2, &E1E2->E1, &E1E2->E2);
        // itranspose(&tp, out32);
        // out->P1.x = tp.x;
        // out->P1.y = tp.y;
        // out->P2.x = tp.z;
        // out->P2.y = tp.t;
        // itranspose(&tp, out32+18);
        // out->P1.z = tp.y;
        // out->P2.z = tp.t;
    }
}

void
couple_jac_to_xz(theta_couple_point_t *P, const theta_couple_jac_point_t *xyP)
{
    jac_to_xz(&P->P1, &xyP->P1);
    jac_to_xz(&P->P2, &xyP->P2);
}

void
copy_bases_to_kernel(theta_kernel_couple_points_t *ker, const ec_basis_t *B1, const ec_basis_t *B2)
{
    // Copy the basis on E1 to (P, _) on T1, T2 and T1 - T2
    copy_point(&ker->T1.P1, &B1->P);
    copy_point(&ker->T2.P1, &B1->Q);
    copy_point(&ker->T1m2.P1, &B1->PmQ);

    // Copy the basis on E2 to (_, P) on T1, T2 and T1 - T2
    copy_point(&ker->T1.P2, &B2->P);
    copy_point(&ker->T2.P2, &B2->Q);
    copy_point(&ker->T1m2.P2, &B2->PmQ);
}

int xDBLMUL_vec(ec_point_t *S,
        const ec_point_t *P,
        const digit_t *k,
        const ec_point_t *Q,
        const digit_t *l,
        const ec_point_t *PQ,
        const int kbits,
        const ec_curve_t *curve)                    
{
    // The Montgomery biladder
    // Input:  projective Montgomery points P=(XP:ZP) and Q=(XQ:ZQ) such that xP=XP/ZP and xQ=XQ/ZQ, scalars k and l of
    //         bitlength kbits, the difference PQ=P-Q=(XPQ:ZPQ), and the Montgomery curve constants (A:C).
    // Output: projective Montgomery point S <- k*P + l*Q = (XS:ZS) such that x(k*P + l*Q)=XS/ZS.
    int i, A_is_zero;
    digit_t evens, mevens, bitk0, bitl0, maskk, maskl, temp, bs1_ip1, bs2_ip1, bs1_i, bs2_i, h;
    digit_t sigma[2] = { 0 }, pre_sigma = 0;
    digit_t k_t[NWORDS_ORDER], l_t[NWORDS_ORDER], one[NWORDS_ORDER] = { 0 }, r[2 * BITS] = { 0 };
    ec_point_t R[3] = {0};

    // differential additions formulas are invalid in this case
    if (ec_has_zero_coordinate(P) | ec_has_zero_coordinate(Q) | ec_has_zero_coordinate(PQ))
        return 0;

    // Derive sigma according to parity
    bitk0 = (k[0] & 1);
    bitl0 = (l[0] & 1);
    maskk = 0 - bitk0; // Parity masks: 0 if even, otherwise 1...1
    maskl = 0 - bitl0;
    sigma[0] = (bitk0 ^ 1);
    sigma[1] = (bitl0 ^ 1);
    evens = sigma[0] + sigma[1]; // Count number of even scalars
    mevens = 0 - (evens & 1);    // Mask mevens <- 0 if # even of scalars = 0 or 2, otherwise mevens = 1...1

    // If k and l are both even or both odd, pick sigma = (0,1)
    sigma[0] = (sigma[0] & mevens);
    sigma[1] = (sigma[1] & mevens) | (1 & ~mevens);

    // Convert even scalars to odd
    one[0] = 1;
    mp_sub(k_t, k, one, NWORDS_ORDER);
    mp_sub(l_t, l, one, NWORDS_ORDER);

    select_ct(k_t, k_t, k, maskk, NWORDS_ORDER);
    select_ct(l_t, l_t, l, maskl, NWORDS_ORDER);


    // Scalar recoding
    for (i = 0; i < kbits; i++) {
        // If sigma[0] = 1 swap k_t and l_t
        maskk = 0 - (sigma[0] ^ pre_sigma);
        swap_ct(k_t, l_t, maskk, NWORDS_ORDER);

        if (i == kbits - 1) {
            bs1_ip1 = 0;
            bs2_ip1 = 0;
        } else {
            bs1_ip1 = mp_shiftr(k_t, 1, NWORDS_ORDER);
            bs2_ip1 = mp_shiftr(l_t, 1, NWORDS_ORDER);
        }
        bs1_i = k_t[0] & 1;
        bs2_i = l_t[0] & 1;

        r[2 * i] = bs1_i ^ bs1_ip1;
        r[2 * i + 1] = bs2_i ^ bs2_ip1;

        // Revert sigma if second bit, r_(2i+1), is 1
        pre_sigma = sigma[0];
        maskk = 0 - r[2 * i + 1];
        select_ct(&temp, &sigma[0], &sigma[1], maskk, 1);
        select_ct(&sigma[1], &sigma[1], &sigma[0], maskk, 1);
        sigma[0] = temp;
    }

    // Point initialization
    ec_point_init(&R[0]);
    maskk = 0 - sigma[0];
    select_point(&R[1], P, Q, maskk);
    select_point(&R[2], Q, P, maskk);

    theta_point_t tp;
    fp_t ExMod = {1638, 0, 0, 0, 35184372088832};
    uint32x4_t In32[18], DIFF1a32[9], DIFF1b32[9], DIFF2a32[9], DIFF2b32[9];
    tp.x = R[1].x;
    tp.y = R[1].z;
    tp.z = R[2].x;
    tp.t = R[2].z;
    theta_montback(&tp, &ExMod);
    transpose(In32, tp);
    for (i=0; i<9; i++){
        DIFF1a32[i][0] = In32[i][0];
        DIFF1a32[i][1] = In32[i+9][0];
        DIFF1a32[i][2] = In32[i][1];
        DIFF1a32[i][3] = In32[i+9][1];

        DIFF1b32[i][0] = In32[i][2];
        DIFF1b32[i][1] = In32[i+9][2];
        DIFF1b32[i][2] = In32[i][3];
        DIFF1b32[i][3] = In32[i+9][3];
    }

    // Initialize DIFF2a <- P+Q, DIFF2b <- P-Q
    xADD(&R[2], &R[1], &R[2], PQ);

    if (ec_has_zero_coordinate(&R[2]))
        return 0; // non valid formulas

    tp.x = R[2].x;
    tp.y = R[2].z;
    tp.z = PQ->x;
    tp.t = PQ->z;
    theta_montback(&tp, &ExMod);
    transpose(In32, tp);
    for (i=0; i<9; i++){
        DIFF2a32[i][0] = In32[i][0];
        DIFF2a32[i][1] = In32[i+9][0];
        DIFF2a32[i][2] = In32[i][1];
        DIFF2a32[i][3] = In32[i+9][1];

        DIFF2b32[i][0] = In32[i][2];
        DIFF2b32[i][1] = In32[i+9][2];
        DIFF2b32[i][2] = In32[i][3];
        DIFF2b32[i][3] = In32[i+9][3];
    }

    A_is_zero = fp2_is_zero(&curve->A);

    uint32x4_t R01[18], R12[18], A24[18], Arith1[18], Arith2[18];
    tp.x = R[0].x;
    tp.y = R[0].z;
    tp.z = R[1].x;
    tp.t = R[1].z;
    theta_montback(&tp, &ExMod);
    transpose(In32, tp);
    for (i=0; i<9; i++){
        R01[i][0] = In32[i][0];
        R01[i][1] = In32[i+9][0];
        R01[i][2] = In32[i][1];
        R01[i][3] = In32[i+9][1];

        R01[i+9][0] = In32[i][2];
        R01[i+9][1] = In32[i+9][2];
        R01[i+9][2] = In32[i][3];
        R01[i+9][3] = In32[i+9][3];
    }

    tp.x = R[1].x;
    tp.y = R[1].z;
    tp.z = R[2].x;
    tp.t = R[2].z;
    theta_montback(&tp, &ExMod);
    transpose(In32, tp);
    for (i=0; i<9; i++){
        R12[i][0] = In32[i][0];
        R12[i][1] = In32[i+9][0];
        R12[i][2] = In32[i][1];
        R12[i][3] = In32[i+9][1];

        R12[i+9][0] = In32[i][2];
        R12[i+9][1] = In32[i+9][2];
        R12[i+9][2] = In32[i][3];
        R12[i+9][3] = In32[i+9][3];
    }

    for (int i=0; i<5; i++){
        tp.x.re[i] = curve->A24.x.re[i];
        tp.y.re[i] = curve->A24.x.im[i];
        tp.z.re[i] = curve->A24.z.re[i];
        tp.t.re[i] = curve->A24.z.im[i];
    }
    theta_montback(&tp, &ExMod);
    transpose(A24, tp);
    // Main loop
    fp_t mb = {0, 0, 0, 0, 35184372088832};
    for (i = kbits - 1; i >= 0; i--) {
        h = r[2 * i] + r[2 * i + 1]; // in {0, 1, 2}
        maskk = 0 - (h & 1);
        fp2_select_vec(Arith1, R01, vdupq_n_u32(maskk));

        maskk = 0 - (h >> 1);
        for (int z=0; z<9; z++){
            Arith1[z+9] = R12[z+9];
        }
        fp2_select_vec(Arith1, Arith1, vdupq_n_u32(maskk));

        if (A_is_zero) {
            xDBL_E0_vec(Arith1, Arith1);
        } else {
            //assert(fp2_is_one(&curve->A24.z));
            xDBL_A24_vec(Arith1, Arith1, A24, true);
        }

        maskk = 0 - r[2 * i + 1]; // in {0, 1}
        fp2_select_vec(Arith1+9, R01, vdupq_n_u32(maskk));
        fp2_select_vec(Arith2, R12, vdupq_n_u32(maskk));

        cswap_points_vec(DIFF1a32, DIFF1b32, maskk);
        
        //Before: Arith1 <- T0T1; Arith2 <- T2;
        for (int z=0; z<9; z++){
            R12[z] = R01[z];
            //R0 update
            R01[z] = Arith1[z];
            Arith1[z] = Arith2[z];
        }
        xADD2_vec(Arith1+9, Arith1, DIFF1a32, Arith2, R12, DIFF2a32);

        // If hw (mod 2) = 1 then swap DIFF2a and DIFF2b
        maskk = 0 - (h & 1);
        cswap_points_vec(DIFF2a32, DIFF2b32, maskk);

        // R <- T
        for (int z=0; z<9; z++){
            //R1, R2 update
            R01[z+9] =  R12[z] = Arith1[z+9];
            R12[z+9] = Arith2[z];
        }
    }

    itranspose(&tp, R01);
    theta_montback(&tp, &mb);
    for (int z=0; z<5; z++){
        R[0].x.re[z] = tp.x.re[z];
        R[0].x.im[z] = tp.y.re[z];
        R[0].z.re[z] = tp.z.re[z];
        R[0].z.im[z] = tp.t.re[z];

        R[1].x.re[z] = tp.x.im[z];
        R[1].x.im[z] = tp.y.im[z];
        R[1].z.re[z] = tp.z.im[z];
        R[1].z.im[z] = tp.t.im[z];
    }

    itranspose(&tp, R12);
    theta_montback(&tp, &mb);
    for (int z=0; z<5; z++){
        R[2].x.re[z] = tp.x.im[z];
        R[2].x.im[z] = tp.y.im[z];
        R[2].z.re[z] = tp.z.im[z];
        R[2].z.im[z] = tp.t.im[z];
    }

    // Output R[evens]
    select_point(S, &R[0], &R[1], mevens);
    maskk = 0 - (bitk0 & bitl0);
    select_point(S, S, &R[2], maskk);
    return 1;
}
