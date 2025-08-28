#include <fp.h>
#include <arm_neon.h>

/*
 * If ctl == 0x00000000, then *d is set to a0
 * If ctl == 0xFFFFFFFF, then *d is set to a1
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 */
void
fp_select(fp_t *d, const fp_t *a0, const fp_t *a1, uint32_t ctl)
{
    digit_t cw = (int32_t)ctl;
    for (unsigned int i = 0; i < NWORDS_FIELD; i++) {
        (*d)[i] = (*a0)[i] ^ (cw & ((*a0)[i] ^ (*a1)[i]));
    }
}


void prop_2(uint32x4_t *n) {
    uint32x4_t mask = vdupq_n_u32(((uint32_t)1 << 29) - 1);
    uint64x2_t mask64 = vdupq_n_u64(((uint32_t)1 << 29) - 1);
    uint64x2_t carry[2] = {0};

    for(int i = 0;i<4;i++) ((uint32_t*)carry)[2*i] = n[0][i];
    carry[0] = vshrq_n_u64(carry[0], 29);
    carry[1] = vshrq_n_u64(carry[1], 29);
    n[0] = vandq_u32(n[0], mask);
    for (int i = 1; i < 8; i++) {
        carry[0] = vaddw_u32(carry[0], ((uint32x2_t*)n)[2*i]);
        carry[1] = vaddw_u32(carry[1], ((uint32x2_t*)n)[2*i+1]);
        ((uint32x2_t*)n)[2*i] = vmovn_u64(vandq_u64(carry[0], mask64));
        ((uint32x2_t*)n)[2*i+1] = vmovn_u64(vandq_u64(carry[1], mask64));
        carry[0] = vshrq_n_u64(carry[0], 29);
        carry[1] = vshrq_n_u64(carry[1], 29);
    }
    ((uint32x2_t*)n)[16] = vadd_u32(((uint32x2_t*)n)[16], vmovn_u64(carry[0]));
    ((uint32x2_t*)n)[17] = vadd_u32(((uint32x2_t*)n)[17], vmovn_u64(carry[1]));
}

// in: a, out: a/5
uint32x4_t div5(uint32x4_t* in){
    uint16x4_t t1 = vmovn_u32(vshrq_n_u32(in[0], 16));
    uint16x4_t t2 = (uint16x4_t)vqdmulh_n_s16((int16x4_t)t1, 6554);
    uint32x4_t t3 = vmovl_u16(vmls_n_u16(t1, t2, 5));
    in[0] = vaddq_u32( vshlq_n_u32(t3, 16), vandq_u32(in[0], vdupq_n_u32((1<<16)-1)));
    return vmovl_u16(t2);
}

void fp_mul_batched(uint32x2_t *out, uint32x4_t *a, uint32x4_t *b){
    uint64x2_t mask = vdupq_n_u64(((uint64_t)1<<29)-1);
    uint64x2_t t[2] = {0};
    uint32x2_t mod = vdup_n_u32(0x50000);
    uint32x2_t tmp[18];

    
    // x^0
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[0], b[0]);
    tmp[0] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[1] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^1
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[0], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[1], b[0]);
    tmp[2] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[3] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^2
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[0], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[1], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[2], b[0]);
    tmp[4] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[5] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^3
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[0], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[1], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[2], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[3], b[0]);
    tmp[6] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[7] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^4
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[0], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[1], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[2], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[3], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[4], b[0]);
    tmp[8] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[9] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^5
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[0], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[1], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[2], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[3], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[4], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[5], b[0]);
    tmp[10] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[11] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^6
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[0], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[1], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[2], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[3], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[4], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[5], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[6], b[0]);
    tmp[12] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[13] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^7
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[0], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[1], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[2], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[3], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[4], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[5], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[6], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[7], b[0]);
    tmp[14] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[15] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^8
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[0], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[0], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[1], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[2], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[3], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[4], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[5], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[6], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[7], b[1]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[0]);
    t[1] = vmlal_high_u32(t[1], a[8], b[0]);
    t[0] = vmlal_u32(t[0], tmp[0], mod);
    t[1] = vmlal_u32(t[1], tmp[1], mod);
    tmp[16] = vmovn_u64( vandq_u64(t[0], mask));
    tmp[17] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);



// -------- upper 9 sets ---------

    // x^9
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[2], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[1], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[2], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[3], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[4], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[5], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[6], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[7], b[2]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[2]);
    t[1] = vmlal_high_u32(t[1], a[8], b[1]);
    t[0] = vmlal_u32(t[0], tmp[2], mod);
    t[1] = vmlal_u32(t[1], tmp[3], mod);
    out[0] = vmovn_u64( vandq_u64(t[0], mask));
    out[1] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^10
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[4], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[2], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[3], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[4], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[5], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[6], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[7], b[3]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[4]);
    t[1] = vmlal_high_u32(t[1], a[8], b[2]);
    t[0] = vmlal_u32(t[0], tmp[4], mod);
    t[1] = vmlal_u32(t[1], tmp[5], mod);
    out[2] = vmovn_u64( vandq_u64(t[0], mask));
    out[3] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^11
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[6], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[3], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[4], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[5], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[6], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[7], b[4]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[6]);
    t[1] = vmlal_high_u32(t[1], a[8], b[3]);
    t[0] = vmlal_u32(t[0], tmp[6], mod);
    t[1] = vmlal_u32(t[1], tmp[7], mod);
    out[4] = vmovn_u64( vandq_u64(t[0], mask));
    out[5] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^12
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[8], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[4], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[5], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[6], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[7], b[5]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[8]);
    t[1] = vmlal_high_u32(t[1], a[8], b[4]);
    t[0] = vmlal_u32(t[0], tmp[8], mod);
    t[1] = vmlal_u32(t[1], tmp[9], mod);
    out[6] = vmovn_u64( vandq_u64(t[0], mask));
    out[7] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^13
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[10], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[5], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[6], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[7], b[6]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[10]);
    t[1] = vmlal_high_u32(t[1], a[8], b[5]);
    t[0] = vmlal_u32(t[0], tmp[10], mod);
    t[1] = vmlal_u32(t[1], tmp[11], mod);
    out[8] = vmovn_u64( vandq_u64(t[0], mask));
    out[9] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^14
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[12], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[6], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[7], b[7]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[12]);
    t[1] = vmlal_high_u32(t[1], a[8], b[6]);
    t[0] = vmlal_u32(t[0], tmp[12], mod);
    t[1] = vmlal_u32(t[1], tmp[13], mod);
    out[10] = vmovn_u64( vandq_u64(t[0], mask));
    out[11] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^15
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[14], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[7], b[8]);
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[14]);
    t[1] = vmlal_high_u32(t[1], a[8], b[7]);
    t[0] = vmlal_u32(t[0], tmp[14], mod);
    t[1] = vmlal_u32(t[1], tmp[15], mod);
    out[12] = vmovn_u64( vandq_u64(t[0], mask));
    out[13] = vmovn_u64( vandq_u64(t[1], mask));
    t[0] = vshrq_n_u64(t[0], 29);
    t[1] = vshrq_n_u64(t[1], 29);


    // x^16
    t[0] = vmlal_u32(t[0], ((uint32x2_t*)a)[16], ((uint32x2_t*)b)[16]);
    t[1] = vmlal_high_u32(t[1], a[8], b[8]);
    t[0] = vmlal_u32(t[0], tmp[16], mod);
    t[1] = vmlal_u32(t[1], tmp[17], mod);
    out[14] = vmovn_u64( vandq_u64(t[0], mask));
    out[15] = vmovn_u64( vandq_u64(t[1], mask));

    // x^17
    out[16] = vmovn_u64(vshrq_n_u64(t[0], 29));
    out[17] = vmovn_u64(vshrq_n_u64(t[1], 29));
}

