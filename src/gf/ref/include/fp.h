#ifndef FP_H
#define FP_H

//////////////////////////////////////////////// NOTE: this is placed here for now
#include <sqisign_namespace.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <tutil.h>
#include <fp_constants.h>
#include <arm_neon.h>

#if defined(__APPLE__)
    #define __fp_mul_asm __fp_mul_batched_asm
#else
    #define __fp_mul_asm __fp_mul_shift_batched__asm
#endif


typedef digit_t fp_t[NWORDS_FIELD]; // Datatype for representing field elements

extern const digit_t ONE[NWORDS_FIELD];
extern const digit_t ZERO[NWORDS_FIELD];
// extern const digit_t PM1O3[NWORDS_FIELD];

void fp_set_small(fp_t *x, const digit_t val);
void fp_mul_small(fp_t *x, const fp_t *a, const uint32_t val);
void fp_set_zero(fp_t *x);
void fp_set_one(fp_t *x);
uint32_t fp_is_equal(const fp_t *a, const fp_t *b);
uint32_t fp_is_zero(const fp_t *a);
void fp_copy(fp_t *out, const fp_t *a);

void fp_encode(void *dst, const fp_t *a);
void fp_decode_reduce(fp_t *d, const void *src, size_t len);
uint32_t fp_decode(fp_t *d, const void *src);

void fp_select(fp_t *d, const fp_t *a0, const fp_t *a1, uint32_t ctl);
void fp_cswap(fp_t *a, fp_t *b, uint32_t ctl);

void fp_add(fp_t *out, const fp_t *a, const fp_t *b);
void fp_sub(fp_t *out, const fp_t *a, const fp_t *b);
void fp_neg(fp_t *out, const fp_t *a);
void fp_sqr(fp_t *out, const fp_t *a);
void fp_mul(fp_t *out, const fp_t *a, const fp_t *b);

void fp_inv(fp_t *x);
uint32_t fp_is_square(const fp_t *a);
void fp_sqrt(fp_t *a);
void fp_half(fp_t *out, const fp_t *a);
void fp_exp3div4(fp_t *out, const fp_t *a);
void fp_div3(fp_t *out, const fp_t *a);

/*New vectorization*/
void prop_2(uint32x4_t *n);
uint32x4_t div5(uint32x4_t* in);
void fp_mul_batched(uint32x2_t *out, uint32x4_t *a, uint32x4_t *b);
void modmul32(const uint32_t *a, const uint32_t *b, uint32_t *c);
uint32_t prop32 (uint32_t *n);
int flatten32(uint32_t *n);
int modfsb32(uint32_t *n);
void redc32(uint32_t *n, uint32_t *m);
uint32_t fp_is_zero_32(uint32_t* p);
void __fp_mul_batched_asm(uint32x2_t *out, uint32x4_t *a, uint32x4_t *b);
void __fp_mul_shift_batched__asm(uint32x2_t *out, uint32x4_t *a, uint32x4_t *b);
void __fp2_add_batched_asm(uint32x4_t *out, uint32x4_t *a, uint32x4_t *b);
#endif
