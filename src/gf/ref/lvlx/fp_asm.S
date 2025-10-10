.align 6
.global __fp_mul_batched_asm
.global ___fp_mul_batched_asm
.global __fp_mul_shift_batched__asm
.global __fp_mul_shift_batched__asm

__fp_mul_batched_asm:
___fp_mul_batched_asm:
    //ABI
    stp x29, x30, [sp, -160]!
    mov     x29, sp
    stp     q8,  q9,  [sp, 16] 
    stp     q10, q11, [sp, 48] 
    stp     q12, q13, [sp, 80] 
    stp     q14, q15, [sp, 112]

    //mask
    MOV x7, #0x1FFFFFFF
    DUP v10.2D, x7
    //t
    MOV x7, #0
    DUP v11.2D, x7
    DUP v12.2D, x7
    //mod
    MOV w7, #0x50000
    DUP v13.4s, w7

    //ldr
    ldr q14, [x1, #0*16]
    ldr q15, [x1, #1*16]
    ldr q16, [x1, #2*16]
    ldr q17, [x1, #3*16]
    ldr q18, [x1, #4*16]
    ldr q19, [x1, #5*16]
    ldr q20, [x1, #6*16]
    ldr q21, [x1, #7*16]
    ldr q22, [x1, #8*16]

    ldr q23, [x2, #0*16]
    ldr q24, [x2, #1*16]
    ldr q25, [x2, #2*16]
    ldr q26, [x2, #3*16]
    ldr q27, [x2, #4*16]
    ldr q28, [x2, #5*16]
    ldr q29, [x2, #6*16]
    ldr q30, [x2, #7*16]
    ldr q31, [x2, #8*16]

    //x0
    umlal	v11.2d, v14.2s, v23.2s
    umlal2	v12.2d, v14.4s, v23.4s
    and	v0.16b, v11.16b, v10.16b
    xtn	v0.2s, v0.2d
    and	v1.16b, v12.16b, v10.16b
    xtn2 v0.4S, v1.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x1
    umlal	v11.2d, v14.2s, v24.2s
    umlal2	v12.2d, v14.4s, v24.4s
    umlal	v11.2d, v15.2s, v23.2s
    umlal2	v12.2d, v15.4s, v23.4s
    and	v1.16b, v11.16b, v10.16b
    xtn	v1.2s, v1.2d
    and	v2.16b, v12.16b, v10.16b
    xtn2 v1.4S, v2.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x2
    umlal	v11.2d, v14.2s, v25.2s
    umlal2	v12.2d, v14.4s, v25.4s
    umlal	v11.2d, v15.2s, v24.2s
    umlal2	v12.2d, v15.4s, v24.4s
    umlal	v11.2d, v16.2s, v23.2s
    umlal2	v12.2d, v16.4s, v23.4s
    and	v2.16b, v11.16b, v10.16b
    xtn	v2.2s, v2.2d
    and	v3.16b, v12.16b, v10.16b
    xtn2 v2.4S, v3.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x3
    umlal	v11.2d, v14.2s, v26.2s
    umlal2	v12.2d, v14.4s, v26.4s
    umlal	v11.2d, v15.2s, v25.2s
    umlal2	v12.2d, v15.4s, v25.4s
    umlal	v11.2d, v16.2s, v24.2s
    umlal2	v12.2d, v16.4s, v24.4s
    umlal	v11.2d, v17.2s, v23.2s
    umlal2	v12.2d, v17.4s, v23.4s
    and	v3.16b, v11.16b, v10.16b
    xtn	v3.2s, v3.2d
    and	v4.16b, v12.16b, v10.16b
    xtn2 v3.4S, v4.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x4
    umlal	v11.2d, v14.2s, v27.2s
    umlal2	v12.2d, v14.4s, v27.4s
    umlal	v11.2d, v15.2s, v26.2s
    umlal2	v12.2d, v15.4s, v26.4s
    umlal	v11.2d, v16.2s, v25.2s
    umlal2	v12.2d, v16.4s, v25.4s
    umlal	v11.2d, v17.2s, v24.2s
    umlal2	v12.2d, v17.4s, v24.4s
    umlal	v11.2d, v18.2s, v23.2s
    umlal2	v12.2d, v18.4s, v23.4s
    and	v4.16b, v11.16b, v10.16b
    xtn	v4.2s, v4.2d
    and	v5.16b, v12.16b, v10.16b
    xtn2 v4.4S, v5.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x5
    umlal	v11.2d, v14.2s, v28.2s
    umlal2	v12.2d, v14.4s, v28.4s
    umlal	v11.2d, v15.2s, v27.2s
    umlal2	v12.2d, v15.4s, v27.4s
    umlal	v11.2d, v16.2s, v26.2s
    umlal2	v12.2d, v16.4s, v26.4s
    umlal	v11.2d, v17.2s, v25.2s
    umlal2	v12.2d, v17.4s, v25.4s
    umlal	v11.2d, v18.2s, v24.2s
    umlal2	v12.2d, v18.4s, v24.4s
    umlal	v11.2d, v19.2s, v23.2s
    umlal2	v12.2d, v19.4s, v23.4s
    and	v5.16b, v11.16b, v10.16b
    xtn	v5.2s, v5.2d
    and	v6.16b, v12.16b, v10.16b
    xtn2 v5.4S, v6.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x6
    umlal	v11.2d, v14.2s, v29.2s
    umlal2	v12.2d, v14.4s, v29.4s
    umlal	v11.2d, v15.2s, v28.2s
    umlal2	v12.2d, v15.4s, v28.4s
    umlal	v11.2d, v16.2s, v27.2s
    umlal2	v12.2d, v16.4s, v27.4s
    umlal	v11.2d, v17.2s, v26.2s
    umlal2	v12.2d, v17.4s, v26.4s
    umlal	v11.2d, v18.2s, v25.2s
    umlal2	v12.2d, v18.4s, v25.4s
    umlal	v11.2d, v19.2s, v24.2s
    umlal2	v12.2d, v19.4s, v24.4s
    umlal	v11.2d, v20.2s, v23.2s
    umlal2	v12.2d, v20.4s, v23.4s
    and	v6.16b, v11.16b, v10.16b
    xtn	v6.2s, v6.2d
    and	v7.16b, v12.16b, v10.16b
    xtn2 v6.4S, v7.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x7
    umlal	v11.2d, v14.2s, v30.2s
    umlal2	v12.2d, v14.4s, v30.4s
    umlal	v11.2d, v15.2s, v29.2s
    umlal2	v12.2d, v15.4s, v29.4s
    umlal	v11.2d, v16.2s, v28.2s
    umlal2	v12.2d, v16.4s, v28.4s
    umlal	v11.2d, v17.2s, v27.2s
    umlal2	v12.2d, v17.4s, v27.4s
    umlal	v11.2d, v18.2s, v26.2s
    umlal2	v12.2d, v18.4s, v26.4s
    umlal	v11.2d, v19.2s, v25.2s
    umlal2	v12.2d, v19.4s, v25.4s
    umlal	v11.2d, v20.2s, v24.2s
    umlal2	v12.2d, v20.4s, v24.4s
    umlal	v11.2d, v21.2s, v23.2s
    umlal2	v12.2d, v21.4s, v23.4s
    and	v7.16b, v11.16b, v10.16b
    xtn	v7.2s, v7.2d
    and	v8.16b, v12.16b, v10.16b
    xtn2 v7.4S, v8.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x8
    umlal	v11.2d, v14.2s, v31.2s
    umlal2	v12.2d, v14.4s, v31.4s
    umlal	v11.2d, v15.2s, v30.2s
    umlal2	v12.2d, v15.4s, v30.4s
    umlal	v11.2d, v16.2s, v29.2s
    umlal2	v12.2d, v16.4s, v29.4s
    umlal	v11.2d, v17.2s, v28.2s
    umlal2	v12.2d, v17.4s, v28.4s
    umlal	v11.2d, v18.2s, v27.2s
    umlal2	v12.2d, v18.4s, v27.4s
    umlal	v11.2d, v19.2s, v26.2s
    umlal2	v12.2d, v19.4s, v26.4s
    umlal	v11.2d, v20.2s, v25.2s
    umlal2	v12.2d, v20.4s, v25.4s
    umlal	v11.2d, v21.2s, v24.2s
    umlal2	v12.2d, v21.4s, v24.4s
    umlal	v11.2d, v22.2s, v23.2s
    umlal2	v12.2d, v22.4s, v23.4s
    //mod
    umlal	v11.2d, v0.2s, v13.2s
    umlal2	v12.2d, v0.4s, v13.4s
    and	v8.16b, v11.16b, v10.16b
    xtn	v8.2s, v8.2d
    and	v9.16b, v12.16b, v10.16b
    xtn2 v8.4S, v9.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x9
    umlal	v11.2d, v15.2s, v31.2s
    umlal2	v12.2d, v15.4s, v31.4s
    umlal	v11.2d, v16.2s, v30.2s
    umlal2	v12.2d, v16.4s, v30.4s
    umlal	v11.2d, v17.2s, v29.2s
    umlal2	v12.2d, v17.4s, v29.4s
    umlal	v11.2d, v18.2s, v28.2s
    umlal2	v12.2d, v18.4s, v28.4s
    umlal	v11.2d, v19.2s, v27.2s
    umlal2	v12.2d, v19.4s, v27.4s
    umlal	v11.2d, v20.2s, v26.2s
    umlal2	v12.2d, v20.4s, v26.4s
    umlal	v11.2d, v21.2s, v25.2s
    umlal2	v12.2d, v21.4s, v25.4s
    umlal	v11.2d, v22.2s, v24.2s
    umlal2	v12.2d, v22.4s, v24.4s
    //mod
    umlal	v11.2d, v1.2s, v13.2s
    umlal2	v12.2d, v1.4s, v13.4s
    and	v0.16b, v11.16b, v10.16b
    xtn	v0.2s, v0.2d
    and	v1.16b, v12.16b, v10.16b
    xtn2 v0.4S, v1.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x10
    umlal	v11.2d, v16.2s, v31.2s
    umlal2	v12.2d, v16.4s, v31.4s
    umlal	v11.2d, v17.2s, v30.2s
    umlal2	v12.2d, v17.4s, v30.4s
    umlal	v11.2d, v18.2s, v29.2s
    umlal2	v12.2d, v18.4s, v29.4s
    umlal	v11.2d, v19.2s, v28.2s
    umlal2	v12.2d, v19.4s, v28.4s
    umlal	v11.2d, v20.2s, v27.2s
    umlal2	v12.2d, v20.4s, v27.4s
    umlal	v11.2d, v21.2s, v26.2s
    umlal2	v12.2d, v21.4s, v26.4s
    umlal	v11.2d, v22.2s, v25.2s
    umlal2	v12.2d, v22.4s, v25.4s
    //mod
    umlal	v11.2d, v2.2s, v13.2s
    umlal2	v12.2d, v2.4s, v13.4s
    and	v1.16b, v11.16b, v10.16b
    xtn	v1.2s, v1.2d
    and	v2.16b, v12.16b, v10.16b
    xtn2 v1.4S, v2.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x11
    umlal	v11.2d, v17.2s, v31.2s
    umlal2	v12.2d, v17.4s, v31.4s
    umlal	v11.2d, v18.2s, v30.2s
    umlal2	v12.2d, v18.4s, v30.4s
    umlal	v11.2d, v19.2s, v29.2s
    umlal2	v12.2d, v19.4s, v29.4s
    umlal	v11.2d, v20.2s, v28.2s
    umlal2	v12.2d, v20.4s, v28.4s
    umlal	v11.2d, v21.2s, v27.2s
    umlal2	v12.2d, v21.4s, v27.4s
    umlal	v11.2d, v22.2s, v26.2s
    umlal2	v12.2d, v22.4s, v26.4s
    //mod
    umlal	v11.2d, v3.2s, v13.2s
    umlal2	v12.2d, v3.4s, v13.4s
    and	v2.16b, v11.16b, v10.16b
    xtn	v2.2s, v2.2d
    and	v3.16b, v12.16b, v10.16b
    xtn2 v2.4S, v3.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x12
    umlal	v11.2d, v18.2s, v31.2s
    umlal2	v12.2d, v18.4s, v31.4s
    umlal	v11.2d, v19.2s, v30.2s
    umlal2	v12.2d, v19.4s, v30.4s
    umlal	v11.2d, v20.2s, v29.2s
    umlal2	v12.2d, v20.4s, v29.4s
    umlal	v11.2d, v21.2s, v28.2s
    umlal2	v12.2d, v21.4s, v28.4s
    umlal	v11.2d, v22.2s, v27.2s
    umlal2	v12.2d, v22.4s, v27.4s
    umlal	v11.2d, v4.2s, v13.2s    //mod
    umlal2	v12.2d, v4.4s, v13.4s
    and	v3.16b, v11.16b, v10.16b
    xtn	v3.2s, v3.2d
    and	v4.16b, v12.16b, v10.16b
    xtn2 v3.4S, v4.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x13
    umlal	v11.2d, v19.2s, v31.2s
    umlal2	v12.2d, v19.4s, v31.4s
    umlal	v11.2d, v20.2s, v30.2s
    umlal2	v12.2d, v20.4s, v30.4s
    umlal	v11.2d, v21.2s, v29.2s
    umlal2	v12.2d, v21.4s, v29.4s
    umlal	v11.2d, v22.2s, v28.2s
    umlal2	v12.2d, v22.4s, v28.4s    //mod
    umlal	v11.2d, v5.2s, v13.2s
    umlal2	v12.2d, v5.4s, v13.4s
    and	v4.16b, v11.16b, v10.16b
    xtn	v4.2s, v4.2d
    and	v5.16b, v12.16b, v10.16b
    xtn2 v4.4S, v5.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x14
    umlal	v11.2d, v20.2s, v31.2s
    umlal2	v12.2d, v20.4s, v31.4s
    umlal	v11.2d, v21.2s, v30.2s
    umlal2	v12.2d, v21.4s, v30.4s
    umlal	v11.2d, v22.2s, v29.2s
    umlal2	v12.2d, v22.4s, v29.4s
    umlal	v11.2d, v6.2s, v13.2s     //mod
    umlal2	v12.2d, v6.4s, v13.4s
    and	v5.16b, v11.16b, v10.16b
    xtn	v5.2s, v5.2d
    and	v6.16b, v12.16b, v10.16b
    xtn2 v5.4S, v6.2D
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x15
    umlal	v11.2d, v21.2s, v31.2s
    umlal2	v12.2d, v21.4s, v31.4s
    umlal	v11.2d, v22.2s, v30.2s
    umlal2	v12.2d, v22.4s, v30.4s
    umlal	v11.2d, v7.2s, v13.2s       //mod
    umlal2	v12.2d, v7.4s, v13.4s
    and	v6.16b, v11.16b, v10.16b
    xtn	v6.2s, v6.2d
    and	v7.16b, v12.16b, v10.16b
    xtn2 v6.4s, v7.2d
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x16
    umlal	v11.2d, v22.2s, v31.2s
    umlal2	v12.2d, v22.4s, v31.4s
    umlal	v11.2d, v8.2s, v13.2s       //mod
    umlal2	v12.2d, v8.4s, v13.4s
    and	v7.16b, v11.16b, v10.16b
    xtn	v7.2s, v7.2d
    and	v8.16b, v12.16b, v10.16b
    xtn2 v7.4s, v8.2d
    ushr	v11.2d, v11.2d, 29
    ushr	v12.2d, v12.2d, 29
    //x17
    xtn	v8.2s, v11.2d
    xtn2 v8.4s, v12.2d

    STR q0, [x0, #0]
    STR q1, [x0, #1*16]
    STR q2, [x0, #2*16]
    STR q3, [x0, #3*16]
    STR q4, [x0, #4*16]
    STR q5, [x0, #5*16]
    STR q6, [x0, #6*16]
    STR q7, [x0, #7*16]
    STR q8, [x0, #8*16]
    

    ldp     q8,  q9,  [sp, 16]
    ldp     q10, q11, [sp, 48]
    ldp     q12, q13, [sp, 80]
    ldp     q14, q15, [sp, 112]
    ldp     x29, x30, [sp], 160
    
    ret

__fp_mul_shift_batched__asm:
___fp_mul_shift_batched__asm:
    stp x29, x30, [sp, -16]!
    mov     x29, sp


    //mask
    MOV x7, #0x1FFFFFFF
	ldr	q8, [x1]


	//x0
    ldr	q30, [x2]
    umull	v13.2d, v8.2s, v30.2s //L[0, 0]
	DUP v0.2D, x7
	umull2	v18.2d, v8.4s, v30.4s
	ldr	q26, [x2, 16]
	ushr	v25.2d, v13.2d, 29
	ushr	v1.2d, v18.2d, 29
    ldr	q28, [x1, 16]

    //x1
	umlal	v25.2d, v8.2s, v26.2s //L[0, 1]
    ldp	q9, q15, [x2, 32]
	umlal2	v1.2d, v8.4s, v26.4s
	umlal	v25.2d, v28.2s, v30.2s //L[1, 0]
    ldr	q11, [x1, 32]
	umlal2	v1.2d, v28.4s, v30.4s
	ushr	v24.2d, v25.2d, 29
	ushr	v23.2d, v1.2d, 29

    //x2
	umlal	v24.2d, v8.2s, v9.2s //L[0, 2]
    and	v13.16b, v13.16b, v0.16b
	umlal2	v23.2d, v8.4s, v9.4s
	and	v18.16b, v18.16b, v0.16b
	umlal	v24.2d, v28.2s, v26.2s //L[1, 1]
    and	v1.16b, v1.16b, v0.16b
	umlal2	v23.2d, v28.4s, v26.4s
	and	v25.16b, v25.16b, v0.16b
	umlal	v24.2d, v11.2s, v30.2s //L[2, 0]
    ldr	q2, [x1, 48]
	umlal2	v23.2d, v11.4s, v30.4s
    ldr	q17, [x2, 64]
	ushr	v22.2d, v24.2d, 29
	ushr	v21.2d, v23.2d, 29

    //x3 4mla + 4mla2 + 2and + 2shr
	umlal	v22.2d, v8.2s, v15.2s //L[0, 3]
    and	v24.16b, v24.16b, v0.16b
	umlal2	v21.2d, v8.4s, v15.4s
    and	v23.16b, v23.16b, v0.16b
	umlal	v22.2d, v28.2s, v9.2s //L[1, 2]
	umlal2	v21.2d, v28.4s, v9.4s
	umlal	v22.2d, v11.2s, v26.2s //L[2, 1]
	umlal2	v21.2d, v11.4s, v26.4s
	umlal	v22.2d, v2.2s, v30.2s //L[3, 0]
	ushr	v20.2d, v22.2d, 29
	umlal2	v21.2d, v2.4s, v30.4s
	and	v22.16b, v22.16b, v0.16b
	umlal	v20.2d, v8.2s, v17.2s //L[0, 4]
	ushr	v19.2d, v21.2d, 29

    //x4 5mla + 5mla2 + 2and + 2shr
	umlal2	v19.2d, v8.4s, v17.4s
	and	v21.16b, v21.16b, v0.16b
	umlal	v20.2d, v28.2s, v15.2s //L[1, 3]
	umlal2	v19.2d, v28.4s, v15.4s
	umlal	v20.2d, v2.2s, v26.2s //L[3, 1]
	umlal2	v19.2d, v2.4s, v26.4s
    umlal	v20.2d, v11.2s, v9.2s //L[2, 2]
	ldr	q2, [x1, 64]
    umlal2	v19.2d, v11.4s, v9.4s
    ldr q16, [x2, 80]
	umlal	v20.2d, v2.2s, v30.2s //L[4, 0]
	ushr	v7.2d, v20.2d, 29
	umlal2	v19.2d, v2.4s, v30.4s
	and	v20.16b, v20.16b, v0.16b
	umlal	v7.2d, v8.2s, v16.2s //L[0, 5]
	ushr	v6.2d, v19.2d, 29

    //x5 6mla + 6mla2 + 2and + 2shr
	umlal2	v6.2d, v8.4s, v16.4s
    and	v19.16b, v19.16b, v0.16b
	umlal	v7.2d, v28.2s, v17.2s //L[1, 4]
    ldr	q2, [x1, 48]
	umlal2	v6.2d, v28.4s, v17.4s
    umlal	v7.2d, v2.2s, v9.2s  //L[3, 2]
    umlal2	v6.2d, v2.4s, v9.4s
	umlal	v7.2d, v11.2s, v15.2s //L[2, 3]
	umlal2	v6.2d, v11.4s, v15.4s
    ldr	q2, [x1, 64]
	umlal	v7.2d, v2.2s, v26.2s //L[4, 1]
    ldr q16, [x1, 80]
	umlal2	v6.2d, v2.4s, v26.4s
    ldr	q2, [x2, 96]
	umlal	v7.2d, v16.2s, v30.2s //L[5, 0]
	ushr	v5.2d, v7.2d, 29
    umlal2	v6.2d, v16.4s, v30.4s
    ushr	v4.2d, v6.2d, 29 

    //x6 7mla + 7mla2 + 2and + 2shr
    umlal	v5.2d, v8.2s, v2.2s //L[0,6]
    and	v7.16b, v7.16b, v0.16b  
	umlal2	v4.2d, v8.4s, v2.4s
	ldr	q2, [x2, 80]
	umlal	v5.2d, v28.2s, v2.2s //L[1, 5]
	and	v6.16b, v6.16b, v0.16b
	umlal2	v4.2d, v28.4s, v2.4s
	ldr	q2, [x1, 48]
	umlal	v5.2d, v11.2s, v17.2s //L[2, 4]
	umlal2	v4.2d, v11.4s, v17.4s
	umlal	v5.2d, v2.2s, v15.2s  //L[3, 3]
    ldr	q17, [x1, 64]
	umlal2	v4.2d, v2.4s, v15.4s
	umlal	v5.2d, v17.2s, v9.2s  //L[4, 2]
	umlal2	v4.2d, v17.4s, v9.4s
	ldr	q2, [x1, 80]
	umlal	v5.2d, v2.2s, v26.2s  //L[5, 1]
    ldr	q17, [x1, 96]
	umlal2	v4.2d, v2.4s, v26.4s
	umlal	v5.2d, v17.2s, v30.2s //L[6, 0]
	umlal2	v4.2d, v17.4s, v30.4s
    ldr	q17, [x2, 112]
    ushr	v3.2d, v5.2d, 29
    and	v5.16b, v5.16b, v0.16b

    //x7 8mla + 8mla2 + 2and + 2shr
    umlal	v3.2d, v8.2s, v17.2s //L[0, 7]
    ushr	v2.2d, v4.2d, 29
	umlal2	v2.2d, v8.4s, v17.4s
    and	v4.16b, v4.16b, v0.16b
    umlal	v3.2d, v16.2s, v9.2s //L[5, 2]
	ldr	q17, [x2, 96]
	umlal2	v2.2d, v16.4s, v9.4s
    umlal	v3.2d, v28.2s, v17.2s //L[1, 6]
	umlal2	v2.2d, v28.4s, v17.4s
	ldr	q16, [x2, 80]
    umlal	v3.2d, v11.2s, v16.2s //[2, 5]
    ldr	q17, [x1, 64]
	umlal2	v2.2d, v11.4s, v16.4s
    ldr	q12, [x2, 64]
    umlal	v3.2d, v17.2s, v15.2s //L[4, 3]
	ldr	q16, [x1, 48]
    umlal2	v2.2d, v17.4s, v15.4s
    umlal	v3.2d, v16.2s, v12.2s //L[3, 4]
	umlal2	v2.2d, v16.4s, v12.4s
    ldr	q17, [x1, 96]
    umlal	v3.2d, v17.2s, v26.2s //L[6, 1]
	umlal2	v2.2d, v17.4s, v26.4s
	ldr	q16, [x1, 112]
	umlal	v3.2d, v16.2s, v30.2s //L[7, 0]
	umlal2	v2.2d, v16.4s, v30.4s
    ushr	v17.2d, v3.2d, 29



	//x8 9mla + 9mla2 + 2and + 2shr + 4add + 4shl
	umlal	v17.2d, v16.2s, v26.2s //L[7 1]
	ushr	v27.2d, v2.2d, 29
	umlal2	v27.2d, v16.4s, v26.4s
	ldr	q16, [x1, 64]
	umlal	v17.2d, v16.2s, v12.2s //L[4, 4]
	and	v3.16b, v3.16b, v0.16b
	umlal2	v27.2d, v16.4s, v12.4s
	ldr	q16, [x2, 128]
    umlal	v17.2d, v8.2s, v16.2s //L[0, 8]
	and	v2.16b, v2.16b, v0.16b
	umlal2	v27.2d, v8.4s, v16.4s
	ldr	q12, [x2, 112]
	umlal	v17.2d, v28.2s, v12.2s //[1, 7]
	ldr	q8, [x2, 96]
	umlal2	v27.2d, v28.4s, v12.4s
	ldr	q14, [x2, 80]
	umlal	v17.2d, v11.2s, v8.2s //[2, 6]
	ldr	q12, [x1, 48]
	umlal2	v27.2d, v11.4s, v8.4s
	ldr	q8, [x1, 80]
	umlal	v17.2d, v12.2s, v14.2s //[3, 5]
	ldr	q16, [x2, 48]
	umlal2	v27.2d, v12.4s, v14.4s
	ldr	q12, [x1, 96]
	umlal	v17.2d, v8.2s, v16.2s //[5, 3]
	ldr	q14, [x1, 128]
	umlal2	v27.2d, v8.4s, v16.4s
	umlal	v17.2d, v12.2s, v9.2s //[6, 2]
	umlal2	v27.2d, v12.4s, v9.4s
	umlal	v17.2d, v14.2s, v30.2s //[8, 0]
	umlal2	v27.2d, v14.4s, v30.4s
	shl	v8.2d, v13.2d, 18
	shl	v13.2d, v13.2d, 16
	shl	v30.2d, v18.2d, 18
	add	v8.2d, v8.2d, v13.2d
	shl	v18.2d, v18.2d, 16
	add	v30.2d, v30.2d, v18.2d
	add	v17.2d, v8.2d, v17.2d
	add	v27.2d, v30.2d, v27.2d
	ushr	v8.2d, v17.2d, 29
	and	v17.16b, v17.16b, v0.16b
	ushr	v12.2d, v27.2d, 29
	
	//x9 8mla + 8mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v14.2s, v26.2s //[8 1]
	and	v27.16b, v27.16b, v0.16b
	umlal2	v12.2d, v14.4s, v26.4s
	ldr	q13, [x1, 96]
	umlal	v8.2d, v13.2s, v16.2s //[6, 3]
	ldr	q14, [x1, 112]
	umlal2	v12.2d, v13.4s, v16.4s
	ldr	q18, [x1, 80]
	umlal	v8.2d, v14.2s, v9.2s //[7, 2]
	ldr	q13, [x2, 64]
	umlal2	v12.2d, v14.4s, v9.4s
	umlal	v8.2d, v18.2s, v13.2s //[5, 4]
	ldr	q14, [x1, 64]
	umlal2	v12.2d, v18.4s, v13.4s
	ldr	q13, [x2, 80]
	umlal	v8.2d, v14.2s, v13.2s //[4, 5]
	umlal2	v12.2d, v14.4s, v13.4s
	ldr	q18, [x1, 48]
	ldr	q14, [x2, 96]
	umlal	v8.2d, v18.2s, v14.2s //[3, 6]
	umlal2	v12.2d, v18.4s, v14.4s
	ldr	q13, [x2, 112]
	umlal	v8.2d, v11.2s, v13.2s //[2, 7]
	umlal2	v12.2d, v11.4s, v13.4s
	ldr	q14, [x2, 128]
	umlal	v8.2d, v28.2s, v14.2s //[1, 8]
	umlal2	v12.2d, v28.4s, v14.4s
	shl	v28.2d, v25.2d, 18

	shl	v25.2d, v25.2d, 16
	add	v28.2d, v28.2d, v25.2d
	shl	v26.2d, v1.2d, 18
	add	v8.2d, v28.2d, v8.2d
	shl	v1.2d, v1.2d, 16
	add	v26.2d, v26.2d, v1.2d
	and v28.16b, v8.16b, v0.16b
	add	v12.2d, v26.2d, v12.2d
	ushr	v8.2d, v8.2d, 29
	and v26.16b, v12.16b, v0.16b
	ushr	v12.2d, v12.2d, 29
	xtn	v28.2s, v28.2d


	//x10 7mla + 7mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v11.2s, v14.2s //[2, 8]
	xtn	v26.2s, v26.2d
	umlal2	v12.2d, v11.4s, v14.4s
	stp	d28, d26, [x0]
	umlal	v8.2d, v18.2s, v13.2s //[3, 7]
	ldr q25, [x1, 64]
	umlal2	v12.2d, v18.4s, v13.4s
	ldr q1, [x2, 96]
	umlal	v8.2d, v25.2s, v1.2s //[4, 6]
	ldr q30, [x1, 80]
	umlal2	v12.2d, v25.4s, v1.4s
	ldr q11, [x2, 80]
	umlal	v8.2d, v30.2s, v11.2s //[5, 5]
	ldr q25, [x1, 96]
	umlal2	v12.2d, v30.4s, v11.4s
	ldr q1, [x2, 64]
	umlal	v8.2d, v25.2s, v1.2s //[6, 4]
	ldr q30, [x1, 112]
	umlal2	v12.2d, v25.4s, v1.4s
	ldr q11, [x2, 48]
	umlal	v8.2d, v30.2s, v11.2s //[7, 3]
	ldr q25, [x1, 128]
	umlal2	v12.2d, v30.4s, v11.4s
	ldr q26, [x1, 64]
	umlal	v8.2d, v25.2s, v9.2s //[8, 2]
	shl	v28.2d, v24.2d, 18
	umlal2	v12.2d, v25.4s, v9.4s
	shl	v24.2d, v24.2d, 16

	add	v24.2d, v28.2d, v24.2d
	shl	v9.2d, v23.2d, 18
	shl	v23.2d, v23.2d, 16
	add	v8.2d, v8.2d, v24.2d
	add	v23.2d, v9.2d, v23.2d
	and v24.16b, v8.16b, v0.16b
	add	v12.2d, v12.2d, v23.2d
	ushr	v8.2d, v8.2d, 29
	and v23.16b, v12.16b, v0.16b
	
	//x11 6mla + 6mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v18.2s, v14.2s //[3, 8]
	ushr	v12.2d, v12.2d, 29
	umlal2	v12.2d, v18.4s, v14.4s
	xtn	v24.2s, v24.2d
	umlal	v8.2d, v25.2s, v11.2s //[8, 3]
	xtn	v23.2s, v23.2d
	umlal2	v12.2d, v25.4s, v11.4s
	stp	d24, d23, [x0, #1*16]
	umlal	v8.2d, v30.2s, v1.2s //[7, 4]
	ldr q18, [x1, 96]
	umlal2	v12.2d, v30.4s, v1.4s
	ldr q23, [x2, 80]
	umlal	v8.2d, v18.2s, v23.2s //[6, 5]
	ldr q11, [x1, 80]
	umlal2	v12.2d, v18.4s, v23.4s
	ldr q24, [x2, 96]
	umlal	v8.2d, v11.2s, v24.2s //[5, 6]
	shl	v28.2d, v22.2d, 18
	umlal2	v12.2d, v11.4s, v24.4s
	shl	v29.2d, v21.2d, 18
	umlal	v8.2d, v26.2s, v13.2s //[4, 7]
	shl	v22.2d, v22.2d, 16
	umlal2	v12.2d, v26.4s, v13.4s
	shl	v21.2d, v21.2d, 16

	add v22.2d, v22.2d, v28.2d
	add v21.2d, v21.2d, v29.2d
	add	v8.2d, v8.2d, v22.2d
	add	v12.2d, v12.2d, v21.2d
	and v28.16b, v8.16b, v0.16b
	ushr	v8.2d, v8.2d, 29
	and v29.16b, v12.16b, v0.16b
	ushr	v12.2d, v12.2d, 29


	//x12 5mla + 5mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v26.2s, v14.2s //[4, 8]
	xtn	v28.2s, v28.2d
	umlal2	v12.2d, v26.4s, v14.4s
	xtn	v29.2s, v29.2d
	umlal	v8.2d, v11.2s, v13.2s //[5, 7]
	stp	d28, d29, [x0, #2*16]
	umlal2	v12.2d, v11.4s, v13.4s
	shl	v28.2d, v20.2d, 18
	umlal	v8.2d, v18.2s, v24.2s //[6, 6]
	shl	v29.2d, v19.2d, 18
	umlal2	v12.2d, v18.4s, v24.4s
	shl	v20.2d, v20.2d, 16
	umlal	v8.2d, v30.2s, v23.2s //[7, 5]
	shl	v19.2d, v19.2d, 16
	umlal2	v12.2d, v30.4s, v23.4s
	add v20.2d, v20.2d, v28.2d
	umlal	v8.2d, v25.2s, v1.2s //[8, 4]
	add v19.2d, v19.2d, v29.2d
	umlal2	v12.2d, v25.4s, v1.4s
	add	v8.2d, v8.2d, v20.2d

	add	v12.2d, v12.2d, v19.2d
	and v28.16b, v8.16b, v0.16b
	ushr	v8.2d, v8.2d, 29
	and v29.16b, v12.16b, v0.16b


	//x13 4mla + 4mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v11.2s, v14.2s //[5, 8]
	ushr	v12.2d, v12.2d, 29
	umlal2	v12.2d, v11.4s, v14.4s
	xtn	v28.2s, v28.2d
	umlal	v8.2d, v18.2s, v13.2s //[6, 7]
	xtn	v29.2s, v29.2d
	umlal2	v12.2d, v18.4s, v13.4s
	stp	d28, d29, [x0, #3*16]
	umlal	v8.2d, v30.2s, v24.2s //[7, 6]
	shl	v28.2d, v7.2d, 18
	umlal2	v12.2d, v30.4s, v24.4s
	shl	v29.2d, v6.2d, 18
	umlal	v8.2d, v25.2s, v23.2s //[8, 5]
	shl	v7.2d, v7.2d, 16
	umlal2	v12.2d, v25.4s, v23.4s
	shl	v6.2d, v6.2d, 16

	add v7.2d, v7.2d, v28.2d
	add v6.2d, v6.2d, v29.2d
	add	v8.2d, v8.2d, v7.2d
	add	v12.2d, v12.2d, v6.2d

	and v28.16b, v8.16b, v0.16b
	ushr	v8.2d, v8.2d, 29
	and v29.16b, v12.16b, v0.16b
	ushr	v12.2d, v12.2d, 29

	//x14 3mla + 3mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v18.2s, v14.2s //[6, 8]
	xtn	v28.2s, v28.2d
	umlal2	v12.2d, v18.4s, v14.4s
	xtn	v29.2s, v29.2d
	umlal	v8.2d, v30.2s, v13.2s //[7, 7]
	stp	d28, d29, [x0, #4*16]
	umlal2	v12.2d, v30.4s, v13.4s
	shl	v28.2d, v5.2d, 18
	umlal	v8.2d, v25.2s, v24.2s //[8, 6]
	shl	v29.2d, v4.2d, 18
	umlal2	v12.2d, v25.4s, v24.4s
	shl	v5.2d, v5.2d, 16

	shl	v4.2d, v4.2d, 16
	add v5.2d, v5.2d, v28.2d
	add v4.2d, v4.2d, v29.2d
	add	v8.2d, v8.2d, v5.2d
	add	v12.2d, v12.2d, v4.2d
	and v28.16b, v8.16b, v0.16b
	ushr	v8.2d, v8.2d, 29
	and v29.16b, v12.16b, v0.16b

	//x15 2mla + 2mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v30.2s, v14.2s //[7, 8]
	ushr	v12.2d, v12.2d, 29
	umlal2	v12.2d, v30.4s, v14.4s
	shl	v11.2d, v3.2d, 18
	umlal	v8.2d, v25.2s, v13.2s //[8, 7]
	shl	v21.2d, v2.2d, 18
	umlal2	v12.2d, v25.4s, v13.4s
	shl	v3.2d, v3.2d, 16

	xtn	v28.2s, v28.2d
	shl	v2.2d, v2.2d, 16

	xtn	v29.2s, v29.2d
	add v3.2d, v3.2d, v11.2d
	add v2.2d, v2.2d, v21.2d
	add	v8.2d, v8.2d, v3.2d
	add	v12.2d, v12.2d, v2.2d
	and v11.16b, v8.16b, v0.16b
	ushr	v8.2d, v8.2d, 29
	and v21.16b, v12.16b, v0.16b
	

	//x16 2mla + 2mla2 + 2and + 2shr + 4add + 4shl
	umlal	v8.2d, v25.2s, v14.2s //[8, 8]
	ushr	v12.2d, v12.2d, 29
	umlal2	v12.2d, v25.4s, v14.4s
	stp	d28, d29, [x0, #5*16]

	xtn	v11.2s, v11.2d
	shl	v31.2d, v17.2d, 18
    xtn	v21.2s, v21.2d
	shl	v16.2d, v27.2d, 18
    stp	d11, d21, [x0, #6*16]
	shl	v17.2d, v17.2d, 16
	add v17.2d, v17.2d, v31.2d
    shl	v27.2d, v27.2d, 16

	add v27.2d, v27.2d, v16.2d
	add	v8.2d, v8.2d, v17.2d
	add	v12.2d, v12.2d, v27.2d
	shrn	v3.2s, v8.2d, 29
    and	v8.16b, v8.16b, v0.16b
    shrn	v4.2s, v12.2d, 29
    and	v12.16b, v12.16b, v0.16b
	xtn	v8.2s, v8.2d
	xtn	v12.2s, v12.2d
	stp	d8, d12, [x0, #7*16]
	stp	d3, d4, [x0, #8*16]
    ldp     x29, x30, [sp], 16
    ret