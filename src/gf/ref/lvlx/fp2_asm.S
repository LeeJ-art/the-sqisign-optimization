.align 6
.global __fp2_add_batched_asm
.global ___fp2_add_batched_asm


__fp2_add_batched_asm:
___fp2_add_batched_asm:
    //ABI
    stp x29, x30, [sp, -16]!
    mov     x29, sp

    //ldr
    ldr q0, [x1, #0*16]
    ldr q23, [x2, #0*16]
    add v0.4s, v0.4s, v23.4s
    str q0, [x0, #0]
    ldr q0, [x1, #0*16+144]
    ldr q23, [x2, #0*16+144]
    add v0.4s, v0.4s, v23.4s
    str q0, [x0, #0+144]
    
    ldr q1, [x1, #1*16]
    ldr q24, [x2, #1*16]
    add v1.4s, v1.4s, v24.4s
    str q1, [x0, #1*16]
    ldr q1, [x1, #1*16+144]
    ldr q24, [x2, #1*16+144]
    add v1.4s, v1.4s, v24.4s
    str q1, [x0, #1*16+144]

    ldr q2, [x1, #2*16]
    ldr q25, [x2, #2*16]
    add v2.4s, v2.4s, v25.4s
    str q2, [x0, #2*16]
    ldr q2, [x1, #2*16+144]
    ldr q25, [x2, #2*16+144]
    add v2.4s, v2.4s, v25.4s
    str q2, [x0, #2*16+144]

    ldr q3, [x1, #3*16]
    ldr q26, [x2, #3*16]
    add v3.4s, v3.4s, v26.4s
    str q3, [x0, #3*16]
    ldr q3, [x1, #3*16+144]
    ldr q26, [x2, #3*16+144]
    add v3.4s, v3.4s, v26.4s
    str q3, [x0, #3*16+144]

    ldr q4, [x1, #4*16]
    ldr q27, [x2, #4*16]
    add v4.4s, v4.4s, v27.4s
    str q4, [x0, #4*16]
    ldr q4, [x1, #4*16+144]
    ldr q27, [x2, #4*16+144]
    add v4.4s, v4.4s, v27.4s
    str q4, [x0, #4*16+144]

    ldr q5, [x1, #5*16]
    ldr q28, [x2, #5*16]
    add v5.4s, v5.4s, v28.4s
    str q5, [x0, #5*16]
    ldr q5, [x1, #5*16+144]
    ldr q28, [x2, #5*16+144]
    add v5.4s, v5.4s, v28.4s
    str q5, [x0, #5*16+144]
    
    ldr q6, [x1, #6*16]
    ldr q29, [x2, #6*16]
    add v6.4s, v6.4s, v29.4s
    str q6, [x0, #6*16]
    ldr q6, [x1, #6*16+144]
    ldr q29, [x2, #6*16+144]
    add v6.4s, v6.4s, v29.4s
    str q6, [x0, #6*16+144]

    ldr q7, [x1, #7*16]
    ldr q30, [x2, #7*16]
    add v7.4s, v7.4s, v30.4s
    str q7, [x0, #7*16]
    ldr q7, [x1, #7*16+144]
    ldr q30, [x2, #7*16+144]
    add v7.4s, v7.4s, v30.4s
    str q7, [x0, #7*16+144]

    ldr q16, [x1, #8*16]
    ldr q31, [x2, #8*16]
    add v16.4s, v16.4s, v31.4s
    str q16, [x0, #8*16]
    ldr q16, [x1, #8*16+144]
    ldr q31, [x2, #8*16+144]
    add v16.4s, v16.4s, v31.4s
    str q16, [x0, #8*16+144]
    
    ldp     x29, x30, [sp], 16
    ret