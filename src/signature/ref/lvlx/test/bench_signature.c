#include <stdio.h>
#include <inttypes.h>
#include <locale.h>
#include <time.h>

#include <verification.h>
#include <signature.h>

#include <tools.h>
#include <rng.h>
#include <bench.h>
#include <bench_test_arguments.h>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static __inline__ uint64_t
rdtsc(void)
{
    return (uint64_t)cpucycles();
}

void
bench_sqisign(uint64_t bench)
{
    setlocale(LC_NUMERIC, "");
    uint64_t t0, t1;
    clock_t t;
    float ms;

    public_key_t pks[bench];
    secret_key_t sks[bench];
    signature_t sigs[bench];

    
    unsigned char msg[32] = { 0 };

    for (uint64_t i = 0; i < bench; i++) {
        public_key_init(&pks[i]);
        secret_key_init(&sks[i]);
    }

    printf("\n\nBenchmarking signatures for " STRINGIFY(SQISIGN_VARIANT) ":\n\n");


    printf("\n======================  KeyGen  ======================\n\n");
    
    uint64_t p1t = 0, p2t = 0, p11 = 0, p12 = 0, p13 = 0;
    for (uint64_t i = 0; i < bench; ++i) {

        ec_basis_t B_0_two;
        t0 = rdtsc();
        protocols_keygen_p1(&pks[i], &sks[i], &B_0_two);
        p1t += rdtsc() - t0;

        int found = 0;
        while(!found){
            t0 = rdtsc();
            found = protocols_keygen_p11(&sks[i], &B_0_two, found);
            p11 += rdtsc() - t0;

            t0 = rdtsc();
            found = protocols_keygen_p12(&sks[i], &B_0_two, found);
            p12 += rdtsc() - t0;

            t0 = rdtsc();
            found = protocols_keygen_p13(&sks[i], &B_0_two, found);
            p13 += rdtsc() - t0;
        }

        t0 = rdtsc();
        protocols_keygen_p2(&pks[i], &sks[i], &B_0_two);
        p2t += rdtsc() - t0;
    }
    printf("\x1b[34mAvg P1: %'" PRIu64 " cycles\x1b[0m\n", (p1t) / bench);
    printf("\x1b[34mAvg P11: %'" PRIu64 " cycles\x1b[0m\n", (p11) / bench);
    printf("\x1b[34mAvg P12: %'" PRIu64 " cycles\x1b[0m\n", (p12) / bench);
    printf("\x1b[34mAvg P13: %'" PRIu64 " cycles\x1b[0m\n", (p13) / bench);
    printf("\x1b[34mAvg P2: %'" PRIu64 " cycles\x1b[0m\n", (p2t) / bench);

    for (uint64_t i = 0; i < bench; i++) {
        public_key_init(&pks[i]);
        secret_key_init(&sks[i]);
    }

    t = tic();
    t0 = rdtsc();
    for (uint64_t i = 0; i < bench; ++i) {
        protocols_keygen(&pks[i], &sks[i]);
    }
    t1 = rdtsc();
    ms = (1000. * (float)(clock() - t) / CLOCKS_PER_SEC);
    printf("Average keygen time [%.2f ms]\n", (float)(ms / bench));
    printf("\x1b[34mAvg keygen: %'" PRIu64 " cycles\x1b[0m\n", (t1 - t0) / bench);


    printf("\n======================   Sign   ======================\n\n");
    uint64_t ttime[7] = {0};

    t = tic();
    t0 = rdtsc();
    for (uint64_t i = 0; i < bench; ++i) {
        protocols_sign(&(sigs[i]), &(pks[i]), &(sks[i]), msg, sizeof(msg) / sizeof(*msg), ttime);
    }
    t1 = rdtsc();
    ms = (1000. * (float)(clock() - t) / CLOCKS_PER_SEC);
    printf("\x1b[34mAvg P_com: %'" PRIu64 " cycles\x1b[0m\n", (ttime[0]) / bench);
    printf("\x1b[34mAvg P_chl: %'" PRIu64 " cycles\x1b[0m\n", (ttime[1]) / bench);
    printf("\x1b[34mAvg P_rsp: %'" PRIu64 " cycles\x1b[0m\n", (ttime[2]) / bench);
    // printf("\x1b[34mAvg P_othr: %'" PRIu64 " cycles\x1b[0m\n", (ttime[4]) / bench);
    // printf("\x1b[34mAvg P_rign: %'" PRIu64 " cycles\x1b[0m\n", (ttime[5]) / bench);
    printf("\x1b[34mAvg P_bas: %'" PRIu64 " cycles\x1b[0m\n", (ttime[3]) / bench);
    printf("Average signature time [%.2f ms]\n", (float)(ms / bench));
    printf("\x1b[34mAvg signature: %'" PRIu64 " cycles\x1b[0m\n", (t1 - t0) / bench);


    printf("\n\n======================  Verify  ======================\n\n");

    t = tic();
    t0 = rdtsc();
    for (uint64_t i = 0; i < bench; ++i) {
        int check = protocols_verify(&(sigs[i]), &(pks[i]), msg, sizeof(msg) / sizeof(*msg));
        if (!check) {
            printf("verify failed ! \n");
        }
    }
    t1 = rdtsc();
    ms = (1000. * (float)(clock() - t) / CLOCKS_PER_SEC);
    printf("Average verification time [%.2f ms]\n", (float)(ms / bench));
    printf("\x1b[34mAvg verification: %'" PRIu64 " cycles\x1b[0m\n", (t1 - t0) / bench);

    for (uint64_t i = 0; i < bench; i++) {
        public_key_finalize(&pks[i]);
        secret_key_finalize(&sks[i]);
    }
}

// run all tests in module
int
main(int argc, char *argv[])
{
    uint32_t seed[12] = { 0 };
    int iterations = SQISIGN_TEST_REPS;
    int help = 0;
    int seed_set = 0;

#ifndef NDEBUG
    fprintf(stderr,
            "\x1b[31mIt looks like SQIsign was compiled with assertions enabled.\n"
            "This will severely impact performance measurements.\x1b[0m\n");
#endif

    for (int i = 1; i < argc; i++) {
        if (!help && strcmp(argv[i], "--help") == 0) {
            help = 1;
            continue;
        }

        if (!seed_set && !parse_seed(argv[i], seed)) {
            seed_set = 1;
            continue;
        }

        if (sscanf(argv[i], "--iterations=%d", &iterations) == 1) {
            continue;
        }
    }

    if (help || iterations <= 0) {
        printf("Usage: %s [--iterations=<iterations>] [--seed=<seed>]\n", argv[0]);
        printf("Where <iterations> is the number of iterations used for benchmarking; if not "
               "present, uses the default: %d)\n",
               iterations);
        printf("Where <seed> is the random seed to be used; if not present, a random seed is "
               "generated\n");
        return 1;
    }

    if (!seed_set) {
        randombytes_select((unsigned char *)seed, sizeof(seed));
    }

    print_seed(seed);

#if defined(TARGET_BIG_ENDIAN)
    for (int i = 0; i < 12; i++) {
        seed[i] = BSWAP32(seed[i]);
    }
#endif

    randombytes_init((unsigned char *)seed, NULL, 256);
    cpucycles_init();

    bench_sqisign(iterations);

    return (0);
}
