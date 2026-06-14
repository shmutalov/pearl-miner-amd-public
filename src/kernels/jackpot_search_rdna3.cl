// RDNA3 (gfx11xx, wave32) optimized batched candidate evaluator.
//
// Bit-identical to jackpot_search.cl but restructured for RDNA3:
//
//   * The original kernel recomputes the noisy operands inside every
//     (u, v) thread's inner loop. But
//         pa[u][l] = A[a_row(u)][l] + noise_a[u][l]
//         pb[v][l] = B[b_col(v)][l] + noise_b[v][l]
//     and the whole per-candidate jackpot is just the tiny GEMM
//         acc[u][v] = sum_l pa[u][l] * pb[v][l].
//     pa depends only on (u, l) yet the original computes it 64x (once per
//     v); pb depends only on (v, l) yet it is computed 2x (once per u). The
//     sparse-permutation noise gather is the bottleneck, so this kernel
//     computes pa / pb cooperatively into LDS exactly once per outer iter,
//     then the inner loop is a clean int8 dot product (2 LDS reads + mad).
//
//   * Because pa, pb are now contiguous int8 in LDS, the inner product is
//     vectorized over char4 — 1 ds_read_b32 instead of 4 ds_read_i8, and a
//     shape the RDNA3 backend can fuse into v_dot4_i32_i8.
//
//   * The WG-wide XOR reduce is made wave32-safe: a barrier is inserted at
//     the 32-lane boundary so the final tree never reads across an
//     unsynchronized subgroup. (Still correct on wave64.)
//
// Value ranges (so int8 LDS storage is lossless): uniform noise e_* in
// [-32, 31] => na, nb in [-63, 63]; A, B in [-64, 63] => pa, pb in
// [-127, 126], i.e. signed int8. Products fit in 14 bits, the 128-term sum
// stays under 2^23, so mad24 is exact and int32 addition is associative
// mod 2^32 -> the XOR-reduced tile is identical regardless of accumulation
// order.

// ---------------------------------------------------------------------------
// BLAKE3 single-block keyed compress (verbatim from jackpot_search.cl)
// ---------------------------------------------------------------------------

inline uint rotr32_js(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

#define G_JS(a, b, c, d, mx, my)            \
    v##a = v##a + v##b + (mx);              \
    v##d = rotr32_js(v##d ^ v##a, 16u);     \
    v##c = v##c + v##d;                     \
    v##b = rotr32_js(v##b ^ v##c, 12u);     \
    v##a = v##a + v##b + (my);              \
    v##d = rotr32_js(v##d ^ v##a, 8u);      \
    v##c = v##c + v##d;                     \
    v##b = rotr32_js(v##b ^ v##c, 7u);

#define ROUND_JS(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) \
    G_JS(0, 4, 8,  12, m[s0], m[s1]);  \
    G_JS(1, 5, 9,  13, m[s2], m[s3]);  \
    G_JS(2, 6, 10, 14, m[s4], m[s5]);  \
    G_JS(3, 7, 11, 15, m[s6], m[s7]);  \
    G_JS(0, 5, 10, 15, m[s8], m[s9]);  \
    G_JS(1, 6, 11, 12, m[sa], m[sb]);  \
    G_JS(2, 7, 8,  13, m[sc], m[sd]);  \
    G_JS(3, 4, 9,  14, m[se], m[sf]);

#define FLAG_CHUNK_START 0x01u
#define FLAG_CHUNK_END   0x02u
#define FLAG_ROOT        0x08u
#define FLAG_KEYED_HASH  0x10u

inline void blake3_compress_root_keyed(
    uint out[8],
    const uint key[8],
    const uint m_in[16],
    uint block_len)
{
    const uint IV0 = 0x6A09E667u;
    const uint IV1 = 0xBB67AE85u;
    const uint IV2 = 0x3C6EF372u;
    const uint IV3 = 0xA54FF53Au;

    uint v0  = key[0], v1  = key[1], v2  = key[2], v3  = key[3];
    uint v4  = key[4], v5  = key[5], v6  = key[6], v7  = key[7];
    uint v8  = IV0,    v9  = IV1,    v10 = IV2,    v11 = IV3;
    uint v12 = 0u,     v13 = 0u;
    uint v14 = block_len;
    uint v15 = FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH;

    uint m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = m_in[i];

    ROUND_JS(0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
    ROUND_JS(2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8);
    ROUND_JS(3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1);
    ROUND_JS(10, 7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6);
    ROUND_JS(12,13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4);
    ROUND_JS(9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7);
    ROUND_JS(11,15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13);

    out[0] = v0 ^ v8;
    out[1] = v1 ^ v9;
    out[2] = v2 ^ v10;
    out[3] = v3 ^ v11;
    out[4] = v4 ^ v12;
    out[5] = v5 ^ v13;
    out[6] = v6 ^ v14;
    out[7] = v7 ^ v15;
}


// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

#ifndef PEARL_H
#define PEARL_H 2     // rows_pattern.size
#endif
#ifndef PEARL_W
#define PEARL_W 64    // cols_pattern.size
#endif
#ifndef PEARL_R
#define PEARL_R 128   // noise_rank
#endif
#define PEARL_HW (PEARL_H * PEARL_W)   // 128

#define LROT_PER_TILE 13u
#define JACKPOT_SIZE  16
#define WG_SIZE       PEARL_HW

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void jackpot_evaluate_batch(
    __global const char* restrict A,         // (m, k) int8
    __global const char* restrict B,         // (n, k) int8
    __global const char* restrict e_al,      // (m, r) int8
    __global const char* restrict e_br_t,    // (n, r) int8
    __global const uint2* restrict e_ar_t,   // (k, 2) uint32 pairs
    __global const uint2* restrict e_bl,     // (k, 2) uint32 pairs
    __global const int* restrict t_rows,     // (B,) int32
    __global const int* restrict t_cols,     // (B,) int32
    __constant const int* restrict row_pattern,  // (h,) int32
    __constant const int* restrict col_pattern,  // (w,) int32
    const int n_cols_A,
    const int n_cols_B,
    const int k,
    __constant const uint* restrict a_noise_seed,  // (8,) uint32 = 32-byte key
    __global uint* restrict out_hash_jackpot  // (B, 8) uint32 = (B, 32) bytes
)
{
    const int cand = get_group_id(0);
    const int tid  = get_local_id(0);
    const int u    = tid / PEARL_W;   // 0..h-1
    const int v    = tid % PEARL_W;   // 0..w-1

    const int t_r  = t_rows[cand];
    const int t_c  = t_cols[cand];
    const int b_col = t_c + col_pattern[v];

    // Secret/uniform-noise strips, cached once per candidate.
    //   el_al: h x r int8  (touched A rows' uniform noise)
    //   el_br: w x r int8  (touched B cols' uniform noise)
    // Noisy operand strips, recomputed once per outer iter:
    //   pa: h x r int8 = A_secret + noise_a
    //   pb: w x r int8 = B_secret + noise_b
    __local char el_al[PEARL_H * PEARL_R];
    __local char el_br[PEARL_W * PEARL_R];
    __local char pa[PEARL_H * PEARL_R];
    __local char pb[PEARL_W * PEARL_R];
    __local volatile uint xor_scratch[PEARL_HW];
    __local uint jackpot_msg[JACKPOT_SIZE];

    for (int idx = tid; idx < PEARL_H * PEARL_R; idx += WG_SIZE) {
        int u_load = idx / PEARL_R;
        int l_load = idx % PEARL_R;
        int row    = t_r + row_pattern[u_load];
        el_al[idx] = e_al[row * PEARL_R + l_load];
    }
    for (int idx = tid; idx < PEARL_W * PEARL_R; idx += WG_SIZE) {
        int v_load = idx / PEARL_R;
        int l_load = idx % PEARL_R;
        int col    = t_c + col_pattern[v_load];
        el_br[idx] = e_br_t[col * PEARL_R + l_load];
    }
    if (tid < JACKPOT_SIZE) {
        jackpot_msg[tid] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int n_iters = k / PEARL_R;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int ll_lo = iter * PEARL_R;

        // ---- Cooperatively build pa (h*r) and pb (w*r) for this slice ----
        // Each (u_, l_) / (v_, l_) noisy operand is computed exactly once
        // (no per-thread redundancy), then shared via LDS.
        for (int idx = tid; idx < PEARL_H * PEARL_R; idx += WG_SIZE) {
            int u_  = idx / PEARL_R;
            int l_  = idx % PEARL_R;
            int gl  = ll_lo + l_;
            int row = t_r + row_pattern[u_];
            uint2 par = e_ar_t[gl];
            __local const char* al = el_al + u_ * PEARL_R;
            int na = (int)al[par.x] - (int)al[par.y];
            int sa = (int)A[(long)row * n_cols_A + gl];
            pa[idx] = (char)(sa + na);
        }
        for (int idx = tid; idx < PEARL_W * PEARL_R; idx += WG_SIZE) {
            int v_  = idx / PEARL_R;
            int l_  = idx % PEARL_R;
            int gl  = ll_lo + l_;
            int col = t_c + col_pattern[v_];
            uint2 pbl = e_bl[gl];
            __local const char* br = el_br + v_ * PEARL_R;
            int nb = (int)br[pbl.x] - (int)br[pbl.y];
            int sb = (int)B[(long)col * n_cols_B + gl];
            pb[idx] = (char)(sb + nb);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- Inner product: acc = sum_l pa[u][l] * pb[v][l] --------------
        // char4-vectorized so the backend can emit v_dot4_i32_i8. pa/pb are
        // int8 in [-127,126]; the int32 sum is associative mod 2^32, so the
        // XOR-reduced result matches the scalar reference bit for bit.
        int acc = 0;
        __local const char4* pa4 = (__local const char4*)(pa + u * PEARL_R);
        __local const char4* pb4 = (__local const char4*)(pb + v * PEARL_R);
        __attribute__((opencl_unroll_hint(4)))
        for (int j = 0; j < PEARL_R / 4; ++j) {
            int4 a4 = convert_int4(pa4[j]);
            int4 b4 = convert_int4(pb4[j]);
            acc = mad24(a4.x, b4.x, acc);
            acc = mad24(a4.y, b4.y, acc);
            acc = mad24(a4.z, b4.z, acc);
            acc = mad24(a4.w, b4.w, acc);
        }

        // ---- WG-wide XOR reduce (wave32-safe) ---------------------------
        xor_scratch[tid] = (uint)acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 64) {
            xor_scratch[tid] ^= xor_scratch[tid + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) {
            xor_scratch[tid] ^= xor_scratch[tid + 32];
        }
        // wave32 boundary: lanes 0..31 are one subgroup below, but the +32
        // read above crossed into subgroup 1, so synchronize before the
        // unbarriered intra-subgroup tree. (Redundant but harmless on wave64.)
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) {
            xor_scratch[tid] ^= xor_scratch[tid + 16];
            xor_scratch[tid] ^= xor_scratch[tid +  8];
            xor_scratch[tid] ^= xor_scratch[tid +  4];
            xor_scratch[tid] ^= xor_scratch[tid +  2];
            xor_scratch[tid] ^= xor_scratch[tid +  1];
        }

        if (tid == 0) {
            uint xored_tile = xor_scratch[0];
            int slot = iter % JACKPOT_SIZE;
            uint cur = jackpot_msg[slot];
            uint rot = (cur << LROT_PER_TILE) | (cur >> (32u - LROT_PER_TILE));
            jackpot_msg[slot] = rot ^ xored_tile;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        uint key[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = a_noise_seed[i];
        uint m[16];
        #pragma unroll
        for (int i = 0; i < JACKPOT_SIZE; ++i) m[i] = jackpot_msg[i];
        uint hash[8];
        blake3_compress_root_keyed(hash, key, m, /*block_len=*/64u);

        __global uint* out_ptr = out_hash_jackpot + (long)cand * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_ptr[i] = hash[i];
    }
}
