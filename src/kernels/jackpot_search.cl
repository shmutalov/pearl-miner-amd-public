// Batched candidate evaluation for Pearl's noisy-GEMM PoW.
//
// One work-group = one candidate. Thread layout: (u, v) over (h, w) where
// h = rows_pattern.size, w = cols_pattern.size (typically 2 × 64 = 128 threads).
//
// For each candidate the kernel:
//   1. Loads e_al strips (h rows of r bytes) and e_br_t strips (w cols of r
//      bytes) into LDS — these depend only on this candidate's t_rows/t_cols
//      and stay constant across the k/r iterations.
//   2. Iterates ll = r, 2r, ..., k:
//      a. Each thread (u, v) computes
//           jackpot_uv = sum_{l in [ll-r, ll)}
//                          (A[a_row][l] + noise_a[u][l]) *
//                          (B[b_col][l] + noise_b[v][l])
//         where noise_a[u][l] = e_al[u][e_ar_t[l][0]] - e_al[u][e_ar_t[l][1]]
//         and   noise_b[v][l] = e_br_t[v][e_bl[l][0]] - e_br_t[v][e_bl[l][1]].
//      b. WG-wide XOR-reduce across all h*w jackpot_uv → xored_tile.
//      c. tid=0 updates one of 16 jackpot_msg slots:
//           slot = (ll/r - 1) % 16
//           jackpot_msg[slot] = rotl32(jackpot_msg[slot], 13) ^ xored_tile.
//   3. After all iterations, thread 0 runs one BLAKE3 block compression
//      (keyed by a_noise_seed) over the 64-byte jackpot_msg → hash_jackpot.
//
// Output is the 32-byte hash_jackpot per candidate; host compares it
// little-endian against the share target.

// ---------------------------------------------------------------------------
// BLAKE3 single-block keyed compress (KEYED_HASH | CHUNK_START | CHUNK_END | ROOT)
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

// Compress one 64-byte block as a keyed BLAKE3 root block.
//   key[8] = 32-byte BLAKE3 key (== a_noise_seed for jackpot hash)
//   m_in[16] = 64-byte message (the jackpot_msg)
//   out[8] = 32-byte digest (post-XOR-fold first half)
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

    // Standard BLAKE3 post-XOR fold; first 8 words = 32-byte digest.
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

// Compile-time tuneables — host must match these exactly.
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

// Left rotation per outer iteration — Pearl protocol constant.
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
    __global const int* restrict t_rows,     // (B,) int32 — per-candidate t_rows
    __global const int* restrict t_cols,     // (B,) int32 — per-candidate t_cols
    __constant const int* restrict row_pattern,  // (h,) int32, e.g. [0, 32]
    __constant const int* restrict col_pattern,  // (w,) int32, e.g. [0..63]
    const int n_cols_A,                       // = k for A[:, k]
    const int n_cols_B,                       // = k for B[:, k]
    const int k,                              // contraction dim (== n_cols_A)
    __constant const uint* restrict a_noise_seed,  // (8,) uint32 = 32-byte key
    __global uint* restrict out_hash_jackpot  // (B, 8) uint32 = (B, 32) bytes
)
{
    const int cand = get_group_id(0);
    const int tid  = get_local_id(0);
    const int u    = tid / PEARL_W;   // 0..h-1
    const int v    = tid % PEARL_W;   // 0..w-1

    // ---- Resolve this candidate's A row / B col ----------------------------
    const int t_r  = t_rows[cand];
    const int t_c  = t_cols[cand];
    const int a_row = t_r + row_pattern[u];
    const int b_col = t_c + col_pattern[v];

    // ---- LDS: per-candidate noise + secret strips + accumulator scratch ---
    // e_al rows for the touched A rows: h × r bytes (256 B at pool shape).
    // e_br_t rows for the touched B cols: w × r bytes (8 KiB at pool shape).
    // a_strip: A's touched-row slice for the current outer iter (h × r bytes).
    //   At pool shape only h=2 distinct a_rows exist per WG, so the naive
    //   global load is 64x redundant; caching once per outer iter cuts that.
    // xor_scratch for the WG-wide XOR reduction: hw uint (512 B).
    // jackpot_msg: 16 uint.
    __local char el_al[PEARL_H * PEARL_R];
    __local char el_br[PEARL_W * PEARL_R];
    __local char a_strip[PEARL_H * PEARL_R];
    // volatile so the compiler can't reorder the manual wavefront-sync
    // tree reduce below; xor_scratch entries are produced and consumed
    // by neighboring lanes within the same wavefront with no barriers.
    __local volatile uint xor_scratch[PEARL_HW];
    __local uint jackpot_msg[JACKPOT_SIZE];

    // Cooperative load: each thread loads one el_al byte for each (u_load, l)
    // combination over PEARL_H × PEARL_R = 256 bytes. With WG_SIZE=128 each
    // thread loads exactly 2 bytes.
    for (int idx = tid; idx < PEARL_H * PEARL_R; idx += WG_SIZE) {
        int u_load = idx / PEARL_R;
        int l_load = idx % PEARL_R;
        int row    = t_r + row_pattern[u_load];
        el_al[idx] = e_al[row * PEARL_R + l_load];
    }
    // el_br: w × r = 64 × 128 = 8192 bytes; 64 bytes per thread.
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

    // ---- Outer loop over k in steps of r -----------------------------------
    // n_iters = k / r (32 at pool shape).
    const int n_iters = k / PEARL_R;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int ll_lo = iter * PEARL_R;

        // Refresh a_strip for this outer iter (h*r = 256 B = 2 B per thread).
        // Skipping the b_strip cache on purpose — w=64 distinct b_cols only
        // halve the redundancy (2x) and the b_strip would cost +8 KiB LDS,
        // dropping occupancy from ~7 to ~3 WGs per CU.
        for (int idx = tid; idx < PEARL_H * PEARL_R; idx += WG_SIZE) {
            int u_load = idx / PEARL_R;
            int l_load = idx % PEARL_R;
            int row    = t_r + row_pattern[u_load];
            a_strip[idx] = A[(long)row * n_cols_A + ll_lo + l_load];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Per-thread (u, v): accumulate the r=128 inner products of this slice.
        // (sa+na) and (sb+nb) fit in 8 bits, product in 14 bits, and the sum
        // over 128 iterations stays well under 2^23 → mad24 is safe and
        // single-cycle on gfx803.
        int acc = 0;
        __local const char* my_el_al = el_al + u * PEARL_R;
        __local const char* my_el_br = el_br + v * PEARL_R;
        __local const char* my_a_strip = a_strip + u * PEARL_R;
        __attribute__((opencl_unroll_hint(4)))
        for (int l = 0; l < PEARL_R; ++l) {
            const int gl = ll_lo + l;
            uint2 par = e_ar_t[gl];
            int na = (int)my_el_al[par.x] - (int)my_el_al[par.y];
            uint2 pbl = e_bl[gl];
            int nb = (int)my_el_br[pbl.x] - (int)my_el_br[pbl.y];

            int sa = (int)my_a_strip[l];
            int sb = (int)B[(long)b_col * n_cols_B + gl];

            acc = mad24(sa + na, sb + nb, acc);
        }

        // WG-wide XOR reduction of all 128 acc values.
        // Two-stage: one explicit barrier to combine the two 64-lane
        // wavefronts, then a wavefront-internal tree reduce with no
        // barriers — GCN executes the 64 lanes of a wavefront in
        // lockstep, and volatile xor_scratch keeps the compiler from
        // reordering the dependent LDS loads/stores. Drops 8 barriers
        // per outer iter to 2.
        xor_scratch[tid] = (uint)acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 64) {
            xor_scratch[tid] ^= xor_scratch[tid + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) {
            xor_scratch[tid] ^= xor_scratch[tid + 32];
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

    // ---- Final BLAKE3 keyed compress over jackpot_msg ----------------------
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
