// RDNA3 (gfx11xx, wave32) jackpot evaluator — w-tiled for higher occupancy.
//
// Same restructuring as jackpot_search_rdna3.cl (build the noisy pa/pb
// operand strips in LDS, clean char4 int8-dot inner loop), but the w
// dimension is processed in PEARL_NTILES_W tiles of TW = w/PEARL_NTILES_W
// columns so the two 8 KiB LDS buffers (el_br + pb) shrink to TW*r each.
// At pool shape (w=64, NTILES=2) that drops LDS/WG from ~17 KiB to ~9 KiB,
// roughly doubling resident workgroups per WGP — the non-tiled kernel is
// occupancy-limited (LDS-bound to ~3 waves/SIMD), and this PoW is
// gather/latency-bound, so more waves in flight hide the LDS-gather latency.
//
// Bit-identical to the reference: each candidate's per-iter XOR tile is the
// full XOR over all h*w accumulators. We compute it per w-tile (lanes
// outside the active tile contribute acc=0, the XOR identity) and combine
// tiles into tile_acc[iter]. The jackpot_msg rotl-chain only depends on the
// ordered sequence tile_acc[0..n_iters), so applying it once at the end in
// iter order reproduces the reference exactly.

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
#define PEARL_H 2
#endif
#ifndef PEARL_W
#define PEARL_W 64
#endif
#ifndef PEARL_R
#define PEARL_R 128
#endif
#ifndef PEARL_NTILES_W
#define PEARL_NTILES_W 4           // split w into this many column tiles
#endif                             // (4 is the measured occupancy sweet spot
                                   //  on gfx1100: LDS 5.4 KiB, ~11 WG/WGP)
#define PEARL_TW (PEARL_W / PEARL_NTILES_W)   // tile width (cols per tile)
#define PEARL_HW (PEARL_H * PEARL_W)
#ifndef PEARL_MAX_ITERS
#define PEARL_MAX_ITERS 64         // upper bound on k/r; host must satisfy this
#endif

#define LROT_PER_TILE 13u
#define JACKPOT_SIZE  16
#define WG_SIZE       PEARL_HW

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void jackpot_evaluate_batch(
    __global const char* restrict A,
    __global const char* restrict B,
    __global const char* restrict e_al,
    __global const char* restrict e_br_t,
    __global const uint2* restrict e_ar_t,
    __global const uint2* restrict e_bl,
    __global const int* restrict t_rows,
    __global const int* restrict t_cols,
    __constant const int* restrict row_pattern,
    __constant const int* restrict col_pattern,
    const int n_cols_A,
    const int n_cols_B,
    const int k,
    __constant const uint* restrict a_noise_seed,
    __global uint* restrict out_hash_jackpot
)
{
    const int cand = get_group_id(0);
    const int tid  = get_local_id(0);
    const int u    = tid / PEARL_W;   // 0..h-1
    const int v    = tid % PEARL_W;   // 0..w-1

    const int t_r  = t_rows[cand];
    const int t_c  = t_cols[cand];

    __local char el_al[PEARL_H * PEARL_R];
    __local char pa[PEARL_H * PEARL_R];
    __local char el_br[PEARL_TW * PEARL_R];   // one w-tile of uniform noise
    __local char pb[PEARL_TW * PEARL_R];       // one w-tile of noisy operands
    __local volatile uint xor_scratch[PEARL_HW];
    __local uint jackpot_msg[JACKPOT_SIZE];
    __local uint tile_acc[PEARL_MAX_ITERS];    // full per-iter XOR, combined over w-tiles

    const int n_iters = k / PEARL_R;

    for (int idx = tid; idx < PEARL_H * PEARL_R; idx += WG_SIZE) {
        int u_load = idx / PEARL_R;
        int l_load = idx % PEARL_R;
        int row    = t_r + row_pattern[u_load];
        el_al[idx] = e_al[row * PEARL_R + l_load];
    }
    for (int idx = tid; idx < JACKPOT_SIZE; idx += WG_SIZE) jackpot_msg[idx] = 0u;
    for (int idx = tid; idx < n_iters;     idx += WG_SIZE) tile_acc[idx]    = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Outer loop over w-tiles -------------------------------------------
    for (int wt = 0; wt < PEARL_NTILES_W; ++wt) {
        const int col_base = wt * PEARL_TW;
        const int v_local  = v - col_base;          // valid iff this lane's col is in tile
        const int active   = (v_local >= 0) && (v_local < PEARL_TW);

        // Cache this tile's uniform-noise rows (TW x r) for all its outer iters.
        for (int idx = tid; idx < PEARL_TW * PEARL_R; idx += WG_SIZE) {
            int vt = idx / PEARL_R;
            int l  = idx % PEARL_R;
            int col = t_c + col_pattern[col_base + vt];
            el_br[idx] = e_br_t[col * PEARL_R + l];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int iter = 0; iter < n_iters; ++iter) {
            const int ll_lo = iter * PEARL_R;

            // pa: h x r (depends on iter only; rebuilt per tile — cheap, 256 B).
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
            // pb: TW x r for this tile's columns.
            for (int idx = tid; idx < PEARL_TW * PEARL_R; idx += WG_SIZE) {
                int vt  = idx / PEARL_R;
                int l_  = idx % PEARL_R;
                int gl  = ll_lo + l_;
                int col = t_c + col_pattern[col_base + vt];
                uint2 pbl = e_bl[gl];
                __local const char* br = el_br + vt * PEARL_R;
                int nb = (int)br[pbl.x] - (int)br[pbl.y];
                int sb = (int)B[(long)col * n_cols_B + gl];
                pb[idx] = (char)(sb + nb);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Inner product for lanes in this tile; others contribute 0.
            int acc = 0;
            if (active) {
                __local const char4* pa4 = (__local const char4*)(pa + u * PEARL_R);
                __local const char4* pb4 = (__local const char4*)(pb + v_local * PEARL_R);
                __attribute__((opencl_unroll_hint(4)))
                for (int j = 0; j < PEARL_R / 4; ++j) {
                    int4 a4 = convert_int4(pa4[j]);
                    int4 b4 = convert_int4(pb4[j]);
                    acc = mad24(a4.x, b4.x, acc);
                    acc = mad24(a4.y, b4.y, acc);
                    acc = mad24(a4.z, b4.z, acc);
                    acc = mad24(a4.w, b4.w, acc);
                }
            }

            // WG-wide XOR reduce (wave32-safe). Inactive lanes' 0 is the XOR
            // identity, so the result is the XOR over this tile's accumulators.
            xor_scratch[tid] = (uint)acc;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid < 64) xor_scratch[tid] ^= xor_scratch[tid + 64];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid < 32) xor_scratch[tid] ^= xor_scratch[tid + 32];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid < 32) {
                xor_scratch[tid] ^= xor_scratch[tid + 16];
                xor_scratch[tid] ^= xor_scratch[tid +  8];
                xor_scratch[tid] ^= xor_scratch[tid +  4];
                xor_scratch[tid] ^= xor_scratch[tid +  2];
                xor_scratch[tid] ^= xor_scratch[tid +  1];
            }
            if (tid == 0) tile_acc[iter] ^= xor_scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ---- Apply the rotl chain in iter order, then hash ---------------------
    if (tid == 0) {
        for (int iter = 0; iter < n_iters; ++iter) {
            int slot = iter % JACKPOT_SIZE;
            uint cur = jackpot_msg[slot];
            uint rot = (cur << LROT_PER_TILE) | (cur >> (32u - LROT_PER_TILE));
            jackpot_msg[slot] = rot ^ tile_acc[iter];
        }
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
