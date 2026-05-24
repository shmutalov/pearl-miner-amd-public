// GPU noise generators ported from `circuit/pearl_noise.rs`.
//
// Two kernels:
//
//   1. pearl_noise_uniform_int8
//      For each (row, block-in-row): build a 64-byte BLAKE3 message
//
//        msg[0..4]  = LE(block_idx + 1)        // block_idx = row*hashes_per_row + block_in_row
//        msg[4..32] = zeros
//        msg[32..64] = seed                    // SEED_LABEL_A or _B, 32 bytes
//
//      keyed by `a_noise_seed` / `b_noise_seed` (single-block KEYED_HASH +
//      CHUNK_START + CHUNK_END + ROOT), and write 32 sign-extended-to-i7
//      bytes (`(b & 0x3F) - 32` in [-32, 31]) into out[row, block_in_row*32:].
//      `num_cols` must be a multiple of 32 (= BLAKE3 digest size); at pool
//      shape r = 128 → 4 hashes per row.
//
//   2. pearl_noise_perm_u32
//      One work-item per "line_block" (= k/8 work-items total). Builds:
//
//        msg[0..4]   = zeros
//        msg[4..8]   = LE(line_block + 1)      // prepend_index = 1
//        msg[8..32]  = zeros
//        msg[32..64] = seed
//
//      Same KEYED_HASH + CHUNK_START + CHUNK_END + ROOT compress. The
//      resulting 8 u32 words produce 8 (first_idx, second_idx) perm pairs
//      (with rank_mask = noise_rank - 1, second_idx XOR rule per Rust source).
//
// Both kernels read the 32-byte `seed` and 32-byte `key` from __constant
// buffers and use a copy-pasted BLAKE3 compress (pyopencl doesn't support
// #include).

inline uint rotr32_n(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

#define G_N(a, b, c, d, mx, my)             \
    v##a = v##a + v##b + (mx);              \
    v##d = rotr32_n(v##d ^ v##a, 16u);      \
    v##c = v##c + v##d;                     \
    v##b = rotr32_n(v##b ^ v##c, 12u);      \
    v##a = v##a + v##b + (my);              \
    v##d = rotr32_n(v##d ^ v##a, 8u);       \
    v##c = v##c + v##d;                     \
    v##b = rotr32_n(v##b ^ v##c, 7u);

#define ROUND_N(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) \
    G_N(0, 4, 8,  12, m[s0], m[s1]);  \
    G_N(1, 5, 9,  13, m[s2], m[s3]);  \
    G_N(2, 6, 10, 14, m[s4], m[s5]);  \
    G_N(3, 7, 11, 15, m[s6], m[s7]);  \
    G_N(0, 5, 10, 15, m[s8], m[s9]);  \
    G_N(1, 6, 11, 12, m[sa], m[sb]);  \
    G_N(2, 7, 8,  13, m[sc], m[sd]);  \
    G_N(3, 4, 9,  14, m[se], m[sf]);

#define NFLAG_CHUNK_START 0x01u
#define NFLAG_CHUNK_END   0x02u
#define NFLAG_ROOT        0x08u
#define NFLAG_KEYED_HASH  0x10u

// Single-block keyed compress. cv[] = key words; flags = KEYED_HASH |
// CHUNK_START | CHUNK_END | ROOT. counter = 0. block_len = 64. Returns
// the 8-word CV (= v[0..8] XOR v[8..16] per BLAKE3 spec).
inline void compress_keyed_single(
    uint out8[8],
    const uint cv[8],
    const uint m_in[16])
{
    const uint IV0 = 0x6A09E667u;
    const uint IV1 = 0xBB67AE85u;
    const uint IV2 = 0x3C6EF372u;
    const uint IV3 = 0xA54FF53Au;

    uint v0  = cv[0], v1  = cv[1], v2  = cv[2], v3  = cv[3];
    uint v4  = cv[4], v5  = cv[5], v6  = cv[6], v7  = cv[7];
    uint v8  = IV0,   v9  = IV1,   v10 = IV2,   v11 = IV3;
    uint v12 = 0u,    v13 = 0u;
    uint v14 = 64u;
    uint v15 = NFLAG_KEYED_HASH | NFLAG_CHUNK_START | NFLAG_CHUNK_END | NFLAG_ROOT;

    uint m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = m_in[i];

    ROUND_N(0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
    ROUND_N(2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8);
    ROUND_N(3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1);
    ROUND_N(10, 7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6);
    ROUND_N(12,13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4);
    ROUND_N(9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7);
    ROUND_N(11,15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13);

    out8[0] = v0 ^ v8;
    out8[1] = v1 ^ v9;
    out8[2] = v2 ^ v10;
    out8[3] = v3 ^ v11;
    out8[4] = v4 ^ v12;
    out8[5] = v5 ^ v13;
    out8[6] = v6 ^ v14;
    out8[7] = v7 ^ v15;
}


// One work-item per BLAKE3 hash block. Each row consumes `hashes_per_row`
// = num_cols / 32 hashes. Total work-items = num_rows * hashes_per_row.
__kernel void pearl_noise_uniform_int8(
    __constant const uint* restrict seed,       // 8 u32 (32 bytes)
    __constant const uint* restrict key,        // 8 u32 (32 bytes)
    __global char* restrict out,                // (num_rows * num_cols) int8
    const uint num_rows,
    const uint num_cols)                        // must be multiple of 32
{
    const uint hashes_per_row = num_cols >> 5;  // num_cols / 32
    const ulong total = (ulong)num_rows * (ulong)hashes_per_row;
    const ulong gid = get_global_id(0);
    if (gid >= total) return;

    const uint block_idx = (uint)gid;           // CPU formula: row * hashes_per_row + block_in_row
                                                // = block_lo + (block - block_lo), exactly block_idx.
    const uint row = block_idx / hashes_per_row;
    const uint block_in_row = block_idx - row * hashes_per_row;

    // Build message: LE(block_idx+1) at offset 0, zeros, seed at offset 32.
    uint m[16];
    m[0] = block_idx + 1u;
    #pragma unroll
    for (int j = 1; j < 8; ++j) m[j] = 0u;
    #pragma unroll
    for (int j = 0; j < 8; ++j) m[8 + j] = seed[j];

    uint cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    uint h[8];
    compress_keyed_single(h, cv, m);

    // Write 32 bytes, sign-extending each: (b & 0x3F) - 32 ∈ [-32, 31].
    const ulong byte_offset = (ulong)row * (ulong)num_cols + (ulong)block_in_row * 32ul;
    __global char* dst = out + byte_offset;
    #pragma unroll
    for (uint i = 0; i < 32u; ++i) {
        uint word = h[i >> 2];
        uint b = (word >> ((i & 3u) * 8u)) & 0xFFu;
        int s = (int)(b & 0x3Fu) - 32;
        dst[i] = (char)s;
    }
}


// One work-item per "line block" = group of 8 consecutive output rows.
// Each emits up to 8 (first_idx, second_idx) uint32 pairs.
__kernel void pearl_noise_perm_u32(
    __constant const uint* restrict seed,
    __constant const uint* restrict key,
    __global uint* restrict out,                // (k, 2) u32 — stored as 2*k u32
    const uint k,
    const uint noise_rank)
{
    const uint lines_per_hash = 8u;
    const uint n_blocks = (k + lines_per_hash - 1u) / lines_per_hash;
    const ulong gid = get_global_id(0);
    if (gid >= (ulong)n_blocks) return;

    const uint line_block = (uint)gid;

    uint m[16];
    m[0] = 0u;
    m[1] = line_block + 1u;        // prepend_index = 1
    #pragma unroll
    for (int j = 2; j < 8; ++j) m[j] = 0u;
    #pragma unroll
    for (int j = 0; j < 8; ++j) m[8 + j] = seed[j];

    uint cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    uint h[8];
    compress_keyed_single(h, cv, m);

    const uint rank_mask = noise_rank - 1u;
    const uint mh_factor = noise_rank - 1u;

    for (uint j = 0; j < lines_per_hash; ++j) {
        uint row = line_block * lines_per_hash + j;
        if (row >= k) break;
        uint v = h[j];
        uint first_idx = v & rank_mask;
        uint second_idx = first_idx ^ (1u + mul_hi(mh_factor, v));
        out[row * 2u + 0u] = first_idx;
        out[row * 2u + 1u] = second_idx;
    }
}
