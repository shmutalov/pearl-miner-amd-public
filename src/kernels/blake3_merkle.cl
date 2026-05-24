// BLAKE3-keyed Merkle tree builder for the miner's A and B matrices.
//
// Two kernels:
//
//   1. blake3_chunk_cvs: one work-item per 1024-byte chunk, produces the
//      leaf CVs of the keyed Merkle tree. With m*k = 512 MiB at pool shape
//      this is 524288 chunks; the kernel fully saturates the RX 570 in a
//      single launch.
//
//   2. blake3_merkle_layer: combines pairs of CVs into one parent CV. The
//      host launches this once per tree level (19 levels at pool shape),
//      flipping the ROOT flag only on the final pair.
//
// The compress function is a copy of the one in blake3_hash.cl —
// pyopencl doesn't support #include so we duplicate. (If kernels move to
// a shared header someday, that's the time to refactor.)

inline uint rotr32_m(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

#define G_M(a, b, c, d, mx, my)             \
    v##a = v##a + v##b + (mx);              \
    v##d = rotr32_m(v##d ^ v##a, 16u);      \
    v##c = v##c + v##d;                     \
    v##b = rotr32_m(v##b ^ v##c, 12u);      \
    v##a = v##a + v##b + (my);              \
    v##d = rotr32_m(v##d ^ v##a, 8u);       \
    v##c = v##c + v##d;                     \
    v##b = rotr32_m(v##b ^ v##c, 7u);

#define ROUND_M(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) \
    G_M(0, 4, 8,  12, m[s0], m[s1]);  \
    G_M(1, 5, 9,  13, m[s2], m[s3]);  \
    G_M(2, 6, 10, 14, m[s4], m[s5]);  \
    G_M(3, 7, 11, 15, m[s6], m[s7]);  \
    G_M(0, 5, 10, 15, m[s8], m[s9]);  \
    G_M(1, 6, 11, 12, m[sa], m[sb]);  \
    G_M(2, 7, 8,  13, m[sc], m[sd]);  \
    G_M(3, 4, 9,  14, m[se], m[sf]);

#define MFLAG_CHUNK_START 0x01u
#define MFLAG_CHUNK_END   0x02u
#define MFLAG_PARENT      0x04u
#define MFLAG_ROOT        0x08u
#define MFLAG_KEYED_HASH  0x10u

// Same compress as blake3_hash.cl — returns the next chaining value
// (v[0..7] XOR v[8..15] per BLAKE3 spec).
inline void compress_block_m(
    uint out[8],
    const uint cv[8],
    const uint m_in[16],
    uint counter_lo,
    uint counter_hi,
    uint block_len,
    uint flags)
{
    const uint IV0 = 0x6A09E667u;
    const uint IV1 = 0xBB67AE85u;
    const uint IV2 = 0x3C6EF372u;
    const uint IV3 = 0xA54FF53Au;

    uint v0  = cv[0], v1  = cv[1], v2  = cv[2], v3  = cv[3];
    uint v4  = cv[4], v5  = cv[5], v6  = cv[6], v7  = cv[7];
    uint v8  = IV0,   v9  = IV1,   v10 = IV2,   v11 = IV3;
    uint v12 = counter_lo, v13 = counter_hi;
    uint v14 = block_len,  v15 = flags;

    uint m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = m_in[i];

    ROUND_M(0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
    ROUND_M(2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8);
    ROUND_M(3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1);
    ROUND_M(10, 7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6);
    ROUND_M(12,13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4);
    ROUND_M(9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7);
    ROUND_M(11,15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13);

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
// Kernel 1: leaf chunk CVs
// ---------------------------------------------------------------------------
// One work-item per 1024-byte chunk. Reads 16 blocks of 64 bytes, chains
// 16 keyed BLAKE3 compressions with CHUNK_START / CHUNK_END flags on the
// first / last block. Writes the 8 u32 CV to ``cvs_out[chunk_idx*8..+8]``.

__kernel void blake3_chunk_cvs(
    __global const uchar* restrict data,
    __global uint* restrict cvs_out,
    __constant const uint* restrict key,
    const uint n_chunks)
{
    const uint chunk_idx = get_global_id(0);
    if (chunk_idx >= n_chunks) return;

    uint cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    const __global uchar* chunk_data = data + (size_t)chunk_idx * 1024;

    uint m[16];
    #pragma unroll
    for (int block_idx = 0; block_idx < 16; ++block_idx) {
        const __global uchar* p = chunk_data + (size_t)block_idx * 64;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            uint b0 = (uint)p[j * 4 + 0];
            uint b1 = (uint)p[j * 4 + 1];
            uint b2 = (uint)p[j * 4 + 2];
            uint b3 = (uint)p[j * 4 + 3];
            m[j] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        uint flags = MFLAG_KEYED_HASH;
        if (block_idx == 0)  flags |= MFLAG_CHUNK_START;
        if (block_idx == 15) flags |= MFLAG_CHUNK_END;

        uint new_cv[8];
        compress_block_m(new_cv, cv, m, chunk_idx, 0u, 64u, flags);
        #pragma unroll
        for (int j = 0; j < 8; ++j) cv[j] = new_cv[j];
    }

    __global uint* out = cvs_out + (size_t)chunk_idx * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = cv[j];
}

// Edge case: input that fits in a single chunk. Same as blake3_chunk_cvs
// but the FINAL block additionally gets the ROOT flag (so the chunk's CV
// IS the root of the tree). Output is a single 8-u32 CV at cvs_out[0..7].
//
// The actual-data-length-aware variant — caller passes the real byte count
// in ``data_len`` so the last block's block_len matches BLAKE3 semantics.
__kernel void blake3_single_chunk_root(
    __global const uchar* restrict data,
    __global uint* restrict cv_out,
    __constant const uint* restrict key,
    const uint data_len)
{
    if (get_global_id(0) != 0) return;
    if (data_len == 0u || data_len > 1024u) return;

    uint cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    uint n_blocks = (data_len + 63u) / 64u;
    if (n_blocks == 0u) n_blocks = 1u;
    uint bytes_remaining = data_len;

    uint m[16];
    for (uint block_idx = 0u; block_idx < n_blocks; ++block_idx) {
        const __global uchar* p = data + (size_t)block_idx * 64;
        // Load (and zero-pad) one block.
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            uint pos = (uint)block_idx * 64u + (uint)j * 4u;
            uint b0 = pos     < data_len ? (uint)data[pos]     : 0u;
            uint b1 = pos + 1 < data_len ? (uint)data[pos + 1] : 0u;
            uint b2 = pos + 2 < data_len ? (uint)data[pos + 2] : 0u;
            uint b3 = pos + 3 < data_len ? (uint)data[pos + 3] : 0u;
            m[j] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        uint flags = MFLAG_KEYED_HASH;
        if (block_idx == 0u) flags |= MFLAG_CHUNK_START;
        if (block_idx == n_blocks - 1u) {
            flags |= MFLAG_CHUNK_END;
            flags |= MFLAG_ROOT;
        }
        uint block_len = bytes_remaining < 64u ? bytes_remaining : 64u;

        uint new_cv[8];
        compress_block_m(new_cv, cv, m, 0u, 0u, block_len, flags);
        #pragma unroll
        for (int j = 0; j < 8; ++j) cv[j] = new_cv[j];
        bytes_remaining -= block_len;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv_out[j] = cv[j];
}


// ---------------------------------------------------------------------------
// Kernel 2: one Merkle layer (n_in CVs → n_in/2 parent CVs)
// ---------------------------------------------------------------------------
// One work-item per parent. The host launches this once per level. If
// ``is_root_layer`` is non-zero (only when n_in == 2), the parent gets the
// ROOT flag — that parent's CV IS the Merkle root.

__kernel void blake3_merkle_layer(
    __global const uint* restrict cvs_in,
    __global uint* restrict cvs_out,
    __constant const uint* restrict key,
    const uint n_in,
    const int is_root_layer)
{
    const uint parent_idx = get_global_id(0);
    const uint n_out = n_in >> 1;
    if (parent_idx >= n_out) return;

    // Build the 64-byte parent message = concat(left_cv, right_cv).
    uint m[16];
    const __global uint* left  = cvs_in + (size_t)(2u * parent_idx) * 8u;
    const __global uint* right = cvs_in + (size_t)(2u * parent_idx + 1u) * 8u;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        m[j]     = left[j];
        m[j + 8] = right[j];
    }

    uint flags = MFLAG_KEYED_HASH | MFLAG_PARENT;
    if (is_root_layer) flags |= MFLAG_ROOT;

    uint key_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) key_reg[j] = key[j];

    uint new_cv[8];
    compress_block_m(new_cv, key_reg, m, 0u, 0u, 64u, flags);

    __global uint* out = cvs_out + (size_t)parent_idx * 8u;
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = new_cv[j];
}
