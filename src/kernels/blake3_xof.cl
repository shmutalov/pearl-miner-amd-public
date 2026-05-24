// BLAKE3 XOF kernel used by derive_matrix on the GPU.
//
// `derive_matrix(seed, rows, cols, domain_tag)` boils down to
// `blake3(domain_tag || seed).digest(length=rows*cols)` plus a per-byte
// sign-extend ((b & 0x7F) - ((b & 0x40) << 1)) producing int8 values in
// [-64, 63]. The input is tiny (≤ 64 bytes), so the host pre-computes the
// "root compress" parameters: an 8-word chaining value cv, a 16-word
// message block m, a block_len, and the base flags (CHUNK_START | CHUNK_END
// for the typical ≤ 64 byte case). The kernel then re-runs the final
// compression with counter = output_block_index and OR's in ROOT, taking
// the full 16-word output (XOR-folded) for each 64 output bytes.
//
// One work-item per 64-byte output block. Each writes up to 64 int8 values
// into `out_int8[byte_offset .. byte_offset+64]`, clamped at n_bytes_total
// for the tail.

inline uint rotr32_x(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

#define G_X(a, b, c, d, mx, my)             \
    v##a = v##a + v##b + (mx);              \
    v##d = rotr32_x(v##d ^ v##a, 16u);      \
    v##c = v##c + v##d;                     \
    v##b = rotr32_x(v##b ^ v##c, 12u);      \
    v##a = v##a + v##b + (my);              \
    v##d = rotr32_x(v##d ^ v##a, 8u);       \
    v##c = v##c + v##d;                     \
    v##b = rotr32_x(v##b ^ v##c, 7u);

#define ROUND_X(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) \
    G_X(0, 4, 8,  12, m[s0], m[s1]);  \
    G_X(1, 5, 9,  13, m[s2], m[s3]);  \
    G_X(2, 6, 10, 14, m[s4], m[s5]);  \
    G_X(3, 7, 11, 15, m[s6], m[s7]);  \
    G_X(0, 5, 10, 15, m[s8], m[s9]);  \
    G_X(1, 6, 11, 12, m[sa], m[sb]);  \
    G_X(2, 7, 8,  13, m[sc], m[sd]);  \
    G_X(3, 4, 9,  14, m[se], m[sf]);

#define XFLAG_CHUNK_START 0x01u
#define XFLAG_CHUNK_END   0x02u
#define XFLAG_ROOT        0x08u

// Compute the full 16-word XOF output of one BLAKE3 compression. Same as
// the "compress" function but the caller wants both halves of v, XOR-folded
// with the matching CV (per the BLAKE3 spec's XOF output rule):
//
//   out[0..8]  = v[0..8]   XOR v[8..16]
//   out[8..16] = v[8..16]  XOR cv[0..8]
//
// (BLAKE3's normal "CV output" only takes the first half — XOF needs both.)
inline void compress_block_xof(
    uint out16[16],
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

    ROUND_X(0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
    ROUND_X(2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8);
    ROUND_X(3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1);
    ROUND_X(10, 7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6);
    ROUND_X(12,13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4);
    ROUND_X(9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7);
    ROUND_X(11,15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13);

    out16[0]  = v0  ^ v8;
    out16[1]  = v1  ^ v9;
    out16[2]  = v2  ^ v10;
    out16[3]  = v3  ^ v11;
    out16[4]  = v4  ^ v12;
    out16[5]  = v5  ^ v13;
    out16[6]  = v6  ^ v14;
    out16[7]  = v7  ^ v15;
    out16[8]  = v8  ^ cv[0];
    out16[9]  = v9  ^ cv[1];
    out16[10] = v10 ^ cv[2];
    out16[11] = v11 ^ cv[3];
    out16[12] = v12 ^ cv[4];
    out16[13] = v13 ^ cv[5];
    out16[14] = v14 ^ cv[6];
    out16[15] = v15 ^ cv[7];
}

// One work-item per output 64-byte block. Writes (up to) 64 int8 values
// in [-64, 63] using the BLAKE3-byte-to-i7 sign-extension:
//
//   signed = (byte & 0x7F) - ((byte & 0x40) << 1)
//
// `out_int8` is a char buffer of length `n_bytes_total`. The tail block
// writes fewer than 64 bytes when n_bytes_total is not a multiple of 64.
__kernel void blake3_xof_derive_int8(
    __constant const uint* restrict cv,         // 8 u32, root chaining value
    __constant const uint* restrict m_words,    // 16 u32, root message block
    const uint block_len,                       // bytes in the root block
    const uint base_flags,                      // flags w/o ROOT; ROOT OR'd inside
    __global char* restrict out_int8,           // (n_bytes_total,) int8 output
    const ulong n_bytes_total)
{
    const ulong gid = get_global_id(0);
    const ulong byte_offset = gid * 64ul;
    if (byte_offset >= n_bytes_total) return;

    const uint counter_lo = (uint)(gid & 0xFFFFFFFFul);
    const uint counter_hi = (uint)(gid >> 32);
    const uint flags = base_flags | XFLAG_ROOT;

    uint cv_reg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv_reg[j] = cv[j];

    uint m_reg[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) m_reg[j] = m_words[j];

    uint out16[16];
    compress_block_xof(out16, cv_reg, m_reg, counter_lo, counter_hi, block_len, flags);

    // How many bytes this block writes (full 64 except possibly the tail).
    ulong remaining = n_bytes_total - byte_offset;
    uint to_write = remaining >= 64ul ? 64u : (uint)remaining;

    __global char* dst = out_int8 + byte_offset;
    for (uint i = 0; i < to_write; ++i) {
        uint word = out16[i >> 2];
        uint b = (word >> ((i & 3u) * 8u)) & 0xFFu;
        // (b & 0x7F) - ((b & 0x40) << 1) = signed 7-bit value in [-64, 63]
        int s = (int)(b & 0x7Fu) - (int)((b & 0x40u) << 1u);
        dst[i] = (char)s;
    }
}
