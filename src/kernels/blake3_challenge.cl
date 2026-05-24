// BLAKE3 proof-of-work search used by pearl/v1 stratum's pearl.challenge gate.
//
// Single-block default-mode (no key) BLAKE3 over (seed || u64_le(nonce)) — that
// 40-byte input fits in one 64-byte BLAKE3 block, so no chunk tree, no parent
// nodes, no Merkle reduction. Each work-item tries one nonce.
//
// Match against the existing pearl-miner-amd BLAKE3 kernel in blake3_hash.cl:
// the compress_block round structure is identical. The only differences are:
//   - cv starts as the BLAKE3 IV (not a user-supplied key)
//   - flags = CHUNK_START | CHUNK_END | ROOT (one-block message)
//   - block_len = 40 (real input length; the rest of the 64-byte block is 0)
//   - no LDS Merkle reduction (one chunk, root output goes straight out)

inline uint rotr32_c(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

#define G_C(a, b, c, d, mx, my)            \
    v##a = v##a + v##b + (mx);             \
    v##d = rotr32_c(v##d ^ v##a, 16u);     \
    v##c = v##c + v##d;                    \
    v##b = rotr32_c(v##b ^ v##c, 12u);     \
    v##a = v##a + v##b + (my);             \
    v##d = rotr32_c(v##d ^ v##a, 8u);      \
    v##c = v##c + v##d;                    \
    v##b = rotr32_c(v##b ^ v##c, 7u);

#define ROUND_C(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) \
    G_C(0, 4, 8,  12, m[s0], m[s1]);  \
    G_C(1, 5, 9,  13, m[s2], m[s3]);  \
    G_C(2, 6, 10, 14, m[s4], m[s5]);  \
    G_C(3, 7, 11, 15, m[s6], m[s7]);  \
    G_C(0, 5, 10, 15, m[s8], m[s9]);  \
    G_C(1, 6, 11, 12, m[sa], m[sb]);  \
    G_C(2, 7, 8,  13, m[sc], m[sd]);  \
    G_C(3, 4, 9,  14, m[se], m[sf]);

#define FLAG_CHUNK_START 0x01u
#define FLAG_CHUNK_END   0x02u
#define FLAG_ROOT        0x08u

// One BLAKE3 block compression. Returns the 32-byte root digest as 8 u32
// (little-endian within each u32, exactly how BLAKE3 hash output is laid out).
inline void compress_root_block(
    uint out[8],
    const uint m_in[16],
    uint block_len)
{
    const uint IV0 = 0x6A09E667u;
    const uint IV1 = 0xBB67AE85u;
    const uint IV2 = 0x3C6EF372u;
    const uint IV3 = 0xA54FF53Au;
    const uint IV4 = 0x510E527Fu;
    const uint IV5 = 0x9B05688Cu;
    const uint IV6 = 0x1F83D9ABu;
    const uint IV7 = 0x5BE0CD19u;

    uint v0  = IV0, v1  = IV1, v2  = IV2, v3  = IV3;
    uint v4  = IV4, v5  = IV5, v6  = IV6, v7  = IV7;
    uint v8  = IV0, v9  = IV1, v10 = IV2, v11 = IV3;
    uint v12 = 0u,  v13 = 0u;   // counter = 0 (single chunk)
    uint v14 = block_len;
    uint v15 = FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT;

    uint m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = m_in[i];

    ROUND_C(0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
    ROUND_C(2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8);
    ROUND_C(3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1);
    ROUND_C(10, 7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6);
    ROUND_C(12,13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4);
    ROUND_C(9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7);
    ROUND_C(11,15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13);

    out[0] = v0 ^ v8;
    out[1] = v1 ^ v9;
    out[2] = v2 ^ v10;
    out[3] = v3 ^ v11;
    out[4] = v4 ^ v12;
    out[5] = v5 ^ v13;
    out[6] = v6 ^ v14;
    out[7] = v7 ^ v15;
}

// difficulty = required number of leading zero bits in the 32-byte digest
// when interpreted as a big-endian byte sequence (memory order). The BLAKE3
// digest is laid out as 8 LE u32 — out[0]'s low byte is digest[0], its high
// byte is digest[3], etc.
inline bool meets_difficulty(const uint out[8], uint difficulty) {
    uint d = difficulty;
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        if (d == 0) return true;
        uint w = out[wi];
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            if (d == 0) return true;
            uint b = (w >> (bi * 8)) & 0xffu;
            if (d >= 8u) {
                if (b != 0u) return false;
                d -= 8u;
            } else {
                // top `d` bits of this byte must be zero
                if ((b >> (8u - d)) != 0u) return false;
                d = 0u;
            }
        }
    }
    return true;
}

// One work-item = one nonce. Host launches in batches.
// On the first thread to find a hit:
//   - sets found_flag[0] = 1 (atomic),
//   - writes the winning nonce to found_nonce[0].
__kernel void blake3_challenge_search(
    __constant const uint* restrict seed_u32,    // 8 u32 = 32 bytes of seed
    const uint difficulty,
    const ulong nonce_base,
    __global volatile uint* restrict found_flag,
    __global ulong* restrict found_nonce)
{
    // Cheap early-out for the remainder of an already-won batch.
    if (atomic_or(found_flag, 0u) != 0u) return;

    const ulong nonce = nonce_base + (ulong)get_global_id(0);

    uint m[16];
    // Bytes 0..31  = seed
    m[0] = seed_u32[0]; m[1] = seed_u32[1];
    m[2] = seed_u32[2]; m[3] = seed_u32[3];
    m[4] = seed_u32[4]; m[5] = seed_u32[5];
    m[6] = seed_u32[6]; m[7] = seed_u32[7];
    // Bytes 32..39 = u64_le(nonce)
    m[8] = (uint)(nonce & 0xFFFFFFFFu);
    m[9] = (uint)(nonce >> 32);
    // Bytes 40..63 = 0 (zero-padded block tail)
    m[10] = 0u; m[11] = 0u; m[12] = 0u;
    m[13] = 0u; m[14] = 0u; m[15] = 0u;

    uint out[8];
    compress_root_block(out, m, 40u);

    if (meets_difficulty(out, difficulty)) {
        // First winner sets the flag; later winners race-overwrite the nonce,
        // which is fine — any winning nonce is valid.
        atomic_xchg(found_flag, 1u);
        found_nonce[0] = nonce;
    }
}
