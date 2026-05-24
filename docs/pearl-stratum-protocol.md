# Pearl stratum protocol (`pearl/v1` dialect, AlphaPool)

Reconstructed from:
1. UTF-16 string table dumped from the official `alpha-miner` binary
   (`alpha-miner-strings.txt` at the repo root). See
   [re/scripts/extract_stratum_details.py](../re/scripts/extract_stratum_details.py)
   for the exact byte offsets.
2. A live probe of [`eu1.alphapool.tech:5566`](https://pearl.alphapool.tech/),
   captured 2026-05-23.

Where wire details could not be pinned down from strings alone (e.g. the exact
field ordering inside `mining.notify`), the doc says "TBC from live capture"
and our [stratum_client.py](../src/pearl_amd/stratum_client.py) records the
raw value so it can be filled in later.

## Endpoint

| Region | Pooled (PPLNS) | Solo |
|--------|----------------|------|
| US     | `us2.alphapool.tech:5566` | `us2.alphapool.tech:5567` |
| EU     | `eu1.alphapool.tech:5566` | `eu1.alphapool.tech:5567` |
| Asia   | `sg1.alphapool.tech:5566` | `sg1.alphapool.tech:5567` |

Plain TCP, no TLS. `pearl.alphapool.tech` itself is the website behind
Cloudflare — **do not** connect a miner there.

## Framing

Newline-delimited JSON-RPC 2.0. Each message is a single JSON object terminated
by `\n`. Server may push notifications (`"id": null`, no result expected) at
any time once the connection is up.

## Handshake

The miner runs this sequence verbatim (from the recovered call sites at
`0x001f38ef` and `0x001f3907`):

```
C -> S   { "id":1, "method":"mining.configure", "params":[["pearl/v1"], {}] }
C -> S   { "id":2, "method":"mining.subscribe", "params":["alpha-miner/0.1"] }
S -> C   { "id":null, "method":"pearl.challenge",
           "params":{ "seed":<hex32>, "difficulty":<int_bits> } }
   ...repeated every ~2s until the client answers correctly...
C -> S   { "method":"pearl.challenge_response",
           "params":{ "seed":<same hex32>, "nonce":<hex8> } }   # notification
C -> S   { "id":3, "method":"mining.authorize",
           "params":["<bech32_addr>[.<worker>]", "x[;d=<diff>]"] }
S -> C   { "id":3, "result":true/false, "error":null }
S -> C   { "id":null, "method":"mining.set_difficulty", "params":[<float>] }
S -> C   { "id":null, "method":"pearl.set_mining_params",
           "params":{ "m":..., "n":..., "k":..., "rank":...,
                      "rows_pattern":[...], "cols_pattern":[...] } }
S -> C   { "id":null, "method":"mining.notify",
           "params":[<job_id>, <incomplete_header_bytes_b64>,
                     <share_nbits>, <clean_jobs?>, ...] }   # field order TBC
```

If the pool dislikes anything before authorize — wrong challenge nonce, wrong
JSON shape, unknown wallet format — it **silently closes the connection**.
The error path used by `mining.authorize` itself is the JSON-RPC `"error"`
object (the binary has the literal string `Stratum error response:`).

## `pearl.challenge` (server → client)

A BLAKE3 proof-of-work gate the pool runs against every new TCP connection
before any other work flows. Found by name at `0x001f385a`. The binary's
solver kernel lives in
`solve_blake3_challenge_kernel(const uint8_t* seed, uint32_t difficulty,
size_t max_nonce, uint32_t* found_flag, uint64_t* found_nonce)` (mangled name
at offset 0x4f9a3 — see `re/scripts/extract_stratum_details.py`).

* `seed` — 32-byte random value, hex-encoded (64 chars). Different per
  connection. Re-sent every ~2s until satisfied.
* `difficulty` — number of leading zero bits required. Observed value: **32**.
  The binary rejects values > 256 (`"BLAKE3 challenge difficulty exceeds 256 bits"`).

### Algorithm

Find a 64-bit `nonce` such that

```
BLAKE3(seed_bytes_32 || u64_le(nonce))     # 32-byte digest, big-endian
```

has at least `difficulty` leading zero bits.

Search complexity is `2^difficulty / 2` expected hashes. The official miner
solves this on the GPU; our [stratum_client.py](../src/pearl_amd/stratum_client.py)
parallelises BLAKE3 across CPU cores via `multiprocessing`. At difficulty 32
that's ~2^31 expected hashes (a few minutes on a 12-core CPU).

### Response

```
{ "method":"pearl.challenge_response",
  "params":{ "seed":<original hex>, "nonce":<hex_8_bytes> } }
```

Sent as a **notification** (no `id`) — the binary uses the
`,"params":{"seed":"...","nonce":"..."}` format string at `0x001f3884`.
`nonce` is the 8-byte value as hex (16 chars). Endianness: we send big-endian
display (`struct.pack(">Q", nonce).hex()`) — needs cross-checking once a
solution lands, see TODO at bottom.

## `mining.authorize`

Standard stratum shape:

```
{ "id":N, "method":"mining.authorize",
  "params":["<bech32_addr>.<worker>", "<password>"] }
```

Username format from the CLI help text (`alpha-miner --help`):
`"<address>"` or `"<address>.<worker>"`.

Password follows the open StratumX convention used elsewhere in mining:

* `"x"` — vardiff (server tunes difficulty toward ~1 share/10s/worker).
* `"x;d=<N>"` — pin static share difficulty `N` (recommended for multi-rig
  setups where vardiff oscillates).

Address is a 62-char bech32m starting with `prl1p…`. (Note: `prl1pm2gvh4…secw2u`
embedded in the binary is the **dev-fee address** — do not use it as your own.)

## `mining.set_difficulty`

Standard stratum: `{ "method":"mining.set_difficulty", "params":[<float>] }`.
Sent before the first job and whenever vardiff updates.

## `pearl.set_mining_params`

Pearl-specific. The matrix shape the pool wants you to mine right now.
**`params` is a single-element array wrapping the object**, not a bare object:

```json
{
  "method": "pearl.set_mining_params",
  "params": [{
    "m": 131072,
    "n": 131072,
    "k": 4096,
    "rank": 128,
    "rows_pattern": [0, 32],
    "cols_pattern": [0, 1, 2, ..., 63],
    "mma_type": "Int7xInt7ToInt32"
  }]
}
```

Live capture from `eu1.alphapool.tech:5566` on 2026-05-24:

| Field          | Type     | Example                       | Meaning |
|----------------|----------|-------------------------------|---------|
| `m`, `n`       | uint     | 131072, 131072                | GEMM output dims (rows × cols of C) |
| `k`            | uint     | 4096                          | Inner contraction dim |
| `rank`         | uint     | 128                           | Low-rank factor for the noise step (also a per-thread tile size) |
| `rows_pattern` | uint[]   | `[0, 32]`                     | Sparse row mask within each rank group (2 of 64 rows live) |
| `cols_pattern` | uint[]   | `[0..63]`                     | Sparse col mask (here: all 64 dense) |
| `mma_type`     | string   | `"Int7xInt7ToInt32"`          | Tensor-core data-type recipe; INT7 × INT7 → INT32 accumulate |

The server can change profiles mid-session; the miner logs
`pool_profile_change` / `profile_changed=true` when it happens.

## `mining.notify`

The work unit, sent as a positional array (the binary parses it with
`expected JSON array` / `malformed mining.notify` error path at 0x001f328d).
Confirmed shape from a live capture on `eu1.alphapool.tech:5566`:

```json
{
  "method": "mining.notify",
  "params": [
    "0000e78a-f8b32d6a910ca626",     // job_id (string, "<heightHex>-<rand64>")
    "1a1cf87025c820a6e4b6eee4dcf7f983c5192b7adade2ded0655780ee30ed7f8",
                                      // prev_block_hash (32 bytes hex, BE)
    "00004020f8d70ee30e7855…810318",  // incomplete_header_bytes (hex)
    59274,                            // block_height (uint)
    "6a11fdc5",                       // ntime (u32 hex, BE)
    "1b014f8a",                       // share_nbits (compact target, u32 hex)
    true                              // clean_jobs (bool — drop in-flight work)
  ]
}
```

Notes:

* `incomplete_header_bytes` is the same field as in the solo-gateway
  `submitPlainProof` payload — Pearl block header pre-image, minus the merkle
  roots the miner is about to commit to. Hex-encoded here, base64-encoded on
  the gateway side (see [src/pearl_amd/proto.py](../src/pearl_amd/proto.py)).
* `share_nbits` is the Bitcoin-style compact representation of the target the
  pool wants for a share at the current `mining.set_difficulty` level.
* `prev_block_hash` is the first 32 bytes embedded into `incomplete_header_bytes`
  re-served as a separate field for convenience.
* `job_id` format is `<heightHex(8)>-<rand64Hex(16)>`.

## `mining.submit`

Sent when a share is found. Confirmed shape (2026-05-24 probe with a
deliberately-malformed payload):

```
{ "id":N, "method":"mining.submit",
  "params": [ "<user>", "<job_id>", "<plain_proof_b64>" ] }
```

Exactly three params, all strings. Pool error message from a probe sending
4 params with an integer at position 3:

```
[25, "bad submit params (need [worker, job_id, plain_proof_b64])", null]
```

`<user>` is the same `address.worker` string used in `mining.authorize`.
`<plain_proof_b64>` is the canonical Pearl PlainProof bytes, base64-encoded —
the same wire format the solo gateway uses in
[src/pearl_amd/proto.py](../src/pearl_amd/proto.py) (`PlainProof.to_base64()`).

The pool returns ``{"id":N, "result": true}`` once it has parsed the params;
**that is not a "share accepted" ack** — proof verification runs
asynchronously and surfaces in pool-stats, not in the JSON-RPC response.
Our probe confirmed this by submitting an obviously-bogus payload and still
getting ``result: true``.

The binary also has reconnect/drop bookkeeping for ambiguous shares
(`reconnect_drop_ambiguous_share`, `dropped:true`) — these are pure
client-side state, not part of the wire.

## Gateway (non-stratum) mode — disabled

For completeness: the binary still has the solo-gateway code path
(`--gateway`, `--socket-path`), but the help text says `"gateway/solo mining
flags are temporarily disabled. Use --pool and --address for pool mining."`
The gateway dialect is plain JSON-RPC 2.0 with methods `getMiningInfo` and
`submitPlainProof` — already implemented in
[src/pearl_amd/gateway_client.py](../src/pearl_amd/gateway_client.py).

## Confirmed via live capture (2026-05-24)

All four pieces of the wire are now nailed down against `eu1.alphapool.tech:5566`:

* `pearl.challenge` nonce hex is **big-endian display of u64** (`struct.pack(">Q",
  nonce).hex()`); the BLAKE3 input itself is `seed_bytes_32 || u64_le(nonce)`,
  default (non-keyed) BLAKE3 mode. Verified by satisfying the gate and
  successfully authorizing.
* `pearl.set_mining_params` `params` is a single-element array wrapping the
  object — see schema above with confirmed field types.
* `mining.notify` field order confirmed — see schema above.
* `mining.submit` params confirmed = `[user, job_id, plain_proof_b64]`
  (pool error on a 4-param probe: `"bad submit params (need [worker, job_id, plain_proof_b64])"`).

The remaining unknowns are all in the *content* of `plain_proof_b64`, not the
wire — see the next section for the byte layout.

## PlainProof bytes

The base64-encoded payload of `mining.submit` is `bincode::serialize(&PlainProof)`
using bincode v1's default config: little-endian, fixed-width (`usize` = 8 bytes),
no varints. We do NOT need to run Pearl's CPU `mine()` function to know this —
the layout is fully determined by the Rust struct definitions in
[reference/zk-pow/src/ffi/plain_proof.rs](../reference/zk-pow/src/ffi/plain_proof.rs)
and the `Serialize` derives.

```
PlainProof:
    m              u64 LE        # matrix rows of A (matches pool's set_mining_params.m)
    n              u64 LE        # matrix cols of B
    k              u64 LE        # contraction dim
    noise_rank     u64 LE        # = rank from set_mining_params
    a              MatrixMerkleProof   # commitment to matrix A
    bt             MatrixMerkleProof   # commitment to matrix B^T

MatrixMerkleProof:
    proof          MerkleProof
    row_indices    Vec<u64>      # rows of the matrix touched by this candidate;
                                  # must follow the relevant *_pattern from
                                  # pearl.set_mining_params (rows_pattern for A,
                                  # cols_pattern for B^T)

MerkleProof:
    leaf_data      Vec<Vec<u8>>  # u64 outer count, then per leaf:
                                  #   u64 inner length (== 1024) + 1024 bytes
                                  # (Rust source is Vec<[u8;1024]> but a
                                  #  serde helper wraps each as &[u8], so
                                  #  every leaf gets its own length prefix)
    leaf_indices   Vec<u64>
    total_leaves   u64 LE
    root           [u8; 32]      # raw, no length prefix
    siblings       Vec<[u8; 32]> # u64 count + N*32 bytes

Vec<T> serializes as: u64 LE length || N × T (concatenated).
```

A pure-Python encoder/decoder + a round-trip selftest lives at
[src/pearl_amd/plain_proof_codec.py](../src/pearl_amd/plain_proof_codec.py).
Run it directly (`python -m pearl_amd.plain_proof_codec`) — it builds a tiny
synthetic proof, hand-verifies the size, then re-decodes it byte-identically.

### Semantics (what's actually inside)

Recovered from `parse_plain_proof` in the same file:

* `a.row_indices` are the rows of A in the candidate's tile, drawn from the
  arithmetic progression encoded by `rows_pattern`. The merkle proof commits
  to the actual bytes of those rows in `leaf_data`.
* `bt.row_indices` are similarly the *columns of B* (i.e. rows of B^T) in the
  candidate tile, drawn from `cols_pattern`.
* For each touched row, `leaf_data` holds that row's full `k`-byte payload
  (rounded up to the BLAKE3 chunk boundary of 1024 bytes), padded with
  merkle-internal data.
* The verifier:
  1. Rebuilds the Merkle trees of A and B^T from `leaf_data` + `siblings`,
     checks `root` matches.
  2. Re-hashes the merkle roots keyed with `job_key = blake3(block_header || mining_config)`
     to get `hash_a` / `hash_b`.
  3. Replays the noisy-GEMM BLAKE3 trace at the (rows × cols) intersection
     and checks the resulting `hash_jackpot` is below the target.
* `hash_jackpot < target` is the actual PoW condition. The search the miner
  runs is over the *choice of noise matrices* (E_A, E_B etc.) — not over A
  and B directly, which are committed up front via their merkle roots.

The ZK Plonky2 wrapping happens on the *gateway* side, not the miner side —
miners only need to ship a `PlainProof`. See
[reference/py-pearl-mining/README.md](../reference/py-pearl-mining/README.md)
for the gateway's `generate_proof()` step.
