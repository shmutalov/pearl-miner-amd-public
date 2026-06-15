"""Compare the shader's DEBUG_MSG output (out.bin = first 8 jackpot_msg words
per cand) against the CPU oracle's jackpot_msg. Isolates jackpot-compute bugs
from BLAKE3 bugs."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pearl_amd.jackpot import compute_jackpot, compute_noise_for_indices  # noqa: E402
from src.pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key  # noqa
from src.pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed  # noqa
import blake3  # noqa

d = Path(sys.argv[1])
meta = dict(zip("m n k r h w batch nca ncb".split(), map(int, (d/"meta.txt").read_text().split())))
m,n,k,r = meta["m"],meta["n"],meta["k"],meta["r"]
mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0,32]),
    cols_pattern=PeriodicPattern.from_list(list(range(64))))
job_key = compute_job_key(bytes(range(76)), mc)
A,B = derive_AB_from_seed(bytes(32), m, n, k)
hash_a = merkle_root_keyed(A, job_key); hash_b = merkle_root_keyed(B, job_key)
b_seed = blake3.blake3(job_key+hash_b).digest(); a_seed = blake3.blake3(b_seed+hash_a).digest()
commit = (b_seed, a_seed)

t_rows = np.fromfile(d/"t_rows.bin", dtype=np.int32)
t_cols = np.fromfile(d/"t_cols.bin", dtype=np.int32)
got = np.fromfile(d/"out.bin", dtype=np.uint32).reshape(-1, 8)

ncheck = min(8, len(t_rows))
bad = 0
for i in range(ncheck):
    tr, tc = int(t_rows[i]), int(t_cols[i])
    a_rows = list(mc.rows_pattern.indices_with_offset(tr))
    b_cols = list(mc.cols_pattern.indices_with_offset(tc))
    noise_a, noise_b = compute_noise_for_indices(k, r, commit, a_rows, b_cols)
    sa = A[a_rows][:, :k]; sb = B[b_cols][:, :k]
    msg = compute_jackpot(sa, sb, noise_a, noise_b, k, r)  # (16,) uint32
    exp8 = msg[:8]
    g8 = got[i]
    ok = np.array_equal(exp8, g8)
    if not ok:
        bad += 1
        print(f"cand {i} (tr={tr},tc={tc}) MISMATCH")
        print(f"  exp: {[hex(int(x)) for x in exp8]}")
        print(f"  got: {[hex(int(x)) for x in g8]}")
    else:
        print(f"cand {i}: msg ok  {[hex(int(x)) for x in exp8]}")
print(f"\n{ncheck-bad}/{ncheck} jackpot_msg match")
