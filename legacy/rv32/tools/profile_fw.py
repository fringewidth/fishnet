"""Per-function instruction profile of the firmware on the reference RV32 machine.
    profile_fw.py ELF FEN"""
import bisect
import collections
import subprocess
import sys

from chipformer.rv import elf, ref
from chipformer.rv import spec as S
from chipformer.rv.bridge import park

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from measure_tokens_lib import rows_for  # noqa: E402

path, fen = sys.argv[1], sys.argv[2]
syms = []
for line in subprocess.run(["docker", "run", "--rm", "-v", f"{path.rsplit('/', 1)[0]}:/o:ro", "rvchess",
                            "riscv64-unknown-elf-nm", "-n", "/o/" + path.rsplit("/", 1)[1]],
                           capture_output=True, text=True, check=True).stdout.splitlines():
    a, t, name = line.split()
    if t.lower() == "t":
        syms.append((int(a, 16), name))
addrs = [a for a, _ in syms]
m = ref.Machine.from_program(elf.program(path))
park(m, max_records=10**7)
m.text = rows_for(f"\n<function=chess>\n<parameter=fen>\n{fen}\n</parameter>\n</function>\n")
cpu = m.cpu if hasattr(m, "cpu") else None
counts = collections.Counter()
toks_by = collections.Counter()
orig = m.record
while not m.halted:
    pc = S.ROM_BASE + 4 * (m.anchor + m.n)
    i = bisect.bisect_right(addrs, pc) - 1
    name = syms[i][1] if i >= 0 else "?"
    r = orig()
    counts[name] += 1
    toks_by[name] += len(r)
tot = sum(toks_by.values())
for name, c in counts.most_common(14):
    print(f"{name:24s} instr {c:9d}  tokens {toks_by[name]:9d}  {100 * toks_by[name] / tot:5.1f}%")
