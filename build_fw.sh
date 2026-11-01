#!/bin/bash
# Build the chess firmware for the Qwenputer RV32 machine (rom map) and for qemu-riscv32 (checks).
#   build_fw.sh NODES      -> out/chess-N.elf, out/chess-N.qemu.elf
# Runs the RISC-V toolchain in the rvchess docker image; flags follow the repo's build_calc.sh.
set -euo pipefail
NODES=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=$HOME/qwenputer/experimental/riscv/chipformer/rv/env
mkdir -p "$HERE/out"
docker run --rm -v "$HERE:/w" -v "$ENV:/env:ro" rvchess bash -c "
set -euo pipefail
PICO=/usr/lib/picolibc/riscv64-unknown-elf
GCC=riscv64-unknown-elf-gcc
LIBGCC=\$(\$GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
LIBS=\"-Wl,--start-group \$PICO/lib/release/rv32im/ilp32/libc.a \$LIBGCC -Wl,--end-group\"
CF='-march=rv32im_zba -mabi=ilp32 -O2 -std=c11 -Wall -Wno-unused-function -nostdlib -nostartfiles -static'
\$GCC \$CF -DNODES=$NODES -isystem \$PICO/include -I/env -I/w/src -T/env/rom.ld \
  -o /w/out/chess-$NODES.elf /env/crt0.S /w/src/fw.c /w/src/engine.c /env/sys.c \$LIBS
\$GCC \$CF -DNODES=$NODES -DQEMU -isystem \$PICO/include -I/env -I/w/src -T/env/qemu.ld \
  -o /w/out/chess-$NODES.qemu.elf /env/crt0.S /w/src/fw.c /w/src/engine.c /env/sys.c \$LIBS
riscv64-unknown-elf-size -A /w/out/chess-$NODES.elf | grep -E '^\.(text|rodata|data|sdata|bss|sbss|tbss)'
riscv64-unknown-elf-nm -n /w/out/chess-$NODES.elf | grep -E ' (__heap_start|_end)$'
"
