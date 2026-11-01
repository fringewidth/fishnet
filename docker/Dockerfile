FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc-riscv64-unknown-elf picolibc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
    qemu-user python3 make gcc stockfish && rm -rf /var/lib/apt/lists/*
