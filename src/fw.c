/* Qwenputer chess firmware. Each run (GO, after </tool_call>) reads the tool call's text from
   the text buffer (the call's i-th token at 0x8000 + 16 i), takes the first parameter's value
   (the FEN: what follows the second '>' up to the next '<', newlines dropped), searches a fixed
   number of nodes, prints the tool response and parks until the next call. Mirrors calc.c's
   bridge loop.
   Built with -DQEMU it instead reads FENs from stdin, one per line, for checking against the
   native build. */
#include <stdint.h>
#include "rt.h"
#include "engine.h"
#include "respond.h"

#ifndef NODES
#error "build with -DNODES=<node limit>"
#endif

static char fen_in[160];

#ifdef QEMU
static void out(int32_t c, void *ctx) {
    (void)ctx;
    rv_putchar((char)c);
}

static int32_t read_line(void) {
    int32_t n = 0;
    for (;;) {
        char c;
        long got = rv_ecall2(63, 0, (long)&c, 1);
        if (got <= 0) return n ? n : -1;
        if (c == '\n') break;
        if (c != '\r' && n < (int32_t)sizeof fen_in - 1) fen_in[n++] = c;
    }
    fen_in[n] = 0;
    return n;
}

int main(void) {
    while (read_line() >= 0) {
        respond(fen_in, NODES, out, 0);
        rv_putchar('\n');
    }
    return 0;
}
#else
#define TEXT ((const volatile unsigned char *)0x8000)
#define ROWS 80
enum { TOOL_RESPONSE = 256, TOOL_RESPONSE_END, THINK, THINK_END };

static void put(int32_t c, void *ctx) {
    (void)ctx;
    rv_ecall2(1, c, 0, 0);
}

static int32_t gather(void) {
    int32_t n = 0, gt = 0, i;
    for (i = 1; i <= ROWS; i++) {
        const volatile unsigned char *r = TEXT + 16 * i;
        int32_t c = r[0], k;
        if (!c) break;
        for (k = 0; k < 16 && c; c = ++k < 16 ? r[k] : 0) {
            if (gt < 2) {
                gt += c == '>';
            } else if (c == '<') {
                fen_in[n] = 0;
                return n;
            } else if (c != '\n' && n < (int32_t)sizeof fen_in - 1) {
                fen_in[n++] = (char)c;
            }
        }
    }
    fen_in[n] = 0;
    if (gt < 2) return -1;
    return i > ROWS ? -2 : n;
}

int main(void) {
    for (;;) {
        int32_t n;
        rv_ecall2(93, 0, 0, 0); /* park: HALT; the next GO resumes here */
        n = gather();
        put('\n', 0);
        put(TOOL_RESPONSE, 0);
        put('\n', 0);
        if (n < 0) put_str(n == -2 ? "error: tool call too long" : "error: no FEN given", put, 0);
        else respond(fen_in, NODES, put, 0);
        put('\n', 0);
        put(TOOL_RESPONSE_END, 0);
        put('\n', 0);
        put(THINK, 0);
        put('\n', 0);
        put('\n', 0);
        put(THINK_END, 0);
        put('\n', 0);
    }
}
#endif
