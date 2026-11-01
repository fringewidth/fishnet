/* The tool response text, shared by the firmware and the native build. */
#pragma once
#include <stdint.h>
#include "engine.h"

typedef void (*PutFn)(int32_t c, void *ctx);

static void put_str(const char *s, PutFn put, void *ctx) {
    while (*s) put((uint8_t)*s++, ctx);
}

static void put_int(int32_t v, PutFn put, void *ctx) {
    char buf[12];
    int32_t n = 0;
    uint32_t u = v < 0 ? (uint32_t)(-(v + 1)) + 1u : (uint32_t)v;
    if (v < 0) put('-', ctx);
    do {
        buf[n++] = (char)('0' + u % 10u);
        u /= 10u;
    } while (u);
    while (n) put((uint8_t)buf[--n], ctx);
}

/* "bestmove e2e4\ndepth 7, eval +35 cp" (or "mate in N"); errors start with "error". */
static void respond(const char *fen, uint32_t limit, PutFn put, void *ctx) {
    char mv[6];
    SearchResult r;
    int32_t err = eng_set_fen(fen);
    if (err) {
        put_str("error: invalid FEN", put, ctx);
        return;
    }
    r = eng_search(limit);
    if (!r.move) {
        put_str(eng_in_check() ? "no legal moves: checkmate" : "no legal moves: stalemate", put, ctx);
        return;
    }
    eng_move_str(r.move, mv);
    put_str("bestmove ", put, ctx);
    put_str(mv, put, ctx);
    put_str("\ndepth ", put, ctx);
    put_int(r.depth, put, ctx);
    if (r.score >= ENG_MATE_BOUND || r.score <= -ENG_MATE_BOUND) {
        int32_t plies = ENG_MATE - (r.score < 0 ? -r.score : r.score);
        put_str(r.score > 0 ? ", mate in " : ", mated in ", put, ctx);
        put_int((plies + 1) / 2, put, ctx);
    } else {
        put_str(", eval ", put, ctx);
        if (r.score >= 0) put('+', ctx);
        put_int(r.score, put, ctx);
        put_str(" cp", put, ctx);
    }
}
