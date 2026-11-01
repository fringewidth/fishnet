/* A small alpha-beta chess engine written to run on the Qwenputer RV32IM machine (32 KB RAM).
   Portable C with fixed-width types: the RV32 build and a native build make identical moves. */
#pragma once
#include <stdint.h>

typedef struct {
    uint32_t move;  /* 0 when the side to move has no legal move */
    int32_t score;  /* centipawns, side to move's view */
    int32_t depth;  /* last fully completed iteration */
    uint32_t nodes;
} SearchResult;

/* 0 on success, negative on a malformed FEN. */
int32_t eng_set_fen(const char *fen);
SearchResult eng_search(uint32_t node_limit);
/* Writes the UCI string (e.g. "e7e8q") and a NUL; out needs 6 bytes. */
void eng_move_str(uint32_t move, char *out);
uint32_t eng_perft(int32_t depth);
/* Nonzero when the side to move is in check. */
int32_t eng_in_check(void);
/* Mate scores have |score| >= ENG_MATE_BOUND. */
#define ENG_MATE 30000
#define ENG_MATE_BOUND 29000
