/* A small alpha-beta chess engine for the Qwenputer RV32IM machine.
 *
 * Board: 0x88 mailbox with piece lists. Evaluation: PeSTO tapered piece-square tables, kept
 * incrementally. Search: iterative deepening PVS with a transposition table, null-move pruning,
 * late-move reductions, check extensions, killer and history ordering, and a capture-only
 * quiescence search. The search stops at a fixed node count, so its result is a pure function
 * of (FEN, node limit): every build plays the same move.
 *
 * Only fixed-width types are used, and plain char is never used for arithmetic (RISC-V char is
 * unsigned, Apple arm64 char is signed).
 */
#include "engine.h"
#include "tables.h"

enum { WHITE = 0, BLACK = 1 };
enum { EMPTY = 0, PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
#define COLOR(p) ((p) >> 3)
#define TYPE(p) ((p) & 7)
#define MAKE_PIECE(c, t) ((uint8_t)(((c) << 3) | (t)))
#define OFF(sq) ((sq) & 0x88)
#define SQ64(sq) (((sq) + ((sq) & 7)) >> 1)
#define NO_SQ 0x88

/* move: from 0-6, to 7-13, promo type 14-16, flag 17-19, ordering score 20-31 */
#define M_FROM(m) ((m) & 0x7F)
#define M_TO(m) (((m) >> 7) & 0x7F)
#define M_PROMO(m) (((m) >> 14) & 7)
#define M_FLAG(m) (((m) >> 17) & 7)
#define M_MASK 0xFFFFFu
#define F_NONE 0
#define F_EP 1
#define F_CASTLE 2
#define F_DOUBLE 3
#define MK(from, to, promo, flag) \
    ((uint32_t)(from) | ((uint32_t)(to) << 7) | ((uint32_t)(promo) << 14) | ((uint32_t)(flag) << 17))

#define MAX_PLY 64
#define MS_SIZE 2048
#define TT_SIZE 1024
#define HIST_SIZE 256
#define INF 32000

static const int8_t KNIGHT_D[8] = {33, 31, 18, 14, -33, -31, -18, -14};
static const int8_t KING_D[8] = {1, -1, 16, -16, 15, 17, -15, -17};
static const int8_t DIAG_D[4] = {15, 17, -15, -17};
static const int8_t ORTH_D[4] = {1, -1, 16, -16};
static const int16_t VALUE[7] = {0, 100, 320, 330, 500, 900, 0};
static const uint8_t PHASE_INC[7] = {0, 0, 1, 1, 2, 4, 0};

typedef struct {
    uint32_t move;
    uint32_t hash;
    int32_t mg, eg;
    uint8_t captured, cap_idx, castle, ep, halfmove, phase;
} Undo;

typedef struct {
    uint16_t key;
    int16_t score;
    uint32_t data; /* move 0-19, bound 20-21, depth 22-29 */
} TTEntry;
enum { TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };

static uint8_t board[128];
static uint8_t plist[2][16];
static uint8_t pidx[128];
static uint8_t king_sq[2];
static uint8_t side, castle, ep, halfmove, phase;
static uint8_t bishops[2];
static uint8_t nonpawn[2];
static int32_t mg, eg;
static uint32_t hash;

static uint32_t ms[MS_SIZE];
static int32_t msp;
static Undo undo[MAX_PLY + 8];
static int32_t uply;
static uint32_t hist[HIST_SIZE];
static int32_t hply;
static TTEntry tt[TT_SIZE];
static uint32_t killers[MAX_PLY][2];
static int16_t history[12][64];

static uint32_t nodes, node_limit;
static int32_t stop;
static uint32_t root_best;

/* ---------- board primitives ---------- */

static int32_t zidx(uint8_t p) { return TYPE(p) - 1 + 6 * COLOR(p); }

static void put_piece(uint8_t p, int32_t sq) {
    int32_t s64 = SQ64(sq);
    board[sq] = p;
    hash ^= ZOB_PIECE[zidx(p)][s64];
    if (COLOR(p) == WHITE) {
        mg += PST_MG[TYPE(p) - 1][s64];
        eg += PST_EG[TYPE(p) - 1][s64];
    } else {
        mg -= PST_MG[TYPE(p) - 1][s64 ^ 56];
        eg -= PST_EG[TYPE(p) - 1][s64 ^ 56];
    }
}

static void take_piece(int32_t sq) {
    uint8_t p = board[sq];
    int32_t s64 = SQ64(sq);
    board[sq] = EMPTY;
    hash ^= ZOB_PIECE[zidx(p)][s64];
    if (COLOR(p) == WHITE) {
        mg -= PST_MG[TYPE(p) - 1][s64];
        eg -= PST_EG[TYPE(p) - 1][s64];
    } else {
        mg += PST_MG[TYPE(p) - 1][s64 ^ 56];
        eg += PST_EG[TYPE(p) - 1][s64 ^ 56];
    }
}

static uint8_t castle_mask(int32_t sq) {
    switch (sq) {
    case 0x00: return 15 & ~2;
    case 0x04: return 15 & ~3;
    case 0x07: return 15 & ~1;
    case 0x70: return 15 & ~8;
    case 0x74: return 15 & ~12;
    case 0x77: return 15 & ~4;
    default: return 15;
    }
}

static int32_t attacked(int32_t sq, int32_t by) {
    int32_t i, t;
    if (by == WHITE) {
        t = sq - 15;
        if (!OFF(t) && board[t] == MAKE_PIECE(WHITE, PAWN)) return 1;
        t = sq - 17;
        if (!OFF(t) && board[t] == MAKE_PIECE(WHITE, PAWN)) return 1;
    } else {
        t = sq + 15;
        if (!OFF(t) && board[t] == MAKE_PIECE(BLACK, PAWN)) return 1;
        t = sq + 17;
        if (!OFF(t) && board[t] == MAKE_PIECE(BLACK, PAWN)) return 1;
    }
    for (i = 0; i < 8; i++) {
        t = sq + KNIGHT_D[i];
        if (!OFF(t) && board[t] == MAKE_PIECE(by, KNIGHT)) return 1;
        t = sq + KING_D[i];
        if (!OFF(t) && board[t] == MAKE_PIECE(by, KING)) return 1;
    }
    for (i = 0; i < 4; i++) {
        int32_t d = DIAG_D[i];
        for (t = sq + d; !OFF(t); t += d) {
            uint8_t p = board[t];
            if (p) {
                if (COLOR(p) == by && (TYPE(p) == BISHOP || TYPE(p) == QUEEN)) return 1;
                break;
            }
        }
        d = ORTH_D[i];
        for (t = sq + d; !OFF(t); t += d) {
            uint8_t p = board[t];
            if (p) {
                if (COLOR(p) == by && (TYPE(p) == ROOK || TYPE(p) == QUEEN)) return 1;
                break;
            }
        }
    }
    return 0;
}

int32_t eng_in_check(void) { return attacked(king_sq[side], side ^ 1); }

/* ---------- make / unmake ---------- */

static void make_move(uint32_t m) {
    Undo *u = &undo[uply++];
    int32_t from = M_FROM(m), to = M_TO(m), flag = M_FLAG(m), promo = M_PROMO(m);
    int32_t c = side, dir = c == WHITE ? 16 : -16;
    int32_t cap_sq = flag == F_EP ? to - dir : to;
    uint8_t p = board[from], cp = board[cap_sq];

    u->move = m;
    u->hash = hash;
    u->mg = mg;
    u->eg = eg;
    u->captured = cp;
    u->castle = castle;
    u->ep = ep;
    u->halfmove = halfmove;
    u->phase = phase;

    if (cp) {
        u->cap_idx = pidx[cap_sq];
        plist[c ^ 1][pidx[cap_sq]] = NO_SQ;
        phase -= PHASE_INC[TYPE(cp)];
        if (TYPE(cp) == BISHOP) bishops[c ^ 1]--;
        if (TYPE(cp) != PAWN) nonpawn[c ^ 1]--;
        take_piece(cap_sq);
    }
    take_piece(from);
    pidx[to] = pidx[from];
    plist[c][pidx[to]] = (uint8_t)to;
    if (promo) {
        put_piece(MAKE_PIECE(c, promo), to);
        phase += PHASE_INC[promo];
        if (promo == BISHOP) bishops[c]++;
        nonpawn[c]++;
    } else {
        put_piece(p, to);
    }
    if (TYPE(p) == KING) {
        king_sq[c] = (uint8_t)to;
        if (flag == F_CASTLE) {
            int32_t rf = to > from ? from + 3 : from - 4, rt = to > from ? from + 1 : from - 1;
            uint8_t r = board[rf];
            take_piece(rf);
            put_piece(r, rt);
            pidx[rt] = pidx[rf];
            plist[c][pidx[rt]] = (uint8_t)rt;
        }
    }
    hash ^= ZOB_CASTLE[castle];
    castle &= castle_mask(from) & castle_mask(to);
    hash ^= ZOB_CASTLE[castle];
    if (ep != NO_SQ) hash ^= ZOB_EP[ep & 7];
    ep = flag == F_DOUBLE ? (uint8_t)(from + dir) : NO_SQ;
    if (ep != NO_SQ) hash ^= ZOB_EP[ep & 7];
    halfmove = (TYPE(p) == PAWN || cp) ? 0 : (uint8_t)(halfmove < 255 ? halfmove + 1 : 255);
    side ^= 1;
    hash ^= ZOB_SIDE;
    hist[hply++] = hash;
}

static void unmake_move(void) {
    Undo *u = &undo[--uply];
    uint32_t m = u->move;
    int32_t from = M_FROM(m), to = M_TO(m), flag = M_FLAG(m);
    int32_t c, dir, cap_sq;
    uint8_t p;

    hply--;
    side ^= 1;
    c = side;
    dir = c == WHITE ? 16 : -16;
    p = M_PROMO(m) ? MAKE_PIECE(c, PAWN) : board[to];
    board[to] = EMPTY;
    board[from] = p;
    pidx[from] = pidx[to];
    plist[c][pidx[from]] = (uint8_t)from;
    if (M_PROMO(m)) {
        if (M_PROMO(m) == BISHOP) bishops[c]--;
        nonpawn[c]--;
    }
    if (TYPE(p) == KING) {
        king_sq[c] = (uint8_t)from;
        if (flag == F_CASTLE) {
            int32_t rf = to > from ? from + 3 : from - 4, rt = to > from ? from + 1 : from - 1;
            board[rf] = board[rt];
            board[rt] = EMPTY;
            pidx[rf] = pidx[rt];
            plist[c][pidx[rf]] = (uint8_t)rf;
        }
    }
    if (u->captured) {
        cap_sq = flag == F_EP ? to - dir : to;
        board[cap_sq] = u->captured;
        pidx[cap_sq] = u->cap_idx;
        plist[c ^ 1][u->cap_idx] = (uint8_t)cap_sq;
        if (TYPE(u->captured) == BISHOP) bishops[c ^ 1]++;
        if (TYPE(u->captured) != PAWN) nonpawn[c ^ 1]++;
    }
    hash = u->hash;
    mg = u->mg;
    eg = u->eg;
    castle = u->castle;
    ep = u->ep;
    halfmove = u->halfmove;
    phase = u->phase;
}

static void make_null(void) {
    Undo *u = &undo[uply++];
    u->move = 0;
    u->hash = hash;
    u->ep = ep;
    u->halfmove = halfmove;
    if (ep != NO_SQ) hash ^= ZOB_EP[ep & 7];
    ep = NO_SQ;
    halfmove = 0;
    side ^= 1;
    hash ^= ZOB_SIDE;
    hist[hply++] = hash;
}

static void unmake_null(void) {
    Undo *u = &undo[--uply];
    hply--;
    side ^= 1;
    hash = u->hash;
    ep = u->ep;
    halfmove = u->halfmove;
}

/* ---------- move generation (pseudo-legal) ---------- */

static void add(uint32_t m) { ms[msp++] = m; }

static void add_promos(int32_t from, int32_t to, int32_t caps_only) {
    add(MK(from, to, QUEEN, F_NONE));
    if (caps_only) return;
    add(MK(from, to, KNIGHT, F_NONE));
    add(MK(from, to, ROOK, F_NONE));
    add(MK(from, to, BISHOP, F_NONE));
}

static void gen_moves(int32_t caps_only) {
    int32_t c = side, them = c ^ 1, i, k;
    int32_t dir = c == WHITE ? 16 : -16;
    int32_t start_rank = c == WHITE ? 1 : 6, last_rank = c == WHITE ? 7 : 0;
    for (i = 0; i < 16; i++) {
        int32_t from = plist[c][i], to, t;
        uint8_t p;
        if (from == NO_SQ) continue;
        p = board[from];
        switch (TYPE(p)) {
        case PAWN:
            to = from + dir;
            if (!board[to]) {
                if ((to >> 4) == last_rank) {
                    add_promos(from, to, caps_only);
                } else if (!caps_only) {
                    add(MK(from, to, 0, F_NONE));
                    if ((from >> 4) == start_rank && !board[to + dir]) add(MK(from, to + dir, 0, F_DOUBLE));
                }
            }
            for (k = -1; k <= 1; k += 2) {
                t = from + dir + k;
                if (OFF(t)) continue;
                if (board[t] && COLOR(board[t]) == them) {
                    if ((t >> 4) == last_rank) add_promos(from, t, caps_only);
                    else add(MK(from, t, 0, F_NONE));
                } else if (t == ep) {
                    add(MK(from, t, 0, F_EP));
                }
            }
            break;
        case KNIGHT:
        case KING: {
            const int8_t *d = TYPE(p) == KNIGHT ? KNIGHT_D : KING_D;
            for (k = 0; k < 8; k++) {
                t = from + d[k];
                if (OFF(t)) continue;
                if (!board[t]) {
                    if (!caps_only) add(MK(from, t, 0, F_NONE));
                } else if (COLOR(board[t]) == them) {
                    add(MK(from, t, 0, F_NONE));
                }
            }
            break;
        }
        default: {
            int32_t ty = TYPE(p);
            for (k = 0; k < 8; k++) {
                int32_t d;
                if (k < 4) {
                    if (ty == ROOK) continue;
                    d = DIAG_D[k];
                } else {
                    if (ty == BISHOP) continue;
                    d = ORTH_D[k - 4];
                }
                for (t = from + d; !OFF(t); t += d) {
                    if (!board[t]) {
                        if (!caps_only) add(MK(from, t, 0, F_NONE));
                        continue;
                    }
                    if (COLOR(board[t]) == them) add(MK(from, t, 0, F_NONE));
                    break;
                }
            }
        }
        }
    }
    if (caps_only) return;
    /* castling: the king is not in check and does not pass through an attacked square */
    if (c == WHITE) {
        if ((castle & 1) && !board[0x05] && !board[0x06] && board[0x07] == MAKE_PIECE(WHITE, ROOK) &&
            !attacked(0x04, BLACK) && !attacked(0x05, BLACK) && !attacked(0x06, BLACK))
            add(MK(0x04, 0x06, 0, F_CASTLE));
        if ((castle & 2) && !board[0x03] && !board[0x02] && !board[0x01] &&
            board[0x00] == MAKE_PIECE(WHITE, ROOK) && !attacked(0x04, BLACK) && !attacked(0x03, BLACK) &&
            !attacked(0x02, BLACK))
            add(MK(0x04, 0x02, 0, F_CASTLE));
    } else {
        if ((castle & 4) && !board[0x75] && !board[0x76] && board[0x77] == MAKE_PIECE(BLACK, ROOK) &&
            !attacked(0x74, WHITE) && !attacked(0x75, WHITE) && !attacked(0x76, WHITE))
            add(MK(0x74, 0x76, 0, F_CASTLE));
        if ((castle & 8) && !board[0x73] && !board[0x72] && !board[0x71] &&
            board[0x70] == MAKE_PIECE(BLACK, ROOK) && !attacked(0x74, WHITE) && !attacked(0x73, WHITE) &&
            !attacked(0x72, WHITE))
            add(MK(0x74, 0x72, 0, F_CASTLE));
    }
}

/* ---------- evaluation ---------- */

static int32_t evaluate(void) {
    int32_t ph = phase > 24 ? 24 : phase;
    int32_t s = (mg * ph + eg * (24 - ph)) / 24;
    if (bishops[WHITE] >= 2) s += 40;
    if (bishops[BLACK] >= 2) s -= 40;
    s = side == WHITE ? s : -s;
    return s + 10; /* tempo */
}

/* ---------- search ---------- */

/* Can this pseudo-legal move leave the mover's king attacked when the mover is not in check?
   Only if the king moves, the move is en passant, or the piece leaves a line through its king. */
static int32_t may_expose(uint32_t m) {
    int32_t from = M_FROM(m);
    if (TYPE(board[from]) == KING || M_FLAG(m) == F_EP) return 1;
    return ALIGN[from - king_sq[side] + 119];
}

static int32_t is_repetition(void) {
    int32_t i, lim = hply - 1 - halfmove;
    if (lim < 0) lim = 0;
    for (i = hply - 3; i >= lim; i -= 2)
        if (hist[i] == hash) return 1;
    return 0;
}

static uint32_t mvv_lva(uint32_t m) {
    int32_t to = M_TO(m);
    int32_t victim = M_FLAG(m) == F_EP ? PAWN : TYPE(board[to]);
    int32_t attacker = TYPE(board[M_FROM(m)]);
    return (uint32_t)(victim * 8 + (7 - attacker));
}

static uint32_t pick(int32_t i, int32_t end) {
    int32_t j, best = i;
    uint32_t m;
    for (j = i + 1; j < end; j++)
        if ((ms[j] >> 20) > (ms[best] >> 20)) best = j;
    m = ms[best];
    ms[best] = ms[i];
    ms[i] = m;
    return m;
}

static int32_t tick(void) {
    if (++nodes >= node_limit) stop = 1;
    return stop;
}

static int32_t qsearch(int32_t alpha, int32_t beta, int32_t ply) {
    int32_t stand, i, start, end, best, in_check = -1, verify;
    if (tick()) return 0;
    stand = evaluate();
    if (stand >= beta) return stand;
    if (ply >= MAX_PLY || msp + 256 > MS_SIZE) return stand;
    if (stand > alpha) alpha = stand;
    best = stand;
    start = msp;
    gen_moves(1);
    end = msp;
    for (i = start; i < end; i++) {
        uint32_t s = mvv_lva(ms[i]) + (M_PROMO(ms[i]) ? 64 : 0);
        ms[i] = (ms[i] & M_MASK) | (s << 20);
    }
    for (i = start; i < end; i++) {
        uint32_t m = pick(i, end) & M_MASK;
        int32_t sc, victim = M_FLAG(m) == F_EP ? PAWN : TYPE(board[M_TO(m)]);
        if (!M_PROMO(m) && stand + VALUE[victim] + 200 < alpha) continue; /* delta pruning */
        if (in_check < 0) in_check = attacked(king_sq[side], side ^ 1);
        verify = in_check || may_expose(m);
        make_move(m);
        if (verify && attacked(king_sq[side ^ 1], side)) {
            unmake_move();
            continue;
        }
        sc = -qsearch(-beta, -alpha, ply + 1);
        unmake_move();
        if (stop) break;
        if (sc > best) {
            best = sc;
            if (sc > alpha) {
                alpha = sc;
                if (sc >= beta) break;
            }
        }
    }
    msp = start;
    return best;
}

static int32_t tt_to(int32_t s, int32_t ply) {
    if (s >= ENG_MATE_BOUND) return s + ply;
    if (s <= -ENG_MATE_BOUND) return s - ply;
    return s;
}

static int32_t tt_from(int32_t s, int32_t ply) {
    if (s >= ENG_MATE_BOUND) return s - ply;
    if (s <= -ENG_MATE_BOUND) return s + ply;
    return s;
}

static int32_t search(int32_t depth, int32_t alpha, int32_t beta, int32_t ply, int32_t do_null) {
    int32_t in_check, pv = beta - alpha > 1, i, start, end, legal = 0, best = -INF, alpha0 = alpha;
    uint32_t tt_move = 0, best_move = 0;
    TTEntry *e;

    if (ply && (halfmove >= 100 || is_repetition())) return 0;
    if (ply >= MAX_PLY || msp + 256 > MS_SIZE) return evaluate();
    in_check = attacked(king_sq[side], side ^ 1);
    if (in_check) depth++;
    if (depth <= 0) return qsearch(alpha, beta, ply);
    if (tick()) return 0;

    e = &tt[hash & (TT_SIZE - 1)];
    if (e->key == (uint16_t)(hash >> 16) && e->data) {
        int32_t ed = (int32_t)((e->data >> 22) & 0xFF), bound = (int32_t)((e->data >> 20) & 3);
        int32_t es = tt_from(e->score, ply);
        tt_move = e->data & M_MASK;
        if (ply && !pv && ed >= depth) {
            if (bound == TT_EXACT) return es;
            if (bound == TT_LOWER && es >= beta) return es;
            if (bound == TT_UPPER && es <= alpha) return es;
        }
    }

    if (!pv && !in_check && do_null && depth >= 3 && nonpawn[side] && evaluate() >= beta) {
        int32_t sc, r = depth >= 6 ? 3 : 2;
        make_null();
        sc = -search(depth - 1 - r, -beta, -beta + 1, ply + 1, 0);
        unmake_null();
        if (stop) return 0;
        if (sc >= beta) return sc >= ENG_MATE_BOUND ? beta : sc;
    }

    start = msp;
    gen_moves(0);
    end = msp;
    for (i = start; i < end; i++) {
        uint32_t m = ms[i] & M_MASK, s;
        int32_t cap = board[M_TO(m)] || M_FLAG(m) == F_EP;
        if (m == tt_move) s = 4000;
        else if (cap || M_PROMO(m) == QUEEN) s = 3000 + mvv_lva(m) + (M_PROMO(m) ? 64 : 0);
        else if (ply < MAX_PLY && m == killers[ply][0]) s = 2900;
        else if (ply < MAX_PLY && m == killers[ply][1]) s = 2800;
        else {
            int32_t h = history[zidx(board[M_FROM(m)])][SQ64(M_TO(m))];
            s = (uint32_t)(h > 2700 ? 2700 : h);
        }
        ms[i] = m | (s << 20);
    }
    for (i = start; i < end; i++) {
        uint32_t m = pick(i, end) & M_MASK;
        int32_t sc, quiet, gives_check, verify, r = 0;
        quiet = !board[M_TO(m)] && M_FLAG(m) != F_EP && !M_PROMO(m);
        verify = in_check || may_expose(m);
        make_move(m);
        if (verify && attacked(king_sq[side ^ 1], side)) {
            unmake_move();
            continue;
        }
        legal++;
        if (legal == 1) {
            sc = -search(depth - 1, -beta, -alpha, ply + 1, 1);
        } else {
            if (depth >= 3 && quiet && legal > 3 && !in_check) {
                gives_check = attacked(king_sq[side], side ^ 1);
                if (!gives_check) r = legal > 8 ? 2 : 1;
            }
            sc = -search(depth - 1 - r, -alpha - 1, -alpha, ply + 1, 1);
            if (!stop && sc > alpha && r) sc = -search(depth - 1, -alpha - 1, -alpha, ply + 1, 1);
            if (!stop && sc > alpha && sc < beta) sc = -search(depth - 1, -beta, -alpha, ply + 1, 1);
        }
        unmake_move();
        if (stop) break;
        if (sc > best) {
            best = sc;
            best_move = m;
            if (sc > alpha) {
                alpha = sc;
                if (!ply) root_best = m;
                if (sc >= beta) {
                    if (quiet && ply < MAX_PLY) {
                        int16_t *h = &history[zidx(board[M_FROM(m)])][SQ64(M_TO(m))];
                        int32_t nh = *h + depth * depth;
                        *h = (int16_t)(nh > 2700 ? 2700 : nh);
                        if (killers[ply][0] != m) {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = m;
                        }
                    }
                    break;
                }
            }
        }
    }
    msp = start;
    if (stop) return 0;
    if (!legal) return in_check ? -ENG_MATE + ply : 0;

    e->key = (uint16_t)(hash >> 16);
    e->score = (int16_t)tt_to(best, ply);
    e->data = best_move | ((uint32_t)(best >= beta ? TT_LOWER : best > alpha0 ? TT_EXACT : TT_UPPER) << 20) |
              ((uint32_t)(depth > 255 ? 255 : depth) << 22);
    return best;
}

static void clear_search_state(void) {
    int32_t i, j;
    for (i = 0; i < TT_SIZE; i++) {
        tt[i].key = 0;
        tt[i].score = 0;
        tt[i].data = 0;
    }
    for (i = 0; i < MAX_PLY; i++) {
        killers[i][0] = 0;
        killers[i][1] = 0;
    }
    for (i = 0; i < 12; i++)
        for (j = 0; j < 64; j++) history[i][j] = 0;
}

SearchResult eng_search(uint32_t limit) {
    SearchResult r;
    int32_t d;
    r.move = 0;
    r.score = 0;
    r.depth = 0;
    nodes = 0;
    node_limit = limit;
    stop = 0;
    root_best = 0;
    msp = 0;
    clear_search_state();

    /* no legal move: mate or stalemate; one legal move: play it without searching */
    {
        int32_t i, legal = 0;
        uint32_t only = 0;
        gen_moves(0);
        for (i = 0; i < msp; i++) {
            make_move(ms[i] & M_MASK);
            if (!attacked(king_sq[side ^ 1], side)) {
                legal++;
                only = ms[i] & M_MASK;
            }
            unmake_move();
        }
        msp = 0;
        if (legal <= 1) {
            r.move = only;
            r.score = legal ? evaluate() : (eng_in_check() ? -ENG_MATE : 0);
            r.nodes = 0;
            return r;
        }
    }

    for (d = 1; d < MAX_PLY; d++) {
        int32_t sc = search(d, -INF, INF, 0, 0);
        if (stop) {
            if (root_best) r.move = root_best;
            break;
        }
        r.move = root_best;
        r.score = sc;
        r.depth = d;
        if ((sc >= ENG_MATE_BOUND || sc <= -ENG_MATE_BOUND) && d >= ENG_MATE - (sc < 0 ? -sc : sc) + 1) break;
    }
    r.nodes = nodes;
    return r;
}

/* ---------- FEN, perft, formatting ---------- */

int32_t eng_set_fen(const char *fen) {
    int32_t i, rank = 7, file = 0, n[2] = {0, 0};
    const uint8_t *s = (const uint8_t *)fen;
    for (i = 0; i < 128; i++) {
        board[i] = EMPTY;
        pidx[i] = 0;
    }
    for (i = 0; i < 16; i++) {
        plist[0][i] = NO_SQ;
        plist[1][i] = NO_SQ;
    }
    mg = eg = 0;
    hash = 0;
    phase = 0;
    bishops[0] = bishops[1] = 0;
    nonpawn[0] = nonpawn[1] = 0;
    king_sq[0] = king_sq[1] = NO_SQ;
    while (*s == ' ') s++;
    for (; *s && *s != ' '; s++) {
        int32_t c, t, sq;
        uint8_t ch = *s;
        if (ch == '/') {
            if (file != 8) return -1;
            rank--;
            file = 0;
            continue;
        }
        if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            if (file > 8) return -1;
            continue;
        }
        c = (ch >= 'a' && ch <= 'z') ? BLACK : WHITE;
        switch (ch | 32) {
        case 'p': t = PAWN; break;
        case 'n': t = KNIGHT; break;
        case 'b': t = BISHOP; break;
        case 'r': t = ROOK; break;
        case 'q': t = QUEEN; break;
        case 'k': t = KING; break;
        default: return -1;
        }
        if (rank < 0 || file > 7) return -1;
        sq = rank * 16 + file++;
        if (t == KING) {
            if (king_sq[c] != NO_SQ) return -1;
            king_sq[c] = (uint8_t)sq;
        }
        put_piece(MAKE_PIECE(c, t), sq);
        phase += PHASE_INC[t];
        if (t == BISHOP) bishops[c]++;
        if (t != PAWN && t != KING) nonpawn[c]++;
    }
    if (rank != 0 || file != 8 || king_sq[0] == NO_SQ || king_sq[1] == NO_SQ) return -1;
    for (i = 0; i < 128; i++) {
        int32_t c;
        if (OFF(i) || !board[i]) continue;
        c = COLOR(board[i]);
        if (TYPE(board[i]) == KING) continue;
        if (n[c] == 0) n[c] = 1;
        if (n[c] >= 16) return -1;
        plist[c][n[c]] = (uint8_t)i;
        pidx[i] = (uint8_t)n[c]++;
    }
    for (i = 0; i < 2; i++) {
        plist[i][0] = king_sq[i];
        pidx[king_sq[i]] = 0;
    }
    while (*s == ' ') s++;
    if (*s == 'w') side = WHITE;
    else if (*s == 'b') side = BLACK;
    else return -1;
    s++;
    while (*s == ' ') s++;
    castle = 0;
    for (; *s && *s != ' '; s++) {
        if (*s == 'K') castle |= 1;
        else if (*s == 'Q') castle |= 2;
        else if (*s == 'k') castle |= 4;
        else if (*s == 'q') castle |= 8;
        else if (*s != '-') return -1;
    }
    /* drop rights the board cannot support */
    if (board[0x04] != MAKE_PIECE(WHITE, KING)) castle &= ~3;
    if (board[0x07] != MAKE_PIECE(WHITE, ROOK)) castle &= ~1;
    if (board[0x00] != MAKE_PIECE(WHITE, ROOK)) castle &= ~2;
    if (board[0x74] != MAKE_PIECE(BLACK, KING)) castle &= ~12;
    if (board[0x77] != MAKE_PIECE(BLACK, ROOK)) castle &= ~4;
    if (board[0x70] != MAKE_PIECE(BLACK, ROOK)) castle &= ~8;
    while (*s == ' ') s++;
    ep = NO_SQ;
    if (*s >= 'a' && *s <= 'h' && (s[1] == '3' || s[1] == '6')) {
        ep = (uint8_t)((s[1] - '1') * 16 + (s[0] - 'a'));
        s += 2;
    } else if (*s == '-') {
        s++;
    } else if (*s) {
        return -1;
    }
    while (*s == ' ') s++;
    halfmove = 0;
    while (*s >= '0' && *s <= '9') {
        int32_t h = halfmove * 10 + (*s++ - '0');
        halfmove = (uint8_t)(h > 255 ? 255 : h);
    }
    /* the side not to move must not be in check */
    if (attacked(king_sq[side ^ 1], side)) return -2;
    hash ^= ZOB_CASTLE[castle];
    if (ep != NO_SQ) hash ^= ZOB_EP[ep & 7];
    if (side == BLACK) hash ^= ZOB_SIDE;
    hply = 0;
    uply = 0;
    hist[hply++] = hash;
    return 0;
}

static uint32_t perft(int32_t depth) {
    int32_t i, start = msp, end, in_check = attacked(king_sq[side], side ^ 1);
    uint32_t n = 0;
    gen_moves(0);
    end = msp;
    for (i = start; i < end; i++) {
        uint32_t m = ms[i] & M_MASK;
        int32_t verify = in_check || may_expose(m); /* the search's legality rule */
        make_move(m);
        if (!verify || !attacked(king_sq[side ^ 1], side)) n += depth <= 1 ? 1 : perft(depth - 1);
        unmake_move();
    }
    msp = start;
    return n;
}

uint32_t eng_perft(int32_t depth) {
    msp = 0;
    return depth <= 0 ? 1 : perft(depth);
}

void eng_move_str(uint32_t m, char *out) {
    int32_t from = M_FROM(m), to = M_TO(m), promo = M_PROMO(m);
    out[0] = (char)('a' + (from & 7));
    out[1] = (char)('1' + (from >> 4));
    out[2] = (char)('a' + (to & 7));
    out[3] = (char)('1' + (to >> 4));
    out[4] = promo ? "  nbrq"[promo] : 0;
    out[5] = 0;
}
