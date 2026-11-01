/* Host driver for the engine. Same engine.c as the firmware.
     chess_native perft DEPTH FEN
     chess_native search NODES        (FENs on stdin, one per line; prints the tool response) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "respond.h"

static void out(int32_t c, void *ctx) {
    (void)ctx;
    putchar(c);
}

int main(int argc, char **argv) {
    char line[512];
    if (argc == 4 && strcmp(argv[1], "perft") == 0) {
        int32_t err = eng_set_fen(argv[3]);
        if (err) {
            printf("bad fen %d\n", err);
            return 1;
        }
        printf("%u\n", eng_perft(atoi(argv[2])));
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "search") == 0) {
        uint32_t limit = (uint32_t)strtoul(argv[2], NULL, 10);
        while (fgets(line, sizeof line, stdin)) {
            line[strcspn(line, "\r\n")] = 0;
            respond(line, limit, out, NULL);
            putchar('\n');
            fflush(stdout);
        }
        return 0;
    }
    fprintf(stderr, "usage: chess_native perft DEPTH FEN | search NODES < fens\n");
    return 2;
}
