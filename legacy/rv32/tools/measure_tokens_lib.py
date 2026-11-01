from chipformer.rv import spec as S


def rows_for(text: str) -> list[bytes]:
    rows = [bytes(S.TEXT_ROW)]
    for i in range(0, len(text), 4):
        rows.append(text[i:i + 4].encode().ljust(S.TEXT_ROW, b"\0"))
    rows += [bytes(S.TEXT_ROW)] * (S.TEXT_ROWS - len(rows))
    return rows
