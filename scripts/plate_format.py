"""Turn raw OCR text into a valid Malaysian plate.

OCR confuses look-alike characters (0/O/Q/D, 1/I/W, 5/S, 8/B ...) and
sometimes adds a stray character from the plate frame. Malaysian plates
have a fixed shape, so we search for the plate that fits that shape with
the lowest "repair cost":

    standard:  1-3 letters + number 1-9999 + 0-2 letters    e.g. WXY 1234 A
    special:   a known word + number + optional letter       e.g. GOLD 1460, 1M4U 792

Try it:  python plate_format.py "IB2744A"   ->   WB 2744 A
"""
import itertools
import re
import sys
from dataclasses import dataclass

VALID_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")  # Malaysian plates never use I or O
SPECIAL_WORDS = ["PUTRAJAYA", "PATRIOT", "MALAYSIA", "PERODUA", "PROTON", "PERFECT",
                 "GOLD", "IMAN", "SUKOM", "1M4U", "G1M", "VIP", "LIMO", "US"]

# What a misread character could really be, with a cost (lower = more likely).
AS_LETTER = {
    "0": [("Q", 1.0), ("D", 1.1)], "O": [("Q", 1.0), ("D", 1.1)],
    "1": [("W", 1.0), ("L", 1.1)], "I": [("W", 1.0), ("V", 1.05), ("L", 1.1)],
    "2": [("Z", 1.0)], "4": [("A", 1.0)], "5": [("S", 1.0)], "6": [("G", 1.0)],
    "7": [("T", 1.0)], "8": [("B", 1.0)],
}
AS_DIGIT = {
    "O": [("0", 1.0)], "D": [("0", 1.0)], "Q": [("0", 1.0)], "U": [("0", 1.2)],
    "I": [("1", 1.0)], "L": [("1", 1.0)], "T": [("7", 1.0)], "Z": [("2", 1.0)],
    "S": [("5", 1.0)], "B": [("8", 1.0)], "G": [("6", 1.0)], "A": [("4", 1.0)],
}
# Extra look-alikes used only when matching a special word such as GOLD or 1M4U.
SAME_SHAPE = {("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("A", "4"), ("4", "A"),
              ("C", "G"), ("G", "C"), ("Q", "O"), ("D", "O")}
DROP_COST = 1.5         # cost of ignoring one stray character
FIRST_DROP_EXTRA = 0.3  # stray characters are rarely at the very start
SUFFIX_EXTRA = 0.6      # a guessed suffix letter is less likely than a guessed prefix letter
SPECIAL_SWAP = 0.8      # cost of one look-alike swap inside a special word


@dataclass(frozen=True)
class Plate:
    text: str      # "WB2744A"
    display: str   # "WB 2744 A"
    kind: str      # "standard" or "special"
    cost: float    # 0 = read exactly as OCR saw it

    @property
    def edits(self) -> int:
        return round(self.cost)


def _letter(c):
    """Cheapest letter this character could be: (letter, cost) or None."""
    if c in VALID_LETTERS:
        return c, 0.0
    return AS_LETTER.get(c, [None])[0]


def _digit(c):
    if c.isdigit():
        return c, 0.0
    return AS_DIGIT.get(c, [None])[0]


def _run(chars, convert):
    """Convert every char with `convert`; return (text, cost) or None."""
    out, cost = [], 0.0
    for c in chars:
        r = convert(c)
        if r is None:
            return None
        out.append(r[0])
        cost += r[1]
    return "".join(out), cost


def _parse_tail(s, start):
    """Best number (1-4 digits, no leading 0) + 0-2 suffix letters from s[start:]."""
    best = None
    for n_len in range(1, 5):
        number = _run(s[start:start + n_len], _digit)
        if number is None or len(number[0]) != n_len or number[0][0] == "0":
            continue
        suffix = _run(s[start + n_len:], _letter)
        if suffix is None or len(suffix[0]) > 2:
            continue
        guessed = sum(a != b for a, b in zip(suffix[0], s[start + n_len:]))
        cost = number[1] + suffix[1] + SUFFIX_EXTRA * guessed
        if best is None or cost < best[2]:
            best = (number[0], suffix[0], cost)
    return best


def _parse(s):
    """Best Plate for string s (no deletions), or None."""
    best = None
    # Special series: a known word at the start.
    for word in SPECIAL_WORDS:
        head = s[:len(word)]
        if len(head) != len(word):
            continue
        cost = 0.0
        for got, want in zip(head, word):
            if got != want:
                cost += SPECIAL_SWAP if (got, want) in SAME_SHAPE else 99
        tail = _parse_tail(s, len(word))
        if cost < 99 and tail and len(tail[1]) <= 1:
            total = cost + tail[2]
            if best is None or total < best.cost:
                best = Plate(word + tail[0] + tail[1], " ".join(p for p in (word, *tail[:2]) if p), "special", total)
    # Standard series.
    for p_len in range(1, 4):
        prefix = _run(s[:p_len], _letter)
        tail = _parse_tail(s, p_len)
        if prefix is None or tail is None:
            continue
        total = prefix[1] + tail[2]
        if best is None or total < best.cost:
            best = Plate(prefix[0] + tail[0] + tail[1], " ".join(p for p in (prefix[0], *tail[:2]) if p),
                         "standard", total)
    return best


def normalize(raw, max_drops=2):
    """Return the most likely Plate for raw OCR text, or None if nothing fits."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not 2 <= len(s) <= 14:
        return None
    best = _parse(s)
    # Also try ignoring 1-2 stray characters (e.g. text from the plate frame).
    for drops in range(1, max_drops + 1):
        for idx in itertools.combinations(range(len(s)), drops):
            shorter = "".join(c for i, c in enumerate(s) if i not in idx)
            p = _parse(shorter)
            if p is None:
                continue
            cost = p.cost + DROP_COST * drops + (FIRST_DROP_EXTRA if 0 in idx else 0)
            cost += 0.01 * sum(len(s) - i for i in idx)  # on a tie, prefer dropping later characters
            cost -= 0.1 * sum(i > 0 and s[i] == s[i - 1] for i in idx)  # OCR often reads one character twice
            if best is None or cost < best.cost:
                best = Plate(p.text, p.display, p.kind, cost)
    return best


def normalize_boxes(boxes):
    """Like normalize(), but for OCR boxes (x, y, height, text).

    Tries reading order first (top row, then left to right) and, for up to
    3 boxes, every other order, because two-row plates are sometimes
    returned in the wrong order. Drops tiny text such as dealer names.
    """
    if not boxes:
        return None, ""
    tallest = max(b[2] for b in boxes)
    boxes = [b for b in boxes if b[2] >= 0.5 * tallest]
    raw = join_lines(boxes)
    best = normalize(raw)
    if len(boxes) <= 3:
        for order in itertools.permutations(boxes):
            p = normalize("".join(b[3] for b in order))
            if p and (best is None or p.cost + 0.5 < best.cost):  # small penalty for reordering
                best = Plate(p.text, p.display, p.kind, p.cost + 0.5)
    return best, raw


def join_lines(boxes):
    """Merge OCR boxes (x, y, height, text) into reading order: top row first."""
    if not boxes:
        return ""
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) < 0.5 * max(b[2], current[-1][2]):
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return "".join(t for row in rows for _, _, _, t in sorted(row, key=lambda b: b[0]))


if __name__ == "__main__":
    p = normalize(" ".join(sys.argv[1:]) or "IB2744A")
    print(p.display if p else "could not read", f"(cost {p.cost:.1f})" if p else "")
