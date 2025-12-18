import sys
from collections import Counter


def interesting(n: int) -> bool:
    c = Counter(str(n))
    vals = sorted(c.values())
    return len(vals) >= 2 and vals[0] == 1 and all(v == vals[1] for v in vals[1:])


def main():
    x, y = map(int, sys.stdin.read().strip().split())
    print(sum(1 for n in range(x, y + 1) if interesting(n)))


if __name__ == "__main__":
    main()
