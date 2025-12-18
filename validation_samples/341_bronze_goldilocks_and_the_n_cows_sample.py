import sys


def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n, x, y, z = map(int, data[:4])
    vals = list(map(int, data[4:]))
    pairs = list(zip(vals[0::2], vals[1::2]))

    events = []
    for a, b in pairs:
        events.append((a, y - x))       # enter comfortable: gain (Y-X)
        events.append((b + 1, z - y))   # leave comfortable: add (Z-Y)
    events.sort()

    current = n * x
    best = current
    for _, delta in events:
        current += delta
        if current > best:
            best = current

    print(best)


if __name__ == "__main__":
    main()
