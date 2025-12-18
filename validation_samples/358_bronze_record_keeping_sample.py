import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    n = int(data[0])
    counts = defaultdict(int)
    for line in data[1:]:
        group = tuple(sorted(line.split()))
        counts[group] += 1
    print(max(counts.values()) if counts else 0)


if __name__ == "__main__":
    main()
