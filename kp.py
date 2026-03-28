import sys
import os
import time

def main():
    start = time.time()

    data = sys.stdin.read().split()
    if not data:
        return

    capacity = int(data[0])
    n = int(data[1])

    weights = []
    values = []

    idx = 2
    for _ in range(n):
        weights.append(int(data[idx]))
        values.append(int(data[idx+1]))
        idx += 2

    if os.getenv("FMI_TIME_ONLY") == "1":
        dp = [0] * (capacity + 1)
        for i in range(n):
            w = weights[i]
            v = values[i]
            for c in range(capacity, w - 1, -1):
                dp[c] = max(dp[c], dp[c - w] + v)

        ms = int((time.time() - start) * 1000)
        print(f"# TIMES_MS: alg={ms}")
        return

    dp = [0] * (capacity + 1)
    snapshots = []
    snap_points = set([0] + [n * i // 9 for i in range(1, 9)] + [n - 1])

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)

        if i in snap_points:
            snapshots.append(max(dp))

    while len(snapshots) < 10:
        snapshots.append(snapshots[-1])

    for val in snapshots:
        print(val)

    print()
    print(snapshots[-1])


if __name__ == "__main__":
    main()
