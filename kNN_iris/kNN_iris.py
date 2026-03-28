import math
import os
import random
import sys
from typing import List, Tuple, Dict

Sample = Tuple[List[float], str]

def load_iris(filename: str = "iris.data") -> List[Sample]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    data: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            x = list(map(float, parts[:4]))
            y = parts[4]
            data.append((x, y))
    return data

def group_by_label(dataset: List[Sample]) -> Dict[str, List[Sample]]:
    g: Dict[str, List[Sample]] = {}
    for x, y in dataset:
        g.setdefault(y, []).append((x, y))
    return g

def stratified_split(dataset: List[Sample], train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    groups = group_by_label(dataset)
    train: List[Sample] = []
    test: List[Sample] = []
    for items in groups.values():
        items = items[:]
        rng.shuffle(items)
        cut = int(len(items) * train_ratio)
        train.extend(items[:cut])
        test.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def stratified_k_folds(dataset: List[Sample], k_folds: int = 10, seed: int = 42):
    rng = random.Random(seed)
    groups = group_by_label(dataset)
    folds: List[List[Sample]] = [[] for _ in range(k_folds)]
    for items in groups.values():
        items = items[:]
        rng.shuffle(items)
        for i, it in enumerate(items):
            folds[i % k_folds].append(it)
    for fold in folds:
        rng.shuffle(fold)
    return folds

def fit_minmax(train_set: List[Sample]):
    cols = len(train_set[0][0])
    mins = [float("inf")] * cols
    maxs = [float("-inf")] * cols
    for x, _ in train_set:
        for i in range(cols):
            if x[i] < mins[i]:
                mins[i] = x[i]
            if x[i] > maxs[i]:
                maxs[i] = x[i]
    return mins, maxs

def apply_minmax(dataset: List[Sample], mins, maxs):
    out: List[Sample] = []
    for x, y in dataset:
        nx = []
        for i in range(len(x)):
            if maxs[i] == mins[i]:
                nx.append(0.0)
            else:
                nx.append((x[i] - mins[i]) / (maxs[i] - mins[i]))
        out.append((nx, y))
    return out

def euclidean(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)

def knn_predict(train_set: List[Sample], x_new, k: int):
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, len(train_set))
    dists = [(euclidean(x, x_new), y) for x, y in train_set]
    dists.sort(key=lambda t: t[0])
    votes: Dict[str, int] = {}
    dist_sum: Dict[str, float] = {}
    for dist, y in dists[:k]:
        votes[y] = votes.get(y, 0) + 1
        dist_sum[y] = dist_sum.get(y, 0.0) + dist
    best = None
    best_votes = -1
    best_mean = float("inf")
    for y, cnt in votes.items():
        mean = dist_sum[y] / cnt
        if cnt > best_votes or (cnt == best_votes and mean < best_mean):
            best = y
            best_votes = cnt
            best_mean = mean
    return best

def accuracy(dataset: List[Sample], preds: List[str]) -> float:
    correct = 0
    for (_, y), p in zip(dataset, preds):
        if y == p:
            correct += 1
    return 100.0 * correct / len(dataset)

def cv_10fold(train_raw: List[Sample], k: int, seed: int = 42):
    folds = stratified_k_folds(train_raw, 10, seed)
    accs: List[float] = []
    for i in range(10):
        test_fold_raw = folds[i]
        train_fold_raw: List[Sample] = []
        for j in range(10):
            if j != i:
                train_fold_raw.extend(folds[j])
        mins, maxs = fit_minmax(train_fold_raw)
        train_fold = apply_minmax(train_fold_raw, mins, maxs)
        test_fold = apply_minmax(test_fold_raw, mins, maxs)
        preds = [knn_predict(train_fold, x, k) for x, _ in test_fold]
        accs.append(accuracy(test_fold, preds))
    avg = sum(accs) / 10.0
    std = math.sqrt(sum((a - avg) ** 2 for a in accs) / 9.0)
    return accs, avg, std

def main():
    data = load_iris("iris.data")
    train_raw, test_raw = stratified_split(data, 0.8, seed=42)

    mins, maxs = fit_minmax(train_raw)
    train = apply_minmax(train_raw, mins, maxs)
    test = apply_minmax(test_raw, mins, maxs)

    k_values = list(range(1, 22, 2))
    if sys.stdin.isatty():
        pass
    else:
        txt = sys.stdin.read().strip().split()
        if txt:
            k_values = [int(txt[0])]

    for k in k_values:
        train_preds = [knn_predict(train, x, k) for x, _ in train]
        train_acc = accuracy(train, train_preds)

        fold_accs, avg_acc, std_acc = cv_10fold(train_raw, k, seed=42)

        test_preds = [knn_predict(train, x, k) for x, _ in test]
        test_acc = accuracy(test, test_preds)

        print(f"k = {k}")
        print("1. Train Set Accuracy:")
        print(f"    Accuracy: {train_acc:.2f}%\n")
        print("2. 10-Fold Cross-Validation Results:")
        for i, a in enumerate(fold_accs, 1):
            print(f"    Accuracy Fold {i}: {a:.2f}%")
        print(f"\n    Average Accuracy: {avg_acc:.2f}%")
        print(f"    Standard Deviation: {std_acc:.2f}%\n")
        print("3. Test Set Accuracy:")
        print(f"    Accuracy: {test_acc:.2f}%")
        print()

if __name__ == "__main__":
    main()
