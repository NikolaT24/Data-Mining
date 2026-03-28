import sys
import math
import random
import urllib.request
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

VOTING_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

DOMAIN = ("y", "n", "a")

Sample = Tuple[List[str], str]

def fetch_votes() -> List[Sample]:
    with urllib.request.urlopen(VOTING_URL) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    data: List[Sample] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 17:
            continue
        label = parts[0]
        feats = parts[1:]
        data.append((feats, label))
    return data

def preprocess_as_abstain(data: List[Sample]) -> List[Sample]:
    out: List[Sample] = []
    for feats, y in data:
        out_feats = [('a' if v == '?' else v) for v in feats]
        out.append((out_feats, y))
    return out


def compute_train_modes(train_data: List[Sample]) -> List[str]:
    modes: List[str] = []
    for j in range(16):
        c = Counter()
        for feats, _ in train_data:
            v = feats[j]
            if v != '?':
                c[v] += 1
        if c:
            modes.append(c.most_common(1)[0][0])
        else:
            modes.append('a')
    return modes


def fill_missing_with_modes(data: List[Sample], modes: List[str]) -> List[Sample]:
    out: List[Sample] = []
    for feats, y in data:
        out_feats = []
        for j, v in enumerate(feats):
            if v == '?':
                out_feats.append(modes[j])
            else:
                out_feats.append(v)
        out_feats = [('a' if v == '?' else v) for v in out_feats]
        out.append((out_feats, y))
    return out

def stratified_split(data: List[Sample], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for feats, y in data:
        by_class[y].append((feats, y))

    train: List[Sample] = []
    test: List[Sample] = []
    for y, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        cut = int(len(items) * train_ratio)
        train.extend(items[:cut])
        test.extend(items[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_k_folds(data: List[Sample], k: int = 10, seed: int = 42) -> List[List[Sample]]:
    rng = random.Random(seed)
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for feats, y in data:
        by_class[y].append((feats, y))

    folds: List[List[Sample]] = [[] for _ in range(k)]
    for y, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        for i, sample in enumerate(items):
            folds[i % k].append(sample)

    for f in folds:
        rng.shuffle(f)
    return folds

class CategoricalNB:
    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.labels: List[str] = []
        self.log_prior: Dict[str, float] = {}
        self.log_likelihood: Dict[str, List[Dict[str, float]]] = {}

    def fit(self, train_data: List[Sample]) -> "CategoricalNB":
        label_counts = Counter(y for _, y in train_data)
        self.labels = sorted(label_counts.keys())

        n = len(train_data)
        k = len(self.labels)

        for lbl in self.labels:
            self.log_prior[lbl] = math.log((label_counts[lbl] + self.lam) / (n + self.lam * k))

        counts: Dict[str, List[Counter]] = {
            lbl: [Counter() for _ in range(16)] for lbl in self.labels
        }
        totals: Dict[str, List[int]] = {
            lbl: [0] * 16 for lbl in self.labels
        }

        for feats, y in train_data:
            for j, v in enumerate(feats):
                if v not in DOMAIN:
                    v = 'a'
                counts[y][j][v] += 1
                totals[y][j] += 1

        self.log_likelihood = {lbl: [] for lbl in self.labels}
        V = len(DOMAIN)

        for lbl in self.labels:
            for j in range(16):
                denom = totals[lbl][j] + self.lam * V
                dist: Dict[str, float] = {}
                for val in DOMAIN:
                    c = counts[lbl][j][val]
                    dist[val] = math.log((c + self.lam) / denom)
                self.log_likelihood[lbl].append(dist)

        return self

    def predict_one(self, feats: List[str]) -> str:
        best_lbl = None
        best_score = -1e300

        for lbl in self.labels:
            s = self.log_prior[lbl]
            ll = self.log_likelihood[lbl]
            for j, v in enumerate(feats):
                if v not in DOMAIN:
                    v = 'a'
                s += ll[j][v]
            if s > best_score:
                best_score = s
                best_lbl = lbl

        return best_lbl

    def predict(self, data: List[Sample]) -> List[str]:
        return [self.predict_one(feats) for feats, _ in data]


def accuracy(data: List[Sample], preds: List[str]) -> float:
    correct = 0
    for (_, y), p in zip(data, preds):
        if y == p:
            correct += 1
    return 100.0 * correct / len(data)


def mean_std(values: List[float]) -> Tuple[float, float]:
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)

def evaluate(mode: int, lam: float = 1.0, seed: int = 42) -> None:
    raw = fetch_votes()

    if mode == 0:
        data = preprocess_as_abstain(raw)
        train, test = stratified_split(data, 0.8, seed)

        nb = CategoricalNB(lam=lam).fit(train)
        train_acc = accuracy(train, nb.predict(train))

        folds = stratified_k_folds(train, 10, seed)
        fold_accs: List[float] = []
        for i in range(10):
            val = folds[i]
            tr = []
            for j in range(10):
                if j != i:
                    tr.extend(folds[j])
            nb_fold = CategoricalNB(lam=lam).fit(tr)
            fold_accs.append(accuracy(val, nb_fold.predict(val)))
        avg, std = mean_std(fold_accs)

        test_acc = accuracy(test, nb.predict(test))

    elif mode == 1:
        train_raw, test_raw = stratified_split(raw, 0.8, seed)

        modes = compute_train_modes(train_raw)
        train = fill_missing_with_modes(train_raw, modes)
        test = fill_missing_with_modes(test_raw, modes)

        nb = CategoricalNB(lam=lam).fit(train)
        train_acc = accuracy(train, nb.predict(train))

        folds_raw = stratified_k_folds(train_raw, 10, seed)
        fold_accs = []
        for i in range(10):
            val_raw = folds_raw[i]
            tr_raw = []
            for j in range(10):
                if j != i:
                    tr_raw.extend(folds_raw[j])

            fold_modes = compute_train_modes(tr_raw)
            tr = fill_missing_with_modes(tr_raw, fold_modes)
            val = fill_missing_with_modes(val_raw, fold_modes)

            nb_fold = CategoricalNB(lam=lam).fit(tr)
            fold_accs.append(accuracy(val, nb_fold.predict(val)))

        avg, std = mean_std(fold_accs)

        test_acc = accuracy(test, nb.predict(test))

    else:
        print("Invalid input. Enter 0 or 1.")
        sys.exit(1)

    print("1. Train Set Accuracy:")
    print(f"    Accuracy: {train_acc:.2f}%\n")

    print("10-Fold Cross-Validation Results:\n")
    for i, a in enumerate(fold_accs, start=1):
        print(f"    Accuracy Fold {i}: {a:.2f}%")
    print()
    print(f"    Average Accuracy: {avg:.2f}%")
    print(f"    Standard Deviation: {std:.2f}%\n")

    print("2. Test Set Accuracy:")
    print(f"    Accuracy: {test_acc:.2f}%")


def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    mode = int(data[0])

    lam = 1.0
    evaluate(mode=mode, lam=lam, seed=42)

if __name__ == "__main__":
    main()
