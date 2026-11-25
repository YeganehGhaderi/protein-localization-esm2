import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}   # 0 is reserved for padding
UNK_IDX = len(AMINO_ACIDS) + 1                                 # rare letters (X, U, B, ...)
VOCAB_SIZE = UNK_IDX + 1


def read_fasta(path):
    """Yield (header, sequence) for every entry of a FASTA file."""
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def load_deeploc(path):
    """Parse a DeepLoc FASTA file into a DataFrame.

    Expected header: '<id> <Location>-<M/S/U> [<partition info>]'.
    """
    rows = []
    for header, seq in read_fasta(path):
        parts = header.split()
        location, _, membrane = parts[1].rpartition("-")
        rows.append({
            "id": parts[0],
            "location": location,
            "membrane": membrane,
            "part": parts[2] if len(parts) > 2 else None,
            "seq": seq,
        })
    return pd.DataFrame(rows)


def split_by_partition(df, test_part, val_part):
    """Homology-aware split: whole partitions go to test / val / train."""
    test = df[df.part == test_part]
    val = df[df.part == val_part]
    train = df[~df.part.isin([test_part, val_part])]
    return train, val, test


def encode_sequence(seq, max_len=1000):
    """Letters -> integers, padded with 0 up to max_len.

    Long sequences keep their first and last max_len/2 residues, because
    targeting signals sit at both ends of a protein.
    """
    if len(seq) > max_len:
        half = max_len // 2
        seq = seq[:half] + seq[-half:]
    ids = [AA_TO_IDX.get(a, UNK_IDX) for a in seq]
    out = np.zeros(max_len, dtype=np.int64)
    out[:len(ids)] = ids
    return out


def encode_many(seqs, max_len=1000):
    return np.stack([encode_sequence(s, max_len) for s in seqs])

def shared_kmer_matrix(seqs, k=5):
    """Return (shared, n_kmers): shared[i, j] = number of k-mers sequences i and j have in common."""
    vec = HashingVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False,
                            n_features=2 ** 24, binary=True, norm=None,
                            alternate_sign=False)
    X = vec.transform(seqs).astype(np.float32).tocsr()
    n_kmers = np.asarray(X.sum(axis=1)).ravel()
    shared = (X @ X.T).tocoo()
    return shared, n_kmers


def cluster_from_shared(shared, n_kmers, min_frac=0.05, min_shared=10):
    """Link two sequences if they share at least `min_shared` k-mers AND at least
    `min_frac` of the k-mers of the shorter one; clusters = connected components."""
    n = len(n_kmers)
    denom = np.minimum(n_kmers[shared.row], n_kmers[shared.col])
    linked = ((shared.row != shared.col)
              & (shared.data >= min_shared)
              & (shared.data / denom >= min_frac))
    graph = csr_matrix((np.ones(linked.sum()), (shared.row[linked], shared.col[linked])),
                       shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def cluster_by_shared_kmers(seqs, k=5, min_frac=0.05, min_shared=10):
    shared, n_kmers = shared_kmer_matrix(seqs, k)
    return cluster_from_shared(shared, n_kmers, min_frac, min_shared)


def split_train_val_by_cluster(df, groups, seed=0, n_splits=5):
    """Hold out ~1/n_splits of the data as validation; clusters are never split."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, df.location, groups))
    return df.iloc[train_idx], df.iloc[val_idx]