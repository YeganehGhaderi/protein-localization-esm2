import numpy as np
import pandas as pd
import torch

from src.data import (AA_TO_IDX, encode_sequence, split_by_partition,cluster_by_shared_kmers, split_train_val_by_cluster)
from src.models import ProteinCNN


def test_encode_pads_short_sequences():
    ids = encode_sequence("ACD", max_len=10)
    assert ids.shape == (10,)
    assert (ids[:3] > 0).all() and (ids[3:] == 0).all()


def test_encode_keeps_both_ends_of_long_sequences():
    ids = encode_sequence("A" * 600 + "Y" * 600, max_len=1000)
    assert ids.shape == (1000,)
    assert ids[0] == AA_TO_IDX["A"] and ids[-1] == AA_TO_IDX["Y"]


def test_split_has_no_overlap():
    df = pd.DataFrame({"id": list("abcdef"), "part": ["0", "0", "1", "2", "2", "test"]})
    train, val, test = split_by_partition(df, test_part="test", val_part="2")
    assert set(train.id).isdisjoint(val.id)
    assert set(train.id).isdisjoint(test.id)
    assert set(val.id).isdisjoint(test.id)
    assert len(train) + len(val) + len(test) == len(df)


def test_cnn_output_shape():
    model = ProteinCNN(n_classes=10)
    x = torch.randint(1, 22, (4, 50))
    assert model(x).shape == (4, 10)


def test_padding_does_not_change_cnn_output():
    model = ProteinCNN(n_classes=10).eval()
    x = torch.randint(1, 22, (1, 40))
    x_padded = torch.cat([x, torch.zeros(1, 30, dtype=torch.long)], dim=1)
    assert torch.allclose(model(x), model(x_padded), atol=1e-5)

def test_clustering_groups_near_duplicates():
    a = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    b = a[:-3] + "AAA"                       # almost identical to a
    c = "WWPQHHFYYCCDDEEGGNNRRSSTTVVIILLMMKKAAPPFFWWYYQQHH"   # unrelated
    labels = cluster_by_shared_kmers([a, b, c], k=5, min_frac=0.02)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_split_never_splits_a_cluster():
    groups = np.repeat(np.arange(20), 2)                        # 20 clusters of 2 sequences
    df = pd.DataFrame({"location": np.repeat(["A", "B"] * 10, 2)})
    train, val = split_train_val_by_cluster(df, groups, seed=0)
    assert set(groups[train.index]).isdisjoint(set(groups[val.index]))
    assert len(train) + len(val) == len(df)

def test_single_shared_kmer_does_not_link_short_sequences():
    a = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELS"
    b = "WWPQHKTVRQFFYYCCDDEEGGNNRRSSTT"      # shares only the 5-mer KTVRQ with a
    labels = cluster_by_shared_kmers([a, b], k=5, min_frac=0.02)
    assert labels[0] != labels[1]

def test_trim_ends_keeps_both_ends():
    from src.esm_embed import trim_ends
    s = trim_ends("A" * 600 + "Y" * 600, max_len=1000)
    assert len(s) == 1000 and s[0] == "A" and s[-1] == "Y"
    assert trim_ends("ACD", max_len=1000) == "ACD"


def test_embedding_classifier_output_shape():
    from src.models import EmbeddingClassifier
    model = EmbeddingClassifier(in_dim=480, n_classes=10)
    assert model(torch.randn(5, 480)).shape == (5, 10)


def test_trainable_state_roundtrip():
    import torch.nn as nn
    from src.lora import count_trainable, load_trainable_state, trainable_state

    m = nn.Linear(4, 2)
    m.bias.requires_grad = False                    # only the weight is trainable
    state = trainable_state(m)
    assert set(state) == {"weight"} and count_trainable(m) == 8

    with torch.no_grad():
        m.weight.zero_()
    load_trainable_state(m, state)
    assert not torch.all(m.weight == 0)             # saved weights were restored
