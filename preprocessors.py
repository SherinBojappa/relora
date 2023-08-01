# Convert dataset into text to text format.
# The code is based on https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py

import re
import torch

def glue(x, benchmark_name, label_names, feature_names=None, id_key='idx'):
    """Convert a dataset from glue to text2text examples.

    This function uses the feature names from the dataset to unpack examples into
    a format amenable for a text2text problem. For example, consider the Quora
    Question Pairs (QQP) benchmark, which would suggest
    benchmark_name="qqp"
    label_names=['not_duplicate', 'duplicate']
    For QQP, a typical example might look like
    {
        "question1": "Why do I easily get bored of my friends?",
        "question2": "Why do I get bored of friends so quickly?",
        "label": 1,
        "idx": 10,
    }

    This example would be transformed to
    {
        "inputs": (
            "qqp question1: Why do I easily get bored of my friends? question2: "
            "Why do I get bored of my friends so quickly?"
        ),
        "targets": "duplicate",
        "idx": 10,
    }

    Args:
        x: an example to process.
        benchmark_name: the name of the GLUE benchmark for this dataset.
        label_names: a list of label names corresponding to class index.
        feature_names: an optional ordered list of feature names. If provided,
        features will be ordered in this way in the output. If not provided, all
        features (except 'idx' and 'label') will be used, sorted by name.
        id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
        if None, no id will be added to the dataset.

    Returns:
        A preprocessed example.
    """
    feature_keys = feature_names or sorted(set(x.keys()).difference(['label', 'idx']))
    strs_to_join = []
    for key in feature_keys:
        strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])
    strs_to_join.insert(0, benchmark_name)
    label_name = '<unk>' if x['label'] == -1 else label_names[x['label']]
    joined = ' '.join(strs_to_join)

    ex = {}
    ex['inputs'] = joined
    ex['targets'] = label_name

    return ex

def stsb(x):
    """Convert STSB examples to text2text format.

    STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    This function uses the feature names from the dataset to unpack examples into
    a format amenable for a text2text problem.

    For example, a typical example from STSB might look like
    {
        "sentence1": "Three more US soldiers killed in Afghanistan",
        "sentence2": "NATO Soldier Killed in Afghanistan",
        "label": 1.8,
    }

    This example would be transformed to
    {
        "inputs": (
            "stsb sentence1: Three more US soldiers killed in Afghanistan "
            "sentence2: NATO Soldier Killed in Afghanistan"
        ),
        "targets": "1.8",
    }

    Args:
        x: an example to process.
    Returns:
        A preprocessed example.
    """
    strs_to_join = ['stsb sentence1:', x['sentence1'], 'sentence2:', x['sentence2']]
    label_string = str(round(x['label'] * 5) / 5)
    joined = ' '.join(strs_to_join)
    return {'inputs': joined, 'targets': label_string, 'idx': x['idx']}

def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default
