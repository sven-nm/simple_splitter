import random
from simple_splitter import split
import pytest

random_seed = 42
random.seed(random_seed)

data_length = 20
strats = [[random.choice(['a', 'b', 'c']) for _ in range(data_length)],
          [random.choice(['d', 'e', 'f']) for _ in range(data_length)],
          [random.choice(['g', 'h', 'i']) for _ in range(data_length)]]
splits = [('train', 0.7), ('test', 0.15), ('dev', 0.15)]
splits_int = [('train', 13), ('test', 6), ('dev', 1)]
shuffle = True


@pytest.mark.parametrize('shuffle, splits', [(v, s) for v in [True, False] for s in [splits, splits_int]])
def test_split(shuffle, splits):
    output = split.split(splits=splits, data_length=data_length, stratification_columns=strats, shuffle=shuffle,
                         random_seed=random_seed)
    assert len(output) == data_length
