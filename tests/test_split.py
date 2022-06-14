import random
from simple_splitter import split
import pytest

random_seed = 42
random.seed(random_seed)

data_length = 20
strats = [random.choice(['a', 'b', 'c']) for _ in range(data_length)]
splits = [('train', 0.7), ('test', 0.15), ('dev', 0.15)]
shuffle = True


def test_split():
    output = split.split(data_length=data_length,
                         splits=splits, strats=strats,
                         shuffle=shuffle,
                         random_seed=random_seed)
    assert len(output) == data_length
