import random
from typing import List, Any, Optional, Tuple

class Bonjour():
    def __init__(self):
        pass

    def saluer(self):
        print('bonjour')


def _atomic_split(index: List[int], splits: List[Tuple[Any, float]]) -> List[Tuple[int, Any]]:
    """The atomic split function"""

    index = index.copy()

    results = []

    if len(index) > len(splits):

        sample_sizes = [int(s[1] * len(index)) for s in splits]  # Compute each split's sample size
        results += [(index.pop(0), s[0]) for s in splits]  # Distribute at least one sample per split

        for s, k in zip(splits, sample_sizes):  # len(splits) = len(k)
            results += [(index.pop(0), s[0]) for _ in range(max(k-1),0)]  # Pop the first indices k-1 times

    if len(index) <= len(splits):  # If there are any left or if very small group, distribute with priority
        results += [(index.pop(0), splits[i][0]) for i in range(len(index))]

    assert not index

    return results


def _sort_output(output: List[Tuple[int, Any]]):
    return [el[1] for el in sorted(output, key=lambda x: x[0])]


def _print_stats(output: List[Any], splits: List[Tuple[Any, float]]):
    for s in splits:
        count = output.count(s[0])
        print(f"""Split {s[0]} got {count} examples (effective ratio: {count / len(output)}, expected: {s[1]})""")


def split(data_length: int,
          splits: List[Tuple[Any, float]],
          strats: Optional[List[Any]] = None,
          shuffle: bool = True,
          random_seed: int = 42,
          ) -> List[Any]:
    """Creates a split index for data of length `data_length` and according to the desired `splits`.

    This function returns a list of length `data_length` representing the distribution of splits, for instance,
    `['train', 'dev', 'dev', ... , 'test']`. On the contrary

    Args:
        data_length: The length of the data. For instance, if your data is a `pandas.DataFrame`, then you should set it
                     to `len(df.index)`.
        splits: A list of tuples, where each tuple specifies the name/number/id of split and its ratio. For instance,
                `[('train', 0.65), ('dev1', 0.10), ('dev2', 0.10), ('test', 0.15)]`.
                Note:
                    - Split-ratios should sum to 1.
                    - You can prioritise the distribution by ordering this list. For very tiny datasets, you sometimes
                      want to make sure that e.g. your test set gets available examples in priority. In that case,
                      you just want to set your test tuple as the first of the list (etc..)
        strats: A list of length `data_length` to be passed split data in a stratified fashion. If provided, splitting
                is done at the level of each subset.
        shuffle: If set to false, data gets distributed into splits in a linear manner.
        random_seed: For reproducibility.

    Returns:
         A list of length `data_length` representing the distribution of splits.
    """

    assert sum([s[1] for s in splits]) == 1.0, """`splits` ratios should sum to 1"""

    index = list(range(data_length))

    if shuffle:
        random.seed(random_seed)
        random.shuffle(index)

    if not strats:
        output = _sort_output(_atomic_split(index, splits=splits))
        _print_stats(output, splits)
        return output

    else:
        assert len(strats) == data_length, """`strats` must be a list of length `data_length`"""

        outputs = []

        for strat in set(strats):
            strat_index = [idx for idx, strat_ in zip(index, strats) if strat_ == strat]
            if shuffle:
                random.shuffle(strat_index)
            outputs += _atomic_split(strat_index, splits)

        outputs = _sort_output(outputs)
        _print_stats(outputs, splits)
        return outputs

# Todo : add version
# Todo: add no dependency needed


bonjour = 'bonjour'