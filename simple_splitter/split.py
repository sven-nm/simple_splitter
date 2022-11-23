import random
from typing import List, Any, Optional, Tuple, Union


def _atomic_split(index: List[int],
                  splits: List[Tuple[Any, float]],
                  shuffle: bool = True,
                  random_seed: int = 42) -> List[Tuple[int, Any]]:
    """The atomic split function"""

    random.seed(random_seed)

    index_ = index.copy()

    if shuffle:
        random.shuffle(index_)

    results = []

    if len(index_) > len(splits):
        sample_sizes = [int(s[1] * len(index_)) for s in splits]  # Compute each split's sample size

        # Rebalance samples size to have at least one element per sample if possible,
        # substracting added element from a random sample than has more than one.
        while 0 in sample_sizes and any([sz > 1 for sz in sample_sizes]):
            zero_index = sample_sizes.index(0)
            stock = random.choice([sz for sz in sample_sizes if sz > 1])
            stock_index = sample_sizes.index(stock)
            sample_sizes[zero_index] += 1
            sample_sizes[stock_index] -= 1

        for s, k in zip(splits, sample_sizes):  # len(splits) = len(sample_sizes)
            # try:
            results += [(index_.pop(0), s[0]) for _ in range(k)]  # Pop the first index k times
            # except IndexError:
            #     print(sample_sizes)

    if len(index_) <= len(splits):  # If there are any left or if very small group, distribute with priority
        results += [(index_.pop(0), splits[i][0]) for i in range(len(index_))]

    assert not index_

    return results


def _sort_output(output: List[Tuple[int, Any]]):
    return [el[1] for el in sorted(output, key=lambda x: x[0])]


def _print_stats(output: List[Any], splits: List[Tuple[Any, float]]):
    for s in splits:
        count = output.count(s[0])
        print(f"""Split {s[0]} got {count} examples (effective ratio: {count / len(output)}, expected: {s[1]})""")


def split(splits: List[Tuple[Any, Union[float, int]]],
          data_length: int = None,
          stratification_columns: Optional[List[Union[list, 'pd.Series']]] = None,
          shuffle: bool = True,
          random_seed: int = 42) -> List[Any]:
    """Creates a split index for data of length `data_length` and according to the desired `splits`.

    This function returns a list of length `data_length` representing the distribution of splits, for instance,
    `['train', 'dev', 'dev', ... , 'test']`. On the contrary

    Args:
        splits: A list of tuples, where each tuple specifies the name/number/id of split and its ratio.

                Examples:
                    >>> splits = [('train', 0.65), ('dev1', 0.10), ('dev2', 0.10), ('test', 0.15)]
                    >>> splits = [('train', 140), ('dev', 18), ('test', 20)]
                Note:
                    - Split-ratios should sum to 1.
                    - You can prioritise the distribution by ordering this list. For very tiny datasets, you sometimes
                      want to make sure that e.g. your test set gets available examples in priority. In that case,
                      you just want to set your test tuple as the first of the list (etc..)

        data_length: The length of the data to split. If None, the length of the `stratification_columns` is used.

        stratification_columns: A list of length `data_length` to be passed split data in a stratified fashion. If provided, splitting
                is done at the level of each subset.

        shuffle: If set to false, data gets distributed into splits in a linear manner.
        random_seed: For reproducibility.

    Returns:
         A list of length `data_length` representing the distribution of splits.
    """

    # We start by creating our index
    index = list(range(data_length)) if data_length is not None else list(range(len(stratification_columns[0])))

    # We convert the splits to a list of tuples ratios if necessary
    if all([s[1] >= 1 for s in splits]) and len(splits) > 1:  # In case of unique split in pipeline
        splits = [(s[0], s[1] / len(index)) for s in splits]

    # We check that the splits are valid
    assert sum([s[1] for s in splits]) == 1.0, """`splits` ratios should sum to 1"""

    # If no stratification columns are provided, we simply return the atomic split
    if stratification_columns is None:
        output = _sort_output(_atomic_split(index, splits=splits))
        _print_stats(output, splits)
        return output

    else:

        # We check that the stratification columns are valid
        assert all([len(stratification_columns[0]) == len(col) for col in stratification_columns]), \
            """All stratification columns should have the same length"""

        # We create a single stratification column
        stratification_column = [tuple([s[i] for s in stratification_columns])
                                 for i in range(len(stratification_columns[0]))]

        outputs = []
        for strat in set(stratification_column):
            strat_index = [idx for idx, strat_ in zip(index, stratification_column) if strat_ == strat]
            outputs += _atomic_split(strat_index, splits, shuffle=shuffle, random_seed=random_seed)

        outputs = _sort_output(outputs)
        _print_stats(outputs, splits)
        return outputs
