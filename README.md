# Simple Splitter 🍌


## Presentation

`simple_splitter` is a handy data-splitter that circumvents the some limitations of `scikit-learn`'s `train_test_split()`:

1. `simple_splitter` allows for a straightforward, custom train-dev-test splitting. As a matter of fact, you can have as many splits as you want. 
2. `simple_splitter` handles very tiny data, included stratified-splitting of classes with a single element. 
The basic idea is the following : 
    - If there is more or as many examples as splits, make sure each splits has at least one example.
    - If there are less examples than splits, prioritise the desired splits (you can set priorities). 

Besides, on the contrary to `train_test_split()`, `simple_splitter` returns a single list representing the distribution 
of splits along your data. A matter of choice 😉

## Example

If your data is in a `pandas.DataFrame`, with columns of length `len(df)`, you can easily create your split column 
with `simple_splitter`:

```python
from simple_splitter.split import split

# Define you splits :
splits = [('train', 0.65), ('dev1', 0.10), ('dev2', 0.10), ('test', 0.15)]

split_column = split(splits=splits, data_length=len(my_df))
```

If you need a stratified split, just passed your stratifying column(s) :  
```python
split_column = split(data_length=len(my_df), splits=splits, strats=[my_df['my_stratitying_class_column_1'], ['my_stratitying_class_column_2']])
```



## Install

Installation requires no external dependancies. 

```shell
python -m pip install git+'https://github.com/sven-nm/simple_splitter'
```


## Let's chat !

Any improvements, suggestions and ideas are welcome ! 