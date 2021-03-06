# pandacell

Author: Eirik B. Stavestrand

Introduces two magic commands `%df` and `%%df`, which can be used in Jupyter notebooks and the IPython console.

The magics execute the contents of a cell on a Pandas DataFrame.

## Motivation
Pandas is great and all, but writing Pandas code can be tedious. For example when simply making summing two columns:

```python
    In [1]: df["a"] + df["b"]
```

It might not look like such a big deal, but all those brackets and quotation marks add up.
Using *pandacell*, the above syntax can be written as:

```python
    In [2]: %df a + b
```

Under the hoods, this is accomplished simply by passing the cell contents as a string to Pandas' [`df.eval`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html) function.
This isn't very complex, but it does provide a fair deal of functionality and adds a whole lot of readability.

If you wish to store the results to a new column, use regular assignment along with the `-i` (or `--inplace`) flag:

```python
    In [3]: %df -i c = a + b
```
It also works with multiple assignments:

```python
    In [4]: %%df -i
       ...: c = a + b
       ...: f = c - a
```
You can use Pandas' various accessors and series method calls:

```python
    In [5]: %%df -i
       ...: name_upper = name.str.upper()
       ...: yr = timestamp.dt.year
       ...: lower_cased = species.where(cond=species.str[0].str.islower(), other=None)
```

Since variable names are assumed to be columns in the dataframe, regular variables in the local/global namespace can be accessed by prefixing with `@`

```python
    In [6]: a = 1
       ...: %df a = @a + 1

    In [7]: def myfunc(row):
        ...:     return row + 43
        ...: %df b = a.apply(@myfunc)
```

By default, pandacell operates on any dataframe named `df`. This can be overridden with the `-n` (or `--name`) flag:

```python
    In [8]: %df -n=df_in c = a + b
```

You can also print subset a dataframe with the `-q` (or `--query`) flag:

```python
    In [9]: %df -q species == "setosa"
    Out[9]:
        sepal_length  sepal_width  petal_length  petal_width species  a
    0            5.1          3.5           1.4          0.2  setosa  0
    1            4.9          3.0           1.4          0.2  setosa  0


    In [10]: %df -q species.isna() #check for missing values
    Out[10]:
    Empty DataFrame
    Columns: [sepal_length, sepal_width, petal_length, petal_width, species]
    Index: []
```

This can be combined with the `-i` flag to subset the dataframe in-place:

```python
    In [10]: %df -q -i species == "setosa"
```

Pandacell even supports comments

```python
    In [11]: %%df -i
       ...: # Line comment
       ...: c = a + b # Comment at end of line
```

## Installation
Install the latest release with:

    pip install pandacell

or download from https://github.com/eirki/pandacell and:

    cd pandacell
    sudo python setup.py install



## Inspired by
https://github.com/catherinedevlin/ipython-sql


## Development
https://github.com/eirki/pandacell
