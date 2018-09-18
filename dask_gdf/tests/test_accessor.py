import pytest
import numpy as np
from pandas.util.testing import assert_series_equal
from pygdf.dataframe import Series
import dask_gdf as dgd
import pandas as pd

#############################################################################
#                        Datetime Accessor                                  #
#############################################################################


def data_dt_1():
    return pd.date_range('20010101', '20020215', freq='400h')


def data_dt_2():
    return np.random.randn(100)


dt_fields = ['year', 'month', 'day', 'hour', 'minute', 'second']


@pytest.mark.parametrize('data', [data_dt_2()])
@pytest.mark.xfail(raises=AttributeError)
def test_datetime_accessor_initialization(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_pygdf(sr, npartitions=5)
    dsr.dt


@pytest.mark.parametrize('data', [data_dt_1()])
def test_series(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_pygdf(sr, npartitions=5)

    np.testing.assert_equal(
        np.array(pdsr),
        np.array(dsr.compute()),
    )


@pytest.mark.parametrize('data', [data_dt_1()])
@pytest.mark.parametrize('field', dt_fields)
def test_dt_series(data, field):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_pygdf(sr, npartitions=5)
    base = getattr(pdsr.dt, field)
    test = getattr(dsr.dt, field).compute()\
                                 .to_pandas().astype('int64')
    assert_series_equal(base, test)


#############################################################################
#                        Categorical Accessor                               #
#############################################################################


def data_cat_1():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    return cat


def data_cat_2():
    cat = pd.Categorical(['a', '_', '_', 'c', 'a'], categories=['a', 'b', 'c'])
    return cat


def data_cat_3():
    return pd.Series([1, 2, 3])


@pytest.mark.parametrize('data', [data_cat_3()])
@pytest.mark.xfail(raises=AttributeError)
def test_categorical_accessor_initialization(data):
    sr = Series(data.copy())
    dsr = dgd.from_pygdf(sr, npartitions=5)
    dsr.cat


@pytest.mark.parametrize('data', [data_cat_1()])
def test_categorical_basic(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dgd.from_pygdf(sr, npartitions=2)
    result = dsr.compute()
    np.testing.assert_array_equal(cat.codes, result.to_array())
    assert dsr.dtype == pdsr.dtype

    # Test attributes
    assert pdsr.cat.ordered == dsr.cat.ordered.compute()
    # TODO: Investigate dsr.cat.categories: It raises
    # ValueError: Expected iterable of tuples of (name, dtype), got ('a', 'b', 'c')
    # assert(tuple(pdsr.cat.categories) == tuple(dsr.cat.categories))

    np.testing.assert_array_equal(pdsr.cat.codes.data, result.to_array())
    np.testing.assert_array_equal(pdsr.cat.codes.dtype, dsr.cat.codes.dtype)

    string = str(result)
    expect_str = """
0 a
1 a
2 b
3 c
4 a
"""
    assert all(x == y for x, y in zip(string.split(), expect_str.split()))
