import numpy as np
import polars as pl
import pytest

from chrono_features.features.autocorrelation import Autocorrelation
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Создает тестовый датасет с временными рядами."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "timestamp": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "value": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 4.0, 3.0, 7.0, 4.0, 10.0],
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_autocorrelation_lag1_expanding(sample_dataset: TSDataset) -> None:
    """Тестирует автокорреляцию с lag=1 на расширяющемся окне."""
    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.EXPANDING(),
        lag=1,
        out_column_names="autocorr_lag1_expanding",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_lag1_expanding"].to_numpy()

    # Ожидаемые значения для расширяющегося окна с lag=1
    expected = np.array(
        [
            # [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            np.nan,  # первый элемент - nan (недостаточно данных)
            np.nan,  # корреляция между [1] и [3] (недостаточно данных)
            1.0,  # корреляция между [1, 3] и [3, 5]
            -0.32732683535398865,  # корреляция между [1, 3, 5] и [3, 5, 2]
            -0.3779644730092272,  # корреляция между [1, 3, 5, 2] и [3, 5, 2, 4]
            0,  # корреляция между [1, 3, 5, 2, 4] и [3, 5, 2, 4, 6]
            # [1.0, 4.0, 3.0, 7.0, 4.0, 10.0]
            np.nan,  # первый элемент нового временного ряда
            np.nan,  # корреляция между [5] и [4] (недостаточно данных)
            -1.0,  # корреляция между [1, 4] и [4, 3]
            -0.05241424183609577,  # корреляция между [1, 4, 3] и [4, 3, 7]
            -0.19245008972987526,  # корреляция между [1, 4, 3, 7] и [4, 3, 7, 4]
            -0.05603766997559098,  # корреляция между [1, 4, 3, 7, 4] и [4, 3, 7, 4, 10]
        ]
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_lag2_rolling(sample_dataset: TSDataset) -> None:
    """Тестирует автокорреляцию с lag=2 на скользящем окне размера 5."""
    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.ROLLING(size=5),
        lag=2,
        out_column_names="autocorr_lag2_rolling",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_lag2_rolling"].to_numpy()

    # Ожидаемые значения для скользящего окна размера 4 с lag=2
    expected = np.array(
        [
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            -0.32732683535398865,  # корреляция между [1, 3, 5], [5, 2, 4] в окне [1, 3, 5, 2, 4]
            -0.32732683535398865,  # корреляция между [3, 5, 2] и [2, 4, 6] в окне [3, 5, 2, 4, 6]
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            np.nan,  # недостаточно данных
            0.8910421112136309,  # корреляция между [1, 4, 3], [3, 7, 4] в окне [1, 4, 3, 7, 4]
            0.9607689228305227,  # корреляция между [4, 3, 7], [7, 4, 10] в окне [4, 3, 7, 4, 10]
        ]
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_multiple_columns(sample_dataset: TSDataset) -> None:
    """Тестирует автокорреляцию на нескольких столбцах."""
    # Добавим второй столбец в датасет
    sample_dataset.add_feature("value2", [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1])

    autocorr = Autocorrelation(
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
        lag=1,
        out_column_names=["autocorr_value", "autocorr_value2"],
    )

    result = autocorr.transform(sample_dataset)

    # Проверяем первый столбец (как в первом тесте)
    result_values1 = result.data["autocorr_value"].to_numpy()
    expected1 = np.array(
        [
            # [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            np.nan,  # первый элемент - nan (недостаточно данных)
            np.nan,  # корреляция между [1] и [3] (недостаточно данных)
            1.0,  # корреляция между [1, 3] и [3, 5]
            -0.32732683535398865,  # корреляция между [1, 3, 5] и [3, 5, 2]
            -0.3779644730092272,  # корреляция между [1, 3, 5, 2] и [3, 5, 2, 4]
            0,  # корреляция между [1, 3, 5, 2, 4] и [3, 5, 2, 4, 6]
            # [1.0, 4.0, 3.0, 7.0, 4.0, 10.0]
            np.nan,  # первый элемент нового временного ряда
            np.nan,  # корреляция между [5] и [4] (недостаточно данных)
            -1.0,  # корреляция между [1, 4] и [4, 3]
            -0.05241424183609577,  # корреляция между [1, 4, 3] и [4, 3, 7]
            -0.19245008972987526,  # корреляция между [1, 4, 3, 7] и [4, 3, 7, 4]
            -0.05603766997559098,  # корреляция между [1, 4, 3, 7, 4] и [4, 3, 7, 4, 10]
        ]
    )
    print(result_values1)
    print(expected1)
    np.testing.assert_array_almost_equal(result_values1, expected1, decimal=5)

    # Проверяем второй столбец (должен быть весь NaN, так как значения постоянны)
    result_values2 = result.data["autocorr_value2"].to_numpy()
    expected2 = np.array([np.nan, np.nan, 1, 1, 1, 1, np.nan, np.nan, 1, 1, 1, 1])
    np.testing.assert_array_equal(result_values2, expected2)


def test_autocorrelation_dynamic_window(sample_dataset: TSDataset) -> None:
    """Тестирует автокорреляцию с динамическим окном."""
    # Добавим столбец с длинами окон
    sample_dataset.add_feature("window_len", [1, 2, 2, 3, 3, 4, 1, 2, 2, 3, 3, 4])

    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
        lag=1,
        out_column_names="autocorr_dynamic",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_dynamic"].to_numpy()

    # Ожидаемые значения для динамического окна
    expected = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            -1,
            -1,
            -0.32733,
            np.nan,
            np.nan,
            np.nan,
            -1,
            -1,
            -0.72058,
        ]
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_invalid_lag(sample_dataset):
    """Тестирует обработку недопустимого lag (должен быть >= 1)."""
    with pytest.raises(ValueError):
        Autocorrelation(
            columns="value",
            window_types=WindowType.EXPANDING,
            lag=0,
            out_column_names="autocorr_invalid",
        )
