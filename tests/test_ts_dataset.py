import polars as pl
import pytest

from chrono_features.ts_dataset import TSDataset


def test_sort_by_id_and_ts():
    data = pl.DataFrame(data={"id": [2, 2, 1, 1], "ts": [3, 2, 1, 0], "value": [10, 20, 30, 40]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    expected_sorted_data = pl.DataFrame(data={"id": [1, 1, 2, 2], "ts": [0, 1, 2, 3], "value": [40, 30, 20, 10]})

    assert dataset.data.equals(expected_sorted_data)


def test_sort_already_sorted():
    data = pl.DataFrame(data={"id": [1, 1, 2, 2], "ts": [0, 1, 2, 3], "value": [40, 30, 20, 10]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    assert dataset.data.equals(other=data)


def test_sort_with_non_numeric_id():
    data = pl.DataFrame(data={"id": ["b", "b", "a", "a"], "ts": [3, 2, 1, 0], "value": [10, 20, 30, 40]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    expected_sorted_data = pl.DataFrame(
        data={"id": ["a", "a", "b", "b"], "ts": [0, 1, 2, 3], "value": [40, 30, 20, 10]},
    )

    assert dataset.data.equals(other=expected_sorted_data)


def test_add_feature():
    data = pl.DataFrame(data={"id": [1, 2, 3], "ts": [100, 200, 300], "value": [10, 20, 30]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    new_column_name = "new_feature"
    new_values = [100, 200, 300]
    dataset.add_feature(name=new_column_name, values=new_values)

    expected_data = pl.DataFrame(
        data={
            "id": [1, 2, 3],
            "ts": [100, 200, 300],
            "value": [10, 20, 30],
            "new_feature": [100, 200, 300],
        },
    )

    assert dataset.data.equals(other=expected_data)


def test_add_feature_with_wrong_length():
    data = pl.DataFrame(data={"id": [1, 2, 3], "ts": [100, 200, 300], "value": [10, 20, 30]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    new_feature_name = "new_feature"
    new_values = [100, 200]

    with pytest.raises(expected_exception=ValueError) as exc_info:
        dataset.add_feature(name=new_feature_name, values=new_values)

    assert str(exc_info.value) == "Length of values does not match the dataset length."


def test_add_feature_existing_column():
    data = pl.DataFrame(data={"id": [1, 2, 3], "ts": [100, 200, 300], "value": [10, 20, 30]})

    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="ts")

    existing_column_name = "value"
    new_values = [100, 200, 300]

    with pytest.raises(expected_exception=ValueError) as exc_info:
        dataset.add_feature(name=existing_column_name, values=new_values)

    assert str(exc_info.value) == "Column 'value' already exists in the dataset"
