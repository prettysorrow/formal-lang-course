"""Tests for task 1."""

import pytest
from networkx.exception import NetworkXError


from project.task1 import (
    GraphDescription,
    download_graph_description_from_cfpq_data,
    store_two_cycles_graph_to_dot_file,
    load_graph_description_from_dot_file,
)

# positive tests for download_graph_description_from_cfpq_data


def test_download_graph_description_bzip():
    description = download_graph_description_from_cfpq_data("bzip")

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 632
    assert description.edges_count == 556
    assert description.labels == {"a", "d"}


def test_download_graph_description_wc():
    description = download_graph_description_from_cfpq_data("wc")

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 332
    assert description.edges_count == 269
    assert description.labels == {"a", "d"}


def test_download_graph_description_gzip():
    description = download_graph_description_from_cfpq_data("gzip")

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 2687
    assert description.edges_count == 2293
    assert description.labels == {"a", "d"}


# negative tests for download_graph_description_from_cfpq_data


def test_download_graph_description_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        download_graph_description_from_cfpq_data("no_such_graph")


# edge cases analysis for store_two_cycles_graph_to_dot_file


def test_store_two_cycles_graph_first_cycle_size_negative_raises(tmp_path):
    with pytest.raises(NetworkXError, match="Negative number of nodes"):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=-1,
            second_cycle_size=4,
            labels=("a", "b"),
            path=tmp_path / "graph.dot",
        )


def test_store_two_cycles_graph_first_cycle_size_zero_raises(tmp_path):
    with pytest.raises(IndexError):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=0,
            second_cycle_size=4,
            labels=("a", "b"),
            path=tmp_path / "graph.dot",
        )


def test_store_two_cycles_graph_first_cycle_size_one(tmp_path):
    path = tmp_path / "two_cycles.dot"
    store_two_cycles_graph_to_dot_file(
        first_cycle_size=1,
        second_cycle_size=4,
        labels=("x", "y"),
        path=path,
    )
    description = load_graph_description_from_dot_file(path)

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 1 + 4 + 1
    assert description.edges_count == 1 + 4 + 2
    assert description.labels == {"x", "y"}


def test_store_two_cycles_graph_second_cycle_size_negative_raises(tmp_path):
    with pytest.raises(NetworkXError, match="Negative number of nodes"):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=3,
            second_cycle_size=-1,
            labels=("a", "b"),
            path=tmp_path / "graph.dot",
        )


def test_store_two_cycles_graph_second_cycle_size_zero_raises(tmp_path):
    with pytest.raises(IndexError):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=3,
            second_cycle_size=0,
            labels=("a", "b"),
            path=tmp_path / "graph.dot",
        )


def test_store_two_cycles_graph_second_cycle_size_one(tmp_path):
    path = tmp_path / "two_cycles.dot"
    store_two_cycles_graph_to_dot_file(
        first_cycle_size=3,
        second_cycle_size=1,
        labels=("x", "y"),
        path=path,
    )
    description = load_graph_description_from_dot_file(path)

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 3 + 1 + 1
    assert description.edges_count == 3 + 1 + 2
    assert description.labels == {"x", "y"}


# negative tests for store_two_cycles_graph_to_dot_file


def test_store_two_cycles_graph_labels_length_one_raises(tmp_path):
    with pytest.raises(IndexError):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=3,
            second_cycle_size=4,
            labels=("a",),
            path=tmp_path / "graph.dot",
        )


def test_store_two_cycles_graph_labels_none_raises(tmp_path):
    with pytest.raises(TypeError):
        store_two_cycles_graph_to_dot_file(
            first_cycle_size=3,
            second_cycle_size=4,
            labels=None,
            path=tmp_path / "graph.dot",
        )


# positive tests for store_two_cycles_graph_to_dot_file and load_graph_description_from_dot_file


def test_load_graph_description_round_trip_small(tmp_path):
    path = tmp_path / "two_cycles.dot"
    store_two_cycles_graph_to_dot_file(
        first_cycle_size=3,
        second_cycle_size=4,
        labels=("x", "y"),
        path=path,
    )
    description = load_graph_description_from_dot_file(path)

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 3 + 4 + 1
    assert description.edges_count == 3 + 4 + 2
    assert description.labels == {"x", "y"}


def test_load_graph_description_round_trip_medium(tmp_path):
    path = tmp_path / "two_cycles.dot"
    store_two_cycles_graph_to_dot_file(
        first_cycle_size=550,
        second_cycle_size=690,
        labels=("x", "y"),
        path=path,
    )
    description = load_graph_description_from_dot_file(path)

    assert isinstance(description, GraphDescription)
    assert description.nodes_count == 550 + 690 + 1
    assert description.edges_count == 550 + 690 + 2
    assert description.labels == {"x", "y"}


# negative tests for load_graph_description_from_dot_file


def test_load_graph_description_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph_description_from_dot_file(tmp_path / "missing.dot")


def test_load_graph_description_from_dot_file_malformed_dot_raises(tmp_path):
    path = tmp_path / "malformed.dot"
    path.write_text("<<< this is not dot >>>")

    with pytest.raises(TypeError):
        load_graph_description_from_dot_file(path)


def test_load_graph_description_from_dot_file_empty_file_raises(tmp_path):
    path = tmp_path / "empty.dot"
    path.write_text("")

    with pytest.raises(TypeError):
        load_graph_description_from_dot_file(path)
