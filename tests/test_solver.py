import src.solver as solver


def test_create_edges_cached() -> None:
    size = solver.PuzzleSize(2, 2)
    edges1 = solver._create_edges(size)
    edges2 = solver._create_edges(size)
    assert edges1 is edges2
