from utils.data_utils import get_neighbors
from utils.data_utils import construct_adjacency_list_core
import numpy as np 

def test_get_neighbors():
    rows, cols = 24, 72

    # Test for "grid" method
    expected_grid_neighbors = [(1, 0), (0, 71), (0, 1)]
    grid_neighbors = get_neighbors(rows, cols, 0, 0, "grid")
    assert sorted(grid_neighbors) == sorted(expected_grid_neighbors), f"Grid neighbors incorrect: {grid_neighbors}"

    # Test for "dense_grid" method
    expected_dense_neighbors = [(1, 0), (0, 71), (0, 1), (1, 71), (1, 1)]
    dense_neighbors = get_neighbors(rows, cols, 0, 0, "dense_grid")
    assert sorted(dense_neighbors) == sorted(expected_dense_neighbors), f"Dense grid neighbors incorrect: {dense_neighbors}"

    print("All tests passed for base neighbors!")

def test_scale_4_grid():
    rows, cols = 24, 72
    scales = [4]
    origins = [(0, 0)]

    adjacency_list = construct_adjacency_list_core(grid_size=(24,72), method="grid", scales=scales, origins=origins)

    # Check edges for node (0,0)
    node_0 = 0
    scale_4_neighbors = [(0, 4), (4, 0), (0, 68)]  # Expected neighbors at scale 4
    scale_4_edges = [(node_0, n[0] * cols + n[1]) for n in scale_4_neighbors]

    for edge in scale_4_edges:
        assert any((edge[0] == e[0] and edge[1] == e[1]) or (edge[0] == e[1] and edge[1] == e[0]) for e in adjacency_list.T), \
            f"Edge {edge} not found in adjacency list for scale 4."

    # Check that no new scale 4 edges are added to (1,0) or (0,1)
    node_1_0 = 1 * cols + 0
    node_0_1 = 0 * cols + 1

    for edge in adjacency_list.T:
        assert edge[0] != node_1_0 and edge[1] != node_1_0, \
            f"Unexpected edge {edge} involving node (1,0)."
        assert edge[0] != node_0_1 and edge[1] != node_0_1, \
            f"Unexpected edge {edge} involving node (0,1)."

    print("All tests passed for scale 4 grid!")

def test_union_over_scales():
    adjacency_list_a = construct_adjacency_list_core(grid_size=(24,72), method="grid", scales=[1], origins=[(0,0)])
    adjacency_list_b = construct_adjacency_list_core(grid_size=(24,72), method="grid", scales=[3], origins=[(1,1)])

    adjacency_list_ab = construct_adjacency_list_core(grid_size=(24,72), method="grid", scales=[1, 3], origins=[(0,0), (1,1)])
    
    adjacency_list_concat = np.concatenate([adjacency_list_a, adjacency_list_b], axis=1)

    normalized_concat = np.sort(adjacency_list_concat, axis=0)  # Shape (2, N+M)
    normalized_ab = np.sort(adjacency_list_ab, axis=0)          # Shape (2, N+M)

    set_concat = set(map(tuple, normalized_concat.T))
    set_ab = set(map(tuple, normalized_ab.T))

    # Check if they are the same
    assert set_concat == set_ab

    print("All tests passed for union over adjacency lists at two scales!")


def main():
    test_get_neighbors()
    test_scale_4_grid()
    test_union_over_scales()

if __name__ == "__main__":
    main()