import numpy as np

def _neighbors_4(matrix, r, c):
    rows, cols = matrix.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def _neighbors_8(matrix, r, c):
    rows, cols = matrix.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

def adjacency_4(matrix):
    rows, cols = matrix.shape
    adj = {}
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 1:
                continue
            adj[(r, c)] = [(nr, nc) for nr, nc in _neighbors_4(matrix, r, c) if matrix[nr, nc] == 1]
    return adj

def adjacency_8(matrix):
    rows, cols = matrix.shape
    adj = {}
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 1:
                continue
            adj[(r, c)] = [(nr, nc) for nr, nc in _neighbors_8(matrix, r, c) if matrix[nr, nc] == 1]
    return adj

def adjacency_m(matrix):
    rows, cols = matrix.shape
    adj = {}
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 1:
                continue
            neighbors = []
            # 4-adjacent always included
            for nr, nc in _neighbors_4(matrix, r, c):
                if matrix[nr, nc] == 1:
                    neighbors.append((nr, nc))
            # diagonal neighbors if no common 4-neighbor with value 1
            for nr, nc in _neighbors_8(matrix, r, c):
                if abs(nr - r) == 1 and abs(nc - c) == 1 and matrix[nr, nc] == 1:
                    common = set(_neighbors_4(matrix, r, c)).intersection(_neighbors_4(matrix, nr, nc))
                    if not any(matrix[cr, cc] == 1 for cr, cc in common):
                        neighbors.append((nr, nc))
            adj[(r, c)] = neighbors
    return adj

test_matrix = np.array([[1, 0, 1],
                        [0, 1, 1], 
                        [1, 0, 1]])

adj4 = adjacency_4(test_matrix)
adj8 = adjacency_8(test_matrix)
adjm = adjacency_m(test_matrix)

# adj4.get(key, []), adj8.get(key, []), adjm.get(key, [])
print("4-adjacency:", adj4)
print("8-adjacency:", adj8)
print("M-adjacency:", adjm)