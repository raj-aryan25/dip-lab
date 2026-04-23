def find_neighbors(matrix, target):
	rows, cols = len(matrix), len(matrix[0])
	tr, tc = target
	four = []
	diagonal = []
	eight = []
	for dr in [-1, 0, 1]:
		for dc in [-1, 0, 1]:
			if dr == 0 and dc == 0:
				continue
			nr, nc = tr + dr, tc + dc
			if 0 <= nr < rows and 0 <= nc < cols:
				neighbor = (nr, nc)
				eight.append(neighbor)
				if abs(dr) + abs(dc) == 1:
					four.append(neighbor)
				elif abs(dr) == 1 and abs(dc) == 1:
					diagonal.append(neighbor)
	return four, diagonal, sorted(eight)

def visualize(matrix, target, four, diagonal):
	print("\nVisualization:")
	print("T = Target | F = 4-point | D = Diagonal | others = value\n")
	for r in range(len(matrix)):
		for c in range(len(matrix[0])):
			if (r, c) == target:
				print("T ", end="")
			elif (r, c) in four:
				print("F ", end="")
			elif (r, c) in diagonal:
				print("D ", end="")
			else:
				print(f"{matrix[r][c]} ", end="")
		print()
		
def main():
	rows = int(input("Enter number of rows: "))
	cols = int(input("Enter number of columns: "))
	print("\nEnter matrix rows (space-separated):")
	matrix = []
	for i in range(rows):
		row = list(map(int, input(f"Row {i}: ").split()))
		if len(row) != cols:
			print("Invalid row length. Exiting.")
			return
		matrix.append(row)
	print("\nMatrix:")
	for row in matrix:
		print(row)
	tr = int(input("\nEnter target row: "))
	tc = int(input("Enter target column: "))
	if not (0 <= tr < rows and 0 <= tc < cols):
		print("Target pixel out of bounds.")
		return
	target = (tr, tc)
	four, diagonal, eight = find_neighbors(matrix, target)
	print(f"\n4-point neighbors: {four}")
	print(f"Diagonal neighbors: {diagonal}")
	print(f"8-point neighbors: {eight}")
	visualize(matrix, target, four, diagonal)

if __name__ == "__main__":
    main()