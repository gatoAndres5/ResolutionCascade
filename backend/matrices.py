import json



def create_matrix():
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    # Save to JSON file
    with open("matrix_output.json", "w") as f:
        json.dump(matrix, f)

    print("Matrix saved to matrix_output.json")

def doSomething():
    result = []
    for row_index in range(1000):
        row = []
        for col_index in range(1000):
            row.append(col_index * 5)  # Multiply x-coordinate (column) by 5
        result.append(row)

    with open("matrix_modified.json", "w") as f:
        json.dump(result, f)
    print("Modified matrix saved to matrix_modified.json")
    

if __name__ == "__main__":
    create_matrix()
    doSomething()

