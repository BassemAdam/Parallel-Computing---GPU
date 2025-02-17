import numpy as np

def generate_matrix_test_cases(num_tests, min_dim, max_dim, min_val, max_val, output_file):
    """
    Generate test cases for matrix addition
    Parameters:
    - num_tests: number of test cases
    - min_dim: minimum dimension (rows/cols)
    - max_dim: maximum dimension (rows/cols)
    - min_val: minimum value in matrices
    - max_val: maximum value in matrices
    - output_file: path to output file
    """
    with open(output_file, 'w') as f:
        # Write number of test cases
        f.write(f"{num_tests}\n")
        
        for _ in range(num_tests):
            # Generate random dimensions
            rows = np.random.randint(min_dim, max_dim + 1)
            cols = np.random.randint(min_dim, max_dim + 1)
            
            # Write dimensions
            f.write(f"{rows} {cols}\n")
            
            # Generate and write first matrix
            matrix1 = np.random.uniform(min_val, max_val, (rows, cols))
            for row in matrix1:
                f.write(" ".join(f"{x:.3f}" for x in row) + "\n")
            
            # Generate and write second matrix
            matrix2 = np.random.uniform(min_val, max_val, (rows, cols))
            for row in matrix2:
                f.write(" ".join(f"{x:.3f}" for x in row) + "\n")

# Set your parameters here
params = {
    'num_tests': 3,              # Number of test cases
    'min_dim': 2,               # Minimum matrix dimension
    'max_dim': 4,               # Maximum matrix dimension
    'min_val': -5.0,           # Minimum value in matrices
    'max_val': 5.0,            # Maximum value in matrices
    'output_file': './matrix_tests.txt'  # Output file name
}

# Run the generator with the specified parameters
if __name__ == "__main__":
    generate_matrix_test_cases(**params)