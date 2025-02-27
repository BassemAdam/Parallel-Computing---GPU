import numpy as np

def generate_3d_matrix_test_cases(num_tests, min_dim, max_dim, min_val, max_val, output_file):
    """
    Generate test cases for 3D matrix addition
    Parameters:
    - num_tests: number of test cases
    - min_dim: minimum dimension (cols/rows/depth)
    - max_dim: maximum dimension (cols/rows/depth)
    - min_val: minimum value in matrices
    - max_val: maximum value in matrices
    - output_file: path to output file
    """
    with open(output_file, 'w') as f:
        # Write number of test cases
        f.write(f"{num_tests}\n")
        
        for _ in range(num_tests):
            # Generate random dimensions
            cols = np.random.randint(min_dim, max_dim + 1)
            rows = np.random.randint(min_dim, max_dim + 1)
            depth = np.random.randint(min_dim, max_dim + 1)
            
            # Write dimensions
            f.write(f"{cols} {rows} {depth}\n")
            
            # Generate and write first 3D matrix
            matrix1 = np.random.uniform(min_val, max_val, (cols, rows, depth))
            for layer in matrix1:
                for row in layer:
                    f.write(" ".join(f"{x:.3f}" for x in row) + "\n")
            
            # Generate and write second 3D matrix
            matrix2 = np.random.uniform(min_val, max_val, (cols, rows, depth))
            for layer in matrix2:
                for row in layer:
                    f.write(" ".join(f"{x:.3f}" for x in row) + "\n")

# Set your parameters here
params = {
    'num_tests': 1,              # Number of test cases
    'min_dim': 100,                # Minimum matrix dimension
    'max_dim': 100,                # Maximum matrix dimension
    'min_val': -500.0,      # Minimum value in matrices
    'max_val': 500.0,     # Maximum value in matrices
    'output_file': './Input_TestCases/t2_50.txt'  # Output file name
}

# Run the generator with the specified parameters
if __name__ == "__main__":
    generate_3d_matrix_test_cases(**params)