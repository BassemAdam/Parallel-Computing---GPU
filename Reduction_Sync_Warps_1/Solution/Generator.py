import numpy as np

def generate_1d_array_test_cases(num_tests, sizes, min_val, max_val, output_file):
    """
    Generate test cases for 1D array max element finding
    Parameters:
    - num_tests: number of test cases
    - sizes: list of array sizes to generate
    - min_val: minimum value in arrays
    - max_val: maximum value in arrays
    - output_file: path to output file
    """
    with open(output_file, 'w') as f:
        # Write number of test cases
        f.write(f"{num_tests}\n")
        
        for i in range(num_tests):
            # Pick a size for this test case
            array_size = sizes[i % len(sizes)]
            
            # Write array size
            f.write(f"{array_size}\n")
            
            # Generate random 1D array
            array = np.random.uniform(min_val, max_val, array_size)
            
            # Ensure we have at least one distinctive max value
            max_pos = np.random.randint(0, array_size)
            max_value = max_val - np.random.uniform(0, 10)  # Close to max_val
            array[max_pos] = max_value
            
            # Write array values, formatted nicely
            values_per_line = 10
            for j in range(0, array_size, values_per_line):
                end = min(j + values_per_line, array_size)
                line_values = array[j:end]
                line = " ".join(f"{x:.6f}" for x in line_values)
                f.write(f"{line}\n")

# Set your parameters here
params = {
    'num_tests': 1,               # Number of test cases
    'sizes': [1000],  # Array sizes
    'min_val': -500.0,            # Minimum value in arrays
    'max_val': 500.0,             # Maximum value in arrays
    'output_file': './Input_TestCases/max_element_tests.txt'  # Output file name
}

# Run the generator
if __name__ == "__main__":
    generate_1d_array_test_cases(**params)
    print(f"Generated {params['num_tests']} test cases with sizes: {params['sizes']}")
    print(f"Output written to {params['output_file']}")