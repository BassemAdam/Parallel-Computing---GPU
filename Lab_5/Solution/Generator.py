import numpy as np

def generate_1d_convolution_test_cases(vector_sizes, mask_sizes, min_val, max_val, output_dir):
    """
    Generate test cases for 1D convolution
    Parameters:
    - vector_sizes: list of vector sizes to generate
    - mask_sizes: list of mask sizes to generate
    - min_val: minimum value in vectors and masks
    - max_val: maximum value in vectors and masks
    - output_dir: directory to save test cases
    
    For each combination of vector_size and mask_size:
    - Creates an input vector file
    - Creates a mask file
    - Creates an expected output file (computed using NumPy)
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for vector_size in vector_sizes:
        for mask_size in mask_sizes:
            # Skip if mask is larger than vector
            if mask_size > vector_size:
                continue
                
            # Generate input vector
            vector = np.random.uniform(min_val, max_val, vector_size)
            
            # Generate mask (convolution kernel)
            mask = np.random.uniform(min_val, max_val, mask_size)
            
            # Compute expected output using NumPy's convolution
            # Note: mode='same' keeps the output size same as input
            output = np.convolve(vector, mask, mode='same')
            mask_reversed = mask[::-1]  # Reverse the mask
            output = np.convolve(vector, mask_reversed, mode='same')   
                
            # Create filenames
            base_name = f"conv_v{vector_size}_m{mask_size}"
            vector_file = os.path.join(output_dir, f"{base_name}_input.txt")
            mask_file = os.path.join(output_dir, f"{base_name}_mask.txt")
            expected_output_file = os.path.join(output_dir, f"{base_name}_expected_output.txt")
            
            # Write input vector to file
            with open(vector_file, 'w') as f:
                f.write(f"{vector_size}\n")
                f.write(" ".join(f"{x:.6f}" for x in vector))
            
            # Write mask to file
            with open(mask_file, 'w') as f:
                f.write(f"{mask_size}\n")
                f.write(" ".join(f"{x:.6f}" for x in mask))
            
            # Write expected output to file
            with open(expected_output_file, 'w') as f:
                f.write(" ".join(f"{x:.6f}" for x in output))
            
            print(f"Generated test case: {base_name}")
            print(f"  - Vector size: {vector_size}")
            print(f"  - Mask size: {mask_size}")

# Parameters for 1D convolution test cases
params_1d = {
    'vector_sizes': [1000, 10000, 100000, 1000000], # Different vector sizes to test
    'mask_sizes': [3, 5, 9, 15, 31],                # Different mask sizes to test
    'min_val': -10.0,                               # Minimum value in vectors/masks
    'max_val': 10.0,                                # Maximum value in vectors/masks
    'output_dir': './Solution/Generator_TestCases/Convolution'   # Output directory
}

# Run the generators with the specified parameters
if __name__ == "__main__":
    # Generate 1D convolution test cases
    generate_1d_convolution_test_cases(**params_1d)
    print("Test case generation complete!")