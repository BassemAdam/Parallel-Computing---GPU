# CUDA DES Encryption Implementation

This project implements the Data Encryption Standard (DES) algorithm using CUDA for GPU acceleration. The implementation focuses on Electronic Codebook (ECB) mode encryption, where each 64-bit block is encrypted independently.

## Overview

DES is a symmetric-key block cipher that operates on 64-bit blocks using a 56-bit key. Despite being considered insecure for modern applications due to its small key size, DES remains an important algorithm for educational purposes and understanding the foundations of modern cryptography.

## Implementation Details

### Parallel Processing Strategy

Our implementation takes advantage of GPU parallelism by:
- Assigning one thread to process one 64-bit block of plaintext
- Using CUDA's parallel architecture to encrypt multiple blocks simultaneously
- Leveraging constant memory for lookup tables to improve cache hit rates

### Workflow

1. **Input Processing**:
   - The program reads text from a user-specified input file
   - Text is converted into 64-bit blocks, with zero-padding for incomplete blocks

2. **Key Scheduling**:
   - A 56-bit key is used to generate 16 round subkeys
   - Each subkey is 48 bits and derived through permutation and bit rotation operations

3. **Encryption Process**:
   - Initial Permutation (IP) rearranges the 64-bit input block
   - 16 rounds of Feistel network operations are applied
   - Final Permutation (FP) produces the ciphertext

4. **Output**:
   - Encrypted blocks are written to a user-specified output file

## Performance

By leveraging GPU parallelism, this implementation can encrypt large volumes of data much faster than a CPU-based approach. Each thread processes a single block independently, allowing the GPU to handle thousands of blocks concurrently.

## Usage

```
./des_cuda <input_file> <output_file>
```

Where:
- `input_file`: Path to the plaintext file to be encrypted
- `output_file`: Path where the encrypted output will be saved

## Implementation Components

- **DES Tables**: Permutation and substitution tables stored in constant memory
- **Kernel Function**: CUDA kernel for parallel block encryption
- **File I/O**: Routines to read plaintext and write ciphertext
- **Block Processing**: Functions to convert between text and 64-bit blocks
- **Error Handling**: Robust error checking for CUDA operations and file I/O
