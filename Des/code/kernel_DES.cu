/* des_cuda.cu
 * CUDA implementation of DES encryption/decryption in ECB mode.
 * One thread handles one 64-bit block.
*/
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>

// ================= DES Tables =================
// Initial Permutation (IP)
__device__ __constant__ uint8_t IP[64] = {
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7};
// Final Permutation (FP)
__device__ __constant__ uint8_t FP[64] = {
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25};
// Expansion table (E)
__device__ __constant__ uint8_t E[48] = {
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1};
// Permutation (P)
__device__ __constant__ uint8_t P[32] = {
    16, 7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2, 8, 24, 14,
    32, 27, 3, 9,
    19, 13, 30, 6,
    22, 11, 4, 25};
// S-boxes
__device__ __constant__ uint8_t SBOX[8][64] = {
    // S1
    {14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13},
    // S2
    {15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9},
    // S3
    {10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
     1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12},
    // S4
    {7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14},
    // S5
    {2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3},
    // S6
    {12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13},
    // S7
    {4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12},
    // S8
    {13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}};

// PC-1 and PC-2 tables
__device__ __constant__ uint8_t PC1[56] = {
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4};
__device__ __constant__ uint8_t PC2[48] = {
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32};
__device__ __constant__ uint8_t SHIFTS[16] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};

// Round subkeys
__device__ __constant__ uint64_t d_subkeys[16];

// Helper: apply permutation
__device__ uint64_t permute(uint64_t in, const uint8_t *table, int n){
    uint64_t out = 0;
    for (int i = 0; i < n; i++)
    {
        uint64_t bit = (in >> (64 - table[i])) & 1ULL;
        out |= bit << (n - 1 - i);
    }
    return out;
}

// Feistel function
__device__ uint32_t feistel(uint32_t R, uint64_t subkey){
    // Expansion
    uint64_t eR = permute((uint64_t)R << 32, E, 48);
    // Key mixing
    uint64_t x = eR ^ subkey;
    // S-box substitution
    uint32_t out = 0;
    for (int i = 0; i < 8; i++)
    {
        uint8_t six = (x >> (42 - 6 * i)) & 0x3F;
        uint8_t row = ((six & 0x20) >> 4) | (six & 1);
        uint8_t col = (six >> 1) & 0xF;
        uint8_t s = SBOX[i][row * 16 + col];
        out |= ((uint32_t)s) << (28 - 4 * i);
    }
    // Permutation P
    return (uint32_t)permute((uint64_t)out << 32, P, 32);
}

// DES kernel: one thread per 64-bit block
__global__ void des_encrypt_kernel(const uint64_t *d_in,uint64_t *d_out,size_t num_blocks){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks)
        return;

    uint64_t block = d_in[idx];
    // Initial Permutation
    block = permute(block, IP, 64);

    uint32_t L = block >> 32;
    uint32_t R = block & 0xFFFFFFFF;
    for (int rnd = 0; rnd < 16; rnd++)
    {
        uint32_t tmp = R;
        R = L ^ feistel(R, d_subkeys[rnd]);
        L = tmp;
    }
    // Pre-output: swap L and R
    uint64_t preout = ((uint64_t)R << 32) | L;
    // Final Permutation
    d_out[idx] = permute(preout, FP, 64);
}

// Helper: Convert host permutation function for key generation
uint64_t host_permute(uint64_t in, const uint8_t *table, int n){
    uint64_t out = 0;
    for (int i = 0; i < n; i++)
    {
        uint64_t bit = (in >> (64 - table[i])) & 1ULL;
        out |= bit << (n - 1 - i);
    }
    return out;
}

// Host version of PC1 and PC2 tables
uint8_t h_PC1[56] = {
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4};
uint8_t h_PC2[48] = {
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32};
uint8_t h_SHIFTS[16] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};

// Host: generate subkeys and copy to device constant memory
void generate_subkeys(uint64_t key, uint64_t *subkeys, bool decrypt){
    // PC-1 permutation
    uint64_t perm = host_permute(key, h_PC1, 56);
    uint32_t C = (perm >> 28) & 0x0FFFFFFF;
    uint32_t D = perm & 0x0FFFFFFF;

    // Generate all subkeys in order
    uint64_t temp_subkeys[16];
    for (int i = 0; i < 16; i++)
    {
        // Left rotations
        C = ((C << h_SHIFTS[i]) | (C >> (28 - h_SHIFTS[i]))) & 0x0FFFFFFF;
        D = ((D << h_SHIFTS[i]) | (D >> (28 - h_SHIFTS[i]))) & 0x0FFFFFFF;
        uint64_t CD = ((uint64_t)C << 28) | D;
        temp_subkeys[i] = host_permute(CD << 8, h_PC2, 48);
    }

    // For decryption, reverse the order of subkeys
    if (decrypt)
    {
        for (int i = 0; i < 16; i++)
        {
            subkeys[i] = temp_subkeys[15 - i];
        }
    }
    else
    {
        // For encryption, use the subkeys in original order
        memcpy(subkeys, temp_subkeys, 16 * sizeof(uint64_t));
    }
}

int main(int argc, char *argv[]){
    // Check command line arguments
    if (argc != 4)
    {
        printf("Usage: %s <encrypt|decrypt> <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *input_filename = argv[2];
    const char *output_filename = argv[3];

    // Check if mode is valid
    bool decrypt = false;
    if (strcmp(mode, "encrypt") == 0)
    {
        decrypt = false;
    }
    else if (strcmp(mode, "decrypt") == 0)
    {
        decrypt = true;
    }
    else
    {
        printf("Error: Mode must be either 'encrypt' or 'decrypt'\n");
        return 1;
    }

    // Open input file - use binary mode
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file)
    {
        printf("Error: Could not open input file %s\n", input_filename);
        return 1;
    }

    // Get file size
    fseek(input_file, 0, SEEK_END);
    size_t file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    if (file_size == 0)
    {
        printf("Error: Input file is empty\n");
        fclose(input_file);
        return 1;
    }

    // Read the content
    char *file_content = (char *)malloc(file_size);
    if (!file_content)
    {
        printf("Error: Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }

    // For reading binary data, don't null-terminate
    size_t bytes_read = fread(file_content, 1, file_size, input_file);
    fclose(input_file);

    size_t padded_size = 0;
    size_t num_blocks = 0;

    if (!decrypt)
    {
        // Encryption: apply PKCS#5/PKCS#7 padding
        uint8_t pad_len = 8 - (bytes_read % 8);
        padded_size = bytes_read + pad_len;
        num_blocks = padded_size / 8;
    }
    else
    {
        // Decryption: no padding yet, just blockify
        padded_size = bytes_read;
        num_blocks = (bytes_read + 7) / 8;
    }

    uint64_t *h_in = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));

    if (!h_in || !h_out)
    {
        printf("Error: Memory allocation failed\n");
        free(file_content);
        return 1;
    }

    // Initialize blocks with zeros
    memset(h_in, 0, num_blocks * sizeof(uint64_t));

    if (!decrypt)
    {
        // Encryption: copy and pad
        size_t i;
        for (i = 0; i < bytes_read; i++)
        {
            size_t block_idx = i / 8;
            size_t bit_pos = (i % 8) * 8;
            h_in[block_idx] |= ((uint64_t)(unsigned char)file_content[i]) << (56 - bit_pos);
        }
        // Add padding bytes
        uint8_t pad_len = 8 - (bytes_read % 8);
        for (size_t j = 0; j < pad_len; j++)
        {
            size_t pad_i = bytes_read + j;
            size_t block_idx = pad_i / 8;
            size_t bit_pos = (pad_i % 8) * 8;
            h_in[block_idx] |= ((uint64_t)pad_len) << (56 - bit_pos);
        }
    }
    else
    {
        // Decryption: just copy as before
        for (size_t i = 0; i < bytes_read; i++)
        {
            size_t block_idx = i / 8;
            size_t bit_pos = (i % 8) * 8;
            h_in[block_idx] |= ((uint64_t)(unsigned char)file_content[i]) << (56 - bit_pos);
        }
    }

    free(file_content);

    // Host subkeys
    uint64_t h_subkeys[16];
    uint64_t key = 0x133457799BBCDFF1ULL;
    generate_subkeys(key, h_subkeys, decrypt);
    cudaMemcpyToSymbol(d_subkeys, h_subkeys, 16 * sizeof(uint64_t));

    // Device buffers
    uint64_t *d_in, *d_out;

    cudaMalloc(&d_in, num_blocks * sizeof(uint64_t));

    cudaMalloc(&d_out, num_blocks * sizeof(uint64_t));

    cudaMemcpy(d_in, h_in, num_blocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t threads = 256;
    size_t blocks = (num_blocks + threads - 1) / threads;
    printf("%s %zu blocks with %zu threads in %zu cuda blocks...\n",
           decrypt ? "Decrypting" : "Encrypting", num_blocks, threads, blocks);

    des_encrypt_kernel<<<blocks, threads>>>(d_in, d_out, num_blocks);

    cudaMemcpy(h_out, d_out, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Write output
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file)
    {
        printf("Error: Could not open output file %s\n", output_filename);
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        return 1;
    }

    if (decrypt)
    {
        // Convert uint64_t array back to bytes for writing
        unsigned char *output_bytes = (unsigned char *)malloc(num_blocks * 8);
        if (!output_bytes)
        {
            printf("Error: Memory allocation failed\n");
            fclose(output_file);
            return 1;
        }

        // Convert the uint64_t blocks back to bytes
        for (size_t i = 0; i < num_blocks * 8; i++)
        {
            size_t block_idx = i / 8;
            size_t byte_pos = i % 8;
            output_bytes[i] = (h_out[block_idx] >> (56 - byte_pos * 8)) & 0xFF;
        }

        // Remove PKCS#5/PKCS#7 padding
        if (num_blocks * 8 == 0)
        {
            printf("Error: Decrypted data is empty\n");
            free(output_bytes);
            fclose(output_file);
            return 1;
        }
        uint8_t pad_len = output_bytes[num_blocks * 8 - 1];
        if (pad_len == 0 || pad_len > 8)
        {
            printf("Error: Invalid padding in decrypted data\n");
            free(output_bytes);
            fclose(output_file);
            return 1;
        }
        // Check all padding bytes
        int valid_padding = 1;
        for (size_t i = 0; i < pad_len; i++)
        {
            if (output_bytes[num_blocks * 8 - 1 - i] != pad_len)
            {
                valid_padding = 0;
                break;
            }
        }
        if (!valid_padding)
        {
            printf("Error: Invalid padding in decrypted data\n");
            free(output_bytes);
            fclose(output_file);
            return 1;
        }
        size_t plaintext_len = num_blocks * 8 - pad_len;

        // Write only the plaintext (unpadded)
        size_t bytes_written = fwrite(output_bytes, 1, plaintext_len, output_file);
        free(output_bytes);

        if (bytes_written != plaintext_len)
        {
            printf("Error: Failed to write all data to output file\n");
            fclose(output_file);
            return 1;
        }
    }
    else
    {
        // Encryption: write big‑endian bytes instead of raw uint64_t
        size_t total_bytes = num_blocks * 8;
        unsigned char *out_bytes = (unsigned char *)malloc(total_bytes);
        if (!out_bytes)
        {
            printf("Error: Memory allocation failed\n");
            fclose(output_file);
            return 1;
        }
        for (size_t i = 0; i < total_bytes; i++)
        {
            size_t blk = i / 8;
            size_t pos = i % 8;
            out_bytes[i] = (h_out[blk] >> (56 - pos * 8)) & 0xFF;
        }
        size_t bytes_written = fwrite(out_bytes, 1, total_bytes, output_file);
        free(out_bytes);
        if (bytes_written != total_bytes)
        {
            printf("Error: Failed to write all data to output file\n");
            fclose(output_file);
            return 1;
        }
    }

    printf("%s complete. Output written to %s\n",
           decrypt ? "Decryption" : "Encryption", output_filename);
    fclose(output_file);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
