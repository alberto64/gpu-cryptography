/**
 * @file	cuda_cryptography.c
 * @author	Alberto De Jesus
 * @brief Using cuda to handle large files and do cryptography
 */

#include <stdio.h>
#include <time.h>
#include <string.h>

/**
 * usage: a method to depict the usaged of the command file.
 */
void usage(const char *command) {
    printf("Usage: %s <inputFile> <outputFile> <key>\n", command);
    exit(0);
}

/**
 * read_file_contents: given a filepath it reads the contents of a file.
 */
int* read_file_contents(char* filepath) {

    // Open file
    FILE* file = fopen(filepath, "rb");

    // Check that opening was a success
    if(file == NULL){
        printf("Error in opening file\n");
        exit(0);
    }

    // Get the total amount of bytes in the file and make an array from it
    fseek(file, 0L, SEEK_END);
    int* contents = (int*) malloc(sizeof(int) * ftell(file));
    rewind(file);

    // Read the contents of the file and save it on the array created.
    int idx = 0;
    while ((contents[idx] = fgetc(file)) != EOF) {
        idx = idx + 1;
    }

    // Close file
    fclose(file);

    return contents;
}

/**
 * write_file_contents: given a filepath and some bytes it writes the bytes on the file.
 */
int* write_file_contents(char* filepath, int* contents) {

    // Open file
    FILE* file = fopen(filepath, "wb+");

    // Check that opening file was a success
    if(file == NULL){
        printf("Error in opening file\n");
        exit(0);
    }

    // Write bytes into the file.
    int idx = 0;
    while (fputc(contents[0], file) != EOF && idx < sizeof(contents)) {
        idx = idx + 1;
    }

    // Close file
    fclose(file);

    return contents;
}

/**
 * simple_encrypt: given an integer array and a key it produces a second array with the edited contents of the first. Sumation to encrypt.
 */
void simple_encrypt(int* plaintext_content, int* ciphertext_content, int key) {
    int idx = 0;
    while (idx < sizeof(plaintext_content)) {
        ciphertext_content[idx] = plaintext_content[idx] + key;
        idx = idx + 1;
    }
}

/**
 * simple_decrypt: given an integer array and a key it produces a second array with the edited contents of the first. Sumation to decrypt.
 */
void simple_decrypt(int* ciphertext_content, int* plaintext_content, int key) {
    int idx = 0;
    while (idx < sizeof(ciphertext_content)) {
        ciphertext_content[idx] = plaintext_content[idx] - key;
        idx = idx + 1;
    }
}

/**
 * testWithoutCUDA: given input and output files and a key, encrypt the input and save it on the output, then do the reverse.
 */
void testWithoutCUDA(char* inputfile, char* outputfile, int key) {
    // Prepare byte buffers
    int* plaintext = read_file_contents(inputfile);
    int* ciphertext = (int*) malloc(sizeof(plaintext));

    printf("Running test without CUDA\n");
    printf("Size of file: %d bytes\n", sizeof(plaintext));

    // Encrypt
    simple_encrypt(plaintext, ciphertext, key);

    // Write ciphertext to a file
    printf("Creating file: %s\n", strcat("no-cuda-",outputfile));
    write_file_contents(strcat("no-cuda-",outputfile), ciphertext);

    // Decrypt 
    simple_decrypt(ciphertext, plaintext, key);

    // Write plaintext to a file
    printf("Creating file: %s\n", strcat("no-cuda-",inputfile));
    write_file_contents(strcat("no-cuda-",inputfile), plaintext);
}

/**
 * simple_encryptCUDA: given an integer array and a key it produces a second array with the edited contents of the first. Sumation to encrypt.
 */
__global__ void simple_encryptCUDA(int* plaintext_content, int* ciphertext_content, int key) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
    ciphertext_content[idx] = plaintext_content[idx] + key;
}

/**
 * simple_decryptCUDA: given an integer array and a key it produces a second array with the edited contents of the first. Sumation to decrypt.
 */
__global__ void simple_decryptCUDA(int* ciphertext_content, int* plaintext_content, int key) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
    ciphertext_content[idx] = plaintext_content[idx] - key;
}

/**
 * testWithCUDA: given input and output files and a key, encrypt the input and save it on the output, then do the reverse.
 */
void testWithCUDA(char* inputfile, char* outputfile, int key) {
    // Prepare byte buffers and cuda variables
    int* plaintext = read_file_contents(inputfile);
    int* ciphertext = (int*) malloc(sizeof(plaintext));
    int totalThreads = sizeof(plaintext);
   	int blockSize = 256;
	int numBlocks = totalThreads/blockSize;

	// validate thread count
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		blockSize = totalThreads / numBlocks;
	}

    printf("Running test with CUDA\n");
    printf("Size of file: %d bytes\n", totalThreads);

    // Use pinned memory
	int* pinned_plaintext;
    int* pinned_ciphertext;
	cudaMallocHost((void**)&pinned_plaintext, totalThreads * sizeof(int));
	cudaMallocHost((void**)&pinned_ciphertext, totalThreads * sizeof(int));
    memcpy(pinned_plaintext, plaintext, totalThreads * sizeof(int));  


    // Prepare cuda variables
	int* dev_plaintext, *dev_ciphertext;

	// Copy inputs into device memory
	cudaHostGetDevicePointer(&dev_plaintext, plaintext, 0);
    cudaHostGetDevicePointer(&dev_ciphertext, ciphertext, 0);

    // Encrypt
    simple_encryptCUDA<<<numBlocks,totalThreads>>> (dev_plaintext, dev_ciphertext, key);
	cudaDeviceSynchronize();

    // Write ciphertext to a file
    printf("Creating file: %s\n", strcat("cuda-",outputfile));
    write_file_contents(strcat("cuda-",outputfile), pinned_ciphertext);
    
    // Decrypt
    simple_decryptCUDA<<<numBlocks,totalThreads>>> (dev_ciphertext, dev_plaintext, key);
	cudaDeviceSynchronize();

    // Write plaintext to a file
    printf("Creating file: %s\n", strcat("cuda-",outputfile));
    write_file_contents(strcat("cuda-",inputfile), pinned_plaintext);

    // Free reserved memory
	cudaFree(dev_plaintext);
	cudaFree(dev_ciphertext);
	cudaFreeHost(pinned_plaintext);
	cudaFreeHost(pinned_ciphertext);
}

int main(int argc, char** argv) {
	char* inputfile;
	char* outputfile;
    int key;

	// read command line arguments
	if (argc == 4) {
	    inputfile = argv[1];
	    outputfile = argv[2];
        key = atoi(argv[3]);
    } else {
        usage(argv[0]);
    }

	// Set up variables for timing
	clock_t start, end;
	double timePassedMiliSeconds;

    // Do test without cuda
    start = clock();
    testWithoutCUDA(inputfile, outputfile, key);
   	end = clock();
    timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Test Time: %f Miliseconds\n\n\n", timePassedMiliSeconds);

    // Do test with cuda
    start = clock();
    testWithCUDA(inputfile, outputfile, key);
   	end = clock();
    timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Test Time: %f Miliseconds\n\n\n", timePassedMiliSeconds);

    return 0;
}