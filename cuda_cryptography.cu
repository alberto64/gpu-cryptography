#include <stdio.h>
#include <time.h>
#include <string.h>


void usage(const char *command) {
    printf("Usage: %s <inputFile> <outputFile> <key>\n", command);
    exit(0);
}

int* read_file_contents(char* filepath) {

    FILE* file = fopen(filepath, "rb");

    if(file == NULL){
        printf("Error in opening file\n");
        exit(0);
    }

    fseek(file, 0L, SEEK_END);
    int* contents = (int*) malloc(sizeof(int) * ftell(file));
    rewind(file);
    int idx = 0;

    while ((contents[idx] = fgetc(file)) != EOF) {
        idx = idx + 1;
    }

    fclose(file);

    return contents;
}

int* write_file_contents(char* filepath, int* contents) {

    FILE* file = fopen(filepath, "wb+");

    if(file == NULL){
        printf("Error in opening file\n");
        exit(0);
    }

    int idx = 0;

    while (fputc(contents[0], file) != EOF && idx < sizeof(contents)) {
        idx = idx + 1;
    }

    fclose(file);

    return contents;
}

void simple_encrypt(int* plaintext_content, int* ciphertext_content, int key) {
    int idx = 0;
    while (idx < sizeof(plaintext_content)) {
        ciphertext_content[idx] = plaintext_content[idx] + key;
        idx = idx + 1;
    }
}

void simple_decrypt(int* ciphertext_content, int* plaintext_content, int key) {
    int idx = 0;
    while (idx < sizeof(ciphertext_content)) {
        ciphertext_content[idx] = plaintext_content[idx] - key;
        idx = idx + 1;
    }
}

void testWithoutCUDA(char* inputfile, char* outputfile, int key) {
    int* plaintext = read_file_contents(inputfile);
    int* ciphertext = (int*) malloc(sizeof(plaintext));

    printf("Running test without CUDA\n");
    printf("Size of file: %d bytes\n", sizeof(plaintext));

    simple_encrypt(plaintext, ciphertext, key);
    write_file_contents(strcat("no-cuda-",outputfile), ciphertext);
    simple_decrypt(ciphertext, plaintext, key);
    write_file_contents(strcat("no-cuda-",inputfile), plaintext);
}

__global__ void simple_encryptCUDA(int* plaintext_content, int* ciphertext_content, int key) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
    ciphertext_content[idx] = plaintext_content[idx] + key;
}

__global__ void simple_decryptCUDA(int* ciphertext_content, int* plaintext_content, int key) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
    ciphertext_content[idx] = plaintext_content[idx] - key;
}

void testWithCUDA(char* inputfile, char* outputfile, int key) {
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

    write_file_contents(strcat("cuda-",outputfile), pinned_ciphertext);
    
    // Decrypt
    simple_decryptCUDA<<<numBlocks,totalThreads>>> (dev_ciphertext, dev_plaintext, key);
	cudaDeviceSynchronize();

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

    start = clock();
    testWithoutCUDA(inputfile, outputfile, key);
   	end = clock();
    timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Test Time: %f Miliseconds\n", timePassedMiliSeconds);


    start = clock();
    testWithCUDA(inputfile, outputfile, key);
   	end = clock();
    timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Test Time: %f Miliseconds\n", timePassedMiliSeconds);

    return 0;
}