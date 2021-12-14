/**
 * @file	cuda_cryptography.c
 * @author	Alberto De Jesus
 * @brief Using cuda to handle large files and encrypt using cuda AES.
 */

#include "source/rijndael.c"
#include <stdio.h>
#include <time.h>

void print(char* content, int size) {
    for (int idx = 0; idx < size; idx++) {
	    printf("%s", content[idx]);
    }
    printf("\n");
}

void encrypt(RijnKeyParam key, char* content, int size) {
    for(int idx = 0; idx < size; idx++) {
        rijn_encrypt(key, content[idx]);
    }
}

void decrypt(RijnKeyParam key, char* content, int size) {
    for(int idx = 0; idx < size; idx++) {
        rijn_decrypt(key, content[idx]);
    }
}

RijnKeyParam key() {
    RijnKeyParam key;
    key.num_key = 4;
    key.num_round = 10;
    key.enc_key = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };
    key.dec_key = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    return key;
}

char* load_file_contents(char* filepath) {
    FILE *fp;
    fp = fopen(filepath, "r");

    fseek(fp, 0L, SEEK_END);

    char* contents = (char*) malloc(sizeof(char) * ftell(fp));

    rewind(fp);

    int count = 0;

    while(fscanf(file, "%d", &contents[count]) == 1) { 
        count++;
    } 

    fclose(fp);

    return contents;
}

void usage(const char *command) {
    printf("Usage: %s <filepath> <blockCount>\n", command);
    exit(0);
}

int main(int argc, char** argv) {
	
	// Set default values in case arguments don't come in command line.
	int numBlocks = 4;
   	int blockSize = 256;

	char* file = "";

	// read command line arguments
	if (argc >= 1) {
        usage(argv[0]);
    } else if (argc <= 2) {
        file = atoi(argv[1]);
	} else if (argc <= 3) {
        file = atoi(argv[1]);
		numBlocks = atoi(argv[2]);
	} else {
        usage(argv[0]);
    }

    int totalThreads = numBlocks * blockSize;

    char* file_content = load_file_contents(file);

    RijnKeyParam* key = key();

    print(file_content);

    encrypt(key, file_content, sizeof(file_content));

    print(file_content);
   
    decrypt(key, file_content, sizeof(file_content));

    print(file_content);

    return 0;
}