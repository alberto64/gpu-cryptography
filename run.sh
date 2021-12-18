nvcc cuda_cryptography.cu -o cuda_cryptography.o 

./cuda_cryptography.o "smalltest.txt" "smalltest_cipher.txt" 8

./cuda_cryptography.o "test.txt" "test_cipher.txt" 8

./cuda_cryptography.o "bigtest.txt" "bigtest_cipher.txt" 8

./cuda_cryptography.o "verybigtest.txt" "verybigtest_cipher.txt" 8
