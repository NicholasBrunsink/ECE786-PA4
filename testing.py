import os

# os.system(f"echo -e '\n\nStarting new run:\n' >> diff.txt")
for i in range(5):
    # os.system(f"echo -e '\nrunning {i}: \n' >> diff.txt")
    os.system(f'./softmaxV1 ./txt/softmax_input{i}.txt > ./out/out{i}_V1.txt')
    os.system(f'./softmaxV2 ./txt/softmax_input{i}.txt > ./out/out{i}_V2.txt')
    # os.system(f'diff ./out/out{i}_V1.txt ./txt/output{i}.txt >> diff.txt')
    # os.system(f'diff ./out/out{i}_V2.txt ./txt/output{i}.txt >> diff.txt')