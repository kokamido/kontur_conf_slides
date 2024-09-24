echo CPU
export CUDA_VISIBLE_DEVICES=,
for i in {32,64,128,256,512} 
do
    for j in {1..3}
    do
        llama.cpp/llama-cli -m /home/alexandr/Downloads/Phi-3-mini-4k-instruct-fp16.gguf -n $i -t 8 -ngl 200000 -p "Please sir, may I have some " 2>&1| grep  "total time"
    done
done

echo GPU
export CUDA_VISIBLE_DEVICES=0
for i in {32,64,128,256,512} 
do
    for j in {1..3}
    do
        llama.cpp/llama-cli -m /home/alexandr/Downloads/Phi-3-mini-4k-instruct-fp16.gguf -n $i -t 8 -ngl 200000 -p "Please sir, may I have some " 2>&1| grep "total time"
    done
done