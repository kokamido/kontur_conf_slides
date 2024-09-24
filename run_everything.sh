rm -rf etc/profiler_outputs/*

echo running 'python scripts/cpu_profile.py'
python scripts/cpu_profile.py 2>/dev/null

total_percents_self_cpu=$(cat etc/profiler_outputs/cpu_forward_profiling | awk '{print $2}' | grep '\.' | tr -d % | awk '{s+=$1} END {print s}')

echo "[LOG] Sum of self_cpu% col in etc/profiler_outputs/cpu_forward_profiling is $total_percents_self_cpu" 

echo running 'python scripts/gpu_profile.py'
python scripts/gpu_profile.py 2>/dev/null

total_percents_self_cuda=$(cat etc/profiler_outputs/gpu_forward_profiling | sed -E 's/[ ]{2,}/\t/g' |cut -f9 | grep -oP "\d+\.\d+%" | tr -d %| awk '{s+=$1} END {print s}')
echo "[LOG] Sum of self_cuda% col in etc/profiler_outputs/gpu_forward_profiling is $total_percents_self_cuda" 

total_percents_self_cuda_no_aten=$(cat etc/profiler_outputs/gpu_forward_profiling | grep -v 'aten::' | sed -E 's/[ ]{2,}/\t/g' |cut -f9 | grep -oP "\d+\.\d+%" | tr -d %| awk '{s+=$1} END {print s}')
echo "[LOG] Sum of self_cuda% without aten:: col in etc/profiler_outputs/gpu_forward_profiling is $total_percents_self_cuda_no_aten" 


echo running nsys profile
nsys profile \
 --trace=cuda,cudnn,cublas,nvtx \
 --output=etc/profiler_outputs/nsys_profiler_out \
 python scripts/gpu_profile_nsys.py 2>/dev/null

echo running llama.cpp "benchmark"

bash scripts/bench_llama.sh | tee etc/profiler_outputs/llama_profiling && python scripts/draw_llama.py