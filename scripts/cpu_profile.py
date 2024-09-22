import random
import time 
import torch.profiler as profiler
from profiling_utils import get_model_and_data, go

asr_model, data = get_model_and_data(model_name="stt_ru_conformer_ctc_large", data_size=64, device='cpu')
random.seed(1234)

#warmup
go(asr_model, data)

random.shuffle(data)
with profiler.profile(with_stack=True, activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
    go(asr_model, data)

random.shuffle(data)
with open('etc/profiler_outputs/cpu_forward_profiling', mode='w') as out:
    out.write(prof.key_averages().table(sort_by='self_cpu_time_total'))

random.shuffle(data)
start = time.perf_counter_ns()
go(asr_model, data)
end = time.perf_counter_ns()
print(f'[LOG] Total time for CPU is {(end - start)/1_000_000} milliseconds')