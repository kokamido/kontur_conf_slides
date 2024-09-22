import time
import random
import torch
import torch.cuda.nvtx as nvtx
from profiling_utils import get_model_and_data, go

asr_model, data = get_model_and_data(model_name="stt_ru_conformer_ctc_large", data_size=64, device='cuda')
random.seed(1234)

#warmup
random.shuffle(data)
nvtx.range_push('Warmup')
go(asr_model, data)
nvtx.range_pop()

start = time.perf_counter_ns()
torch.cuda.synchronize()
end = time.perf_counter_ns()
print(f'Time of torch.cuda.synchronize() 1 is {end - start} nanoseconds')

random.shuffle(data)
nvtx.range_push('Forward Pass 1')
go(asr_model, data)
nvtx.range_pop()

start = time.perf_counter_ns()
torch.cuda.synchronize()
end = time.perf_counter_ns()
print(f'Time of torch.cuda.synchronize() 2 is {end - start} nanoseconds')

random.shuffle(data)
nvtx.range_push('Forward Pass 2')
go(asr_model, data)
nvtx.range_pop()

start = time.perf_counter_ns()
torch.cuda.synchronize()
end = time.perf_counter_ns()
print(f'Time of torch.cuda.synchronize() 3 is {end - start} nanoseconds')
