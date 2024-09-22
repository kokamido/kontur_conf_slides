import random
import torch
from torch.profiler import profile, ProfilerActivity
from profiling_utils import get_model_and_data, go

asr_model, data = get_model_and_data(model_name="stt_ru_conformer_ctc_large", data_size=64, device='cuda')
random.seed(1234)
# warmup
go(asr_model, data)

random.shuffle(data)
with profile(with_stack=True, activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    go(asr_model, data)

with open('etc/profiler_outputs/gpu_forward_profiling', mode='w') as out:
    out.write(prof.key_averages().table(sort_by='self_cuda_time_total',max_name_column_width=4096, max_shapes_column_width=4096))

random.shuffle(data)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
go(asr_model, data)
end.record()
print(f'[LOG] Total time for GPU is {start.elapsed_time(end)} milliseconds')
