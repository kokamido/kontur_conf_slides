import torch.profiler as profiler
from profiling_utils import get_model_and_data
# import torch
# torch.backends.mkldnn.enabled = False   

asr_model, data = get_model_and_data(model_name="stt_ru_conformer_ctc_large", data_size=16, device='cpu')

#warmup
for processed_signal, processed_signal_length in data:
    asr_model.encoder(audio_signal=processed_signal, length=processed_signal_length)

with profiler.profile(with_stack=True, record_shapes=True, activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
    for processed_signal, processed_signal_length in data:
        asr_model.encoder(audio_signal=processed_signal, length=processed_signal_length)

with open('etc/profiler_outputs/cpu_forward_profiling', mode='w') as out:
    out.write(prof.key_averages().table(sort_by='self_cpu_time_total'))