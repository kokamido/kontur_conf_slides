import re
import numpy as np
import pylab as plt
from draw_utils import COLOR_PINK, COLOR_VIOLET, fix_axes_style

extract_data = re.compile(r' (?P<time>[0-9.]+) ms /[ ]+(?P<tokens>[0-9]+) tokens')

cpu_points = []
gpu_points = []

current = None
with open('etc/profiler_outputs/llama_profiling') as inp:
    for line in inp:
        line = line.strip()
        if line == 'CPU':
            current = cpu_points
        elif line == 'GPU':
            current = gpu_points
        else:
            data = extract_data.search(line)
            current.append((float(data.group('time')), int(data.group('tokens'))) )
cpu_points = np.array(cpu_points)
gpu_points = np.array(gpu_points)
ax = plt.gca()
ax.scatter(cpu_points[:, 1], cpu_points[:, 0], c='#0082ff', s=45)
ax.plot(cpu_points[:, 1], cpu_points[:, 0], c='#0082ff', lw=3, label='CPU')
ax.scatter(gpu_points[:, 1], gpu_points[:, 0], c='#00ffc3', s=45)
ax.plot(gpu_points[:, 1], gpu_points[:, 0], c='#00ffc3', lw=3, label='GPU')
ax.set_ylabel('миллисекунды')
ax.set_xlabel('количество токенов')
ax.set_title('llama.cpp')
fix_axes_style(ax)
plt.legend(framealpha=0.0)
plt.savefig('pics/llama.png', bbox_inches='tight')