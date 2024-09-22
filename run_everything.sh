echo running 'python scripts/cpu_profile.py'
python scripts/cpu_profile.py

total_percents=$(cat etc/profiler_outputs/cpu_forward_profiling | awk '{print $2}' | grep '\.' | tr -d % | awk '{s+=$1} END {print s}')

echo "Sum of self_cpu% col in etc/profiler_outputs/cpu_forward_profiling is $total_percents" 