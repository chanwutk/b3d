import json
import os

benchmark = []

for file in os.listdir('output'):
    if file.startswith('parallel-') and not file.endswith('write.log') and not file.startswith('parallel-8'):
        with open(os.path.join('output', file), 'r') as f:
            for line in f.readlines():
                if line.startswith("{"):
                    benchmark.append(json.loads(line))

with open('output/parallel_combined.json', 'w') as f:
    json.dump(benchmark, f, indent=2)
