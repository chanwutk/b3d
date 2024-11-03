import json


for skip in [1, 2, 4, 8, 16, 32]:
    runtimes = {}
    with open(f'test.{skip}', 'r') as f:
        for line in f:
            res = json.loads(line)
            action = res['action']
            runtime = res['runtime']
            if action not in runtimes:
                runtimes[action] = []
            runtimes[action].append(runtime)
    

    for a, r in runtimes.items():
        print(f"{'{'} \"action\": \"{a}\", \"runtime\": {sum(r)}, \"skip\": {skip} {'}'},")