import csv
import sys
import numpy as np

result_name = sys.argv[1]
method = sys.argv[2]

with open('./train_log/' + result_name + '.log', 'r', encoding='utf-8') as file:
    dataset = file.read()
with open('./prompts/' + result_name + '.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['reward', 'prompt'])

start, end = 0, 0
prompts, rewards = [], []

if method == 'fuzz':
    while True:
        if (start == -1 + len('New seed generated: ')): break
        start = dataset.find('New seed generated: ', end) + len('New seed generated: ')
        assert(start > end or start == -1 + len('New seed generated: '))
        end = dataset.find('Reward:  ', start) + len('Reward:  ')
        assert(end > start)
        prompts.append(dataset[start: end - len('Reward:  ') - 1].replace('\n', '\\n'))
        rewards.append(float(dataset[end: end + 8]))
elif method == 're':
    while True:
        if (start == -1 + len(': ')): break
        start = dataset.find('New seed generated with action ', end)
        start = dataset.find(': ', start) + len(': ')
        assert(start > end or start == -1 + len(': '))
        end1 = dataset.find('Step ', start) + len('Step ')
        end2 = dataset.find('training succeeded', start) + len('training succeeded')
        if end1 == -1 + len('Step '): end1 = np.inf
        if end2 == -1 + len('training succeeded'): end2 = np.inf
        end = min(end1, end2)
        assert(end > start)
        if end1 < end2:
            prompts.append(dataset[start: end - len('Step ') - 1].replace('\n', '\\n'))
            idx = dataset.find('reward: ', end) + len('reward: ')
            rewards.append(float(dataset[idx: idx + 8]))
        else:
            prompt = dataset[start: end - len('The 1th training succeeded') - 1].replace('\n', '\\n')
            if prompt[-1] == 'T': prompt = prompt[0: len(prompt) - 1]
            prompts.append(prompt)
            idx = dataset.find('reward: ', end) + len('reward: ')
            rewards.append(float(dataset[idx: idx + 8]))

pair = zip(rewards, prompts)
pair = sorted(pair, reverse=True)
for i in range(10):
    with open('./prompts/' + result_name + '.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([pair[i][0], pair[i][1]])
