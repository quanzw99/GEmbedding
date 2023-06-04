
dataset = {
    'cite': '../dataset/tmall/tmall_edgelist.txt',
    'edge': '../dataset/tmall/tmall.txt',
}
citations = []
with open(dataset['cite'], 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) == 3:
            citations.append((parts[0], parts[1]))

with open(dataset['edge'], 'w') as f:
    for tup in citations:
        line = ' '.join([str(x) for x in tup])
        f.write(line + '\n')