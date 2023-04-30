# for cora
data = {
    'cite': './cora/cora.cites',
    'edge': '../dataset/cora/cora_edgelist.txt',
    'content': './cora/cora.content',
    'labels': '../dataset/cora/cora_labels.txt'
}

citations = []
with open(data['cite'], 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            citations.append((parts[0], parts[1]))

with open(data['edge'], 'w') as f:
    for tup in citations:
        line = ' '.join([str(x) for x in tup])
        f.write(line + '\n')

content = []
with open(data['content'], "r") as f:
    content = f.readlines()
# print(len(content))
label_data = []
labels = set()
for line in content:
    elements = line.strip().split("\t")
    label_data.append((elements[0], elements[-1]))
    labels.add(elements[-1])

# print(label_data)
label_map = {label: str(i) for i, label in enumerate(labels)}
print(label_map)
mapped_data = [(paper_id, label_map[class_label]) for paper_id, class_label in label_data]
# print(f'num of labels = {len(labels)}')

with open(data['labels'], "w") as f:
    for row in mapped_data:
        f.write(" ".join(row) + "\n")


# for dblp
# dblp = {
#     'cite': './dblp/dblp_edgelist.txt',
#     'edge': '../dataset/dblp/dblp_edgelist.txt',
# }
# citations = []
# with open(dblp['cite'], 'r') as f:
#     for line in f:
#         parts = line.strip().split(' ')
#         if len(parts) == 3:
#             citations.append((parts[0], parts[1]))
#
# with open(dblp['edge'], 'w') as f:
#     for tup in citations:
#         line = ' '.join([str(x) for x in tup])
#         f.write(line + '\n')