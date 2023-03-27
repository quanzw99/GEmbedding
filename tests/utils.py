import json

def get_dateset(data_name='wiki'):
    with open('./dataset.json', 'r') as dataset_file:
        dataset_data = json.load(dataset_file)
        return dataset_data[data_name]