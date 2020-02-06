"""transfer learning experiments train"""
#!/usr/bin/env python

from src.transfer_learning.train import train

CONFIG = {
    'data.dataset_name': ['ciarp', 'lsa16', 'rwth'],
    'data.test_size': 0.25,
    'data.train_size': [0.33, 0.5, 0.64, 0.75],
    'data.rotation_range': 10,
    'data.width_shift_range': 0.1,
    'data.height_shift_range': 0.2,
    'data.horizontal_flip': True,
    'model.type': ['DenseNet121','DenseNet169','DenseNet201']
}

rotation_range = CONFIG['data.rotation_range']
width_shift_range = CONFIG['data.width_shift_range']
height_shift_range = CONFIG['data.height_shift_range']
horizontal_flip = CONFIG['data.horizontal_flip']
test_size = CONFIG['data.test_size']
for i in range(3):
    dataset_name = CONFIG['data.dataset_name'][i]
    for j in range(3):
        model_type = CONFIG['model.type'][j]
        for train_size in CONFIG['data.train_size']:
            train(dataset_name=dataset_name, rotation_range=rotation_range,
                  width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                  horizontal_flip=horizontal_flip, test_size=test_size, model_type=model_type
                  train_size=train_size, batch_size=16)
            print("Finished transfer learning densenet with")
            print("dataset_name: {}".format(dataset_name))
            print("rotation_range: {}".format(rotation_range))
            print("width_shift_range: {}".format(width_shift_range))
            print("height_shift_range: {}".format(height_shift_range))
            print("horizontal_flip: {}".format(horizontal_flip))
            print("train size: {}".format(train_size))
            print("model type: {}".format(model_type))
