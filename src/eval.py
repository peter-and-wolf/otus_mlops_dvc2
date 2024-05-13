import pickle
import json

import numpy as np
from sklearn.metrics import accuracy_score


def eval(test_path: str = 'data/scaled_test_data.npy',
         model_path: str = 'data/model.pkl') -> None:

  print('\nevaluation...')

  test_data = np.load(test_path)

  with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print('model has been loaded')

  acc = accuracy_score(test_data[:, 0], model.predict(test_data[:, 1:]))

  print(f'accuracy on test: {acc}')

  with open('metrics/eval_metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)


if __name__ == '__main__':
  eval()