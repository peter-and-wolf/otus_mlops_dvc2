import time
import pickle
import json

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def train(train_path: str = 'data/scaled_train_data.npy', 
          model_path: str = 'data/model.pkl') -> None:

  t1 = time.time()
  print('\ntraining...')

  np.random.seed(42)

  train_data = np.load(train_path)

  model = OneVsRestClassifier(LogisticRegression(solver='newton-cg'), n_jobs=6)
  model.fit(train_data[:, 1:], train_data[:, 0])

  with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    print('model has been saved')

  total_time = time.time() - t1

  with open('metrics/train_metrics.json', 'w') as f:
    json.dump({'training_time': total_time}, f)

  print(f'done for {total_time:.2f}s')


if __name__ == '__main__':
  train()