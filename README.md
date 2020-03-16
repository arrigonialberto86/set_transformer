# set-transformer
A TensorFlow implementation of the paper 'Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks'

[![Build Status](https://travis-ci.com/arrigonialberto86/set_transformer.svg?branch=master)](https://travis-ci.com/arrigonialberto86/set_transformer)

<img src="imgs/transformer.png" alt="Image not found" width="400"/>

## Using the Docker

In this project a Dockerfile and a docker-compose.yml files have been added. You can use the listed services after cloning this project by doing:

```docker-compose build```

and then to start a Jupyter notebook with this package already installed in `develop` mode:

```docker-compose run -p 8001:8001 jupyter```

or start a `bash` session:

```docker-compose run bash```

and execute the automated unit test suite:

```pytest -W ignore```

## Basic example usage

```python
from set_transformer.data.simulation import gen_max_dataset
from set_transformer.model import BasicSetTransformer
import numpy as np

train_X, train_y = gen_max_dataset(dataset_size=100000, set_size=9, seed=1)
test_X, test_y = gen_max_dataset(dataset_size=15000, set_size=9, seed=3)

set_transformer = BasicSetTransformer()
set_transformer.compile(loss='mae', optimizer='adam')
set_transformer.fit(train_X, train_y, epochs=3)
predictions = set_transformer.predict(test_X)
print("MAE on test set is: ", np.abs(test_y - predictions).mean())
```

Which returns:

```bash
Train on 100000 samples
Epoch 1/3
100000/100000 [==============================] - 27s 270us/sample - loss: 32.8959
Epoch 2/3
100000/100000 [==============================] - 20s 197us/sample - loss: 6.6131
Epoch 3/3
100000/100000 [==============================] - 22s 216us/sample - loss: 6.6121
MAE on test set is:  6.558687
```