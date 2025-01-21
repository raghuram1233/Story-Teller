# TinyStories Language Model

This project trains a small language model that can learn to speak English with very few parameters using the TinyStories Dataset.


## Setup

1. Clone the repository:
    ```sh
    git clone [<repository-url>](https://github.com/raghuram1233/Story-Teller)
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the TinyStories dataset and place it in the [Hugging Face dataset provided by the authors of TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) directory.

## HyperParameters

batch_size : 64

block_size : 128 

vocab_size : 50257

max_iters : 5000

eval_interval : 50

learning_rate : 3e-4

device : 'cuda'

eval_iters : 50

n_embd : 768

n_head : 4

n_layer : 2

dropout : 0.2

## Training

To train the model, run:
```sh
python train.py
```
I used RTX3050(4GB VRAM) For 6hrs and It generated Pretty Good Samples


## Tokenization
To Tokenize the dataset, run:
```sh
from Prototypes.train import pre_tokenize_dataset

pre_tokenize_dataset('data/TinyStories-train.txt', 'data/tiny_tokenized.npy')
```

## Testing
To Test the Model, run test.py
```sh
python test.py
```

## Monitoring Weights & Biases
I used WandB for Monitoring. You Can use Terminal If You Want.
