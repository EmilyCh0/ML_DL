import numpy as np
from torch.utils import data
import torch.nn as nn

np.random.seed(0)
trainingset_size = 270
testset_size = 30
dataset_size = trainingset_size + testset_size


def generate_dataset(num_sequences):
    samples = []

    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 12)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)

    return samples


sequences = generate_dataset(dataset_size)


class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]

        return x, y


def create_datasets(sequences, dataset_class, num_train, num_test):
    sequences_train = sequences[:num_train]
    sequences_test = sequences[num_train:num_train + num_test]

    def get_inputs_targets_from_sequences(sequences):
        inputs, targets = [], []

        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])

        return inputs, targets

    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    training_set = dataset_class(inputs_train, targets_train)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, test_set


training_set, test_set = create_datasets(sequences, Dataset, trainingset_size, testset_size)


def one_hot_encode(token):
    one_hot = np.zeros(3)
    # your job
    if token=='a':
        one_hot[0]=1
    elif token=='b':
        one_hot[1]=1
    else:
        one_hot[2]=1

    return one_hot


def one_hot_encode_sequence(sequence):
    encoding = np.array([one_hot_encode(token) for token in sequence])

    return encoding.T


def init_orthogonal(w):
    rows, cols = w.shape
    new_w = np.random.randn(rows, cols)

    if rows < cols:
        new_w = new_w.T

    q, r = np.linalg.qr(new_w)

    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    new_w = q

    return new_w


def init_rnn(hidden_size, vocab_size):
    Whh = np.zeros((hidden_size, hidden_size))
    Wxh = np.zeros((hidden_size, vocab_size))
    Why = np.zeros((vocab_size, hidden_size))
    bh = np.zeros((hidden_size, 1))
    by = np.zeros((vocab_size, 1))

    Whh = init_orthogonal(Whh) * 1e-3
    Wxh = init_orthogonal(Wxh) * 1e-3
    Why = init_orthogonal(Why) * 1e-3

    return Whh, Wxh, Why, bh, by


def tanh(x, derivative=False):
    # your job
    y = np.tanh(x)

    if derivative:
        return 1 - y ** 2
    else:
        return y


def softmax(x):
    # your job
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def forward(inputs, hidden_state, params):
    Whh, Wxh, Why, bh, by = params
    outputs, hidden_states = [], []

    # your job
    for x in inputs.T:

        Ht = np.dot(Wxh, x).reshape(50,1) + np.dot(Whh, hidden_state) +bh
        Ht = tanh(Ht)
        Yt = np.dot(Why, Ht) + by
        Y = softmax(Yt)
        outputs.append(Y)
        hidden_states.append(Ht)
    return outputs, hidden_states


def clip_gradient_norm(grads, max_norm=0.25):
    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads


def backprop(inputs, outputs, hidden_states, targets, params):
    Whh, Wxh, Why, bh, by = params

    dWhh, dWxh, dWhy = np.zeros_like(Whh), np.zeros_like(Wxh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    dhnext = np.zeros_like(hidden_states[0])
    loss = 0

    for t in reversed(range(len(outputs))):
        loss += np.sum([-np.log(outputs[t][i] + 1e-12) * targets[i, t] for i in range(vocab_size)])

        dO = outputs[t].copy()
        dO[np.argmax(targets[:, t].reshape(vocab_size, 1))] -= 1

        dh = np.dot(Why.T, dO) + dhnext
        df = tanh(hidden_states[t].reshape(hidden_size, 1), derivative=True) * dh

        dWhh += np.dot(df, hidden_states[t - 1].T)  # W
        dWxh += np.dot(df, inputs[:, t].reshape(vocab_size, 1).T)  # U
        dWhy += np.dot(dO, hidden_states[t].T)  # V

        dbh += df
        dby += dO

        dhnext = np.dot(Whh.T, df)

    grads = dWhh, dWxh, dWhy, dbh, dby
    grads = clip_gradient_norm(grads)

    return loss, grads


def update_parameters(params, grads, lr=1e-4):
    # your job
    Whh, Wxh, Why, bh, by = params
    dWhh, dWxh, dWhy, dbh, dby = grads
    Whh -= lr*dWhh
    Wxh -= lr*dWxh
    Why -= lr * dWhy
    bh -= lr * dbh
    by -= lr * dby
    params = Whh, Wxh, Why, bh, by
    return params


hidden_size = 50
vocab_size = 3

num_epochs = 1000
params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)

hidden_state = np.zeros((hidden_size, 1))

training_loss, test_loss = [], []

for i in range(num_epochs):
    epoch_training_loss = 0
    epoch_test_loss = 0

    for inputs, targets in test_set:
        inputs_one_hot = one_hot_encode_sequence(inputs)
        targets_one_hot = one_hot_encode_sequence(targets)

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = forward(inputs_one_hot, hidden_state, params)
        loss, _ = backprop(inputs_one_hot, outputs, hidden_states, targets_one_hot, params)

        epoch_test_loss += loss

    for inputs, targets in training_set:
        inputs_one_hot = one_hot_encode_sequence(inputs)
        targets_one_hot = one_hot_encode_sequence(targets)

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = forward(inputs_one_hot, hidden_state, params)
        loss, grads = backprop(inputs_one_hot, outputs, hidden_states, targets_one_hot, params)

        params = update_parameters(params, grads, lr=1e-3)
        epoch_training_loss += loss

    training_loss.append(epoch_training_loss / len(training_set))
    test_loss.append(epoch_test_loss / len(test_loss))

    if (i % 100 == 0):
        print(f'Epoch {i}, training loss: {training_loss[-1]}, test loss: {test_loss[-1]}')