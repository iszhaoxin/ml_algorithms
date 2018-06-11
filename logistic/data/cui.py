from collections import Counter
from pathlib import Path
from random import shuffle

positive = Path(r'/home/dreamer/codes/algorithm_code/logistic/data/data/books/negative.review')
negative = Path(r'/home/dreamer/codes/algorithm_code/logistic/data/data/books/positive.review')


def sentence_stream(path: Path):
    with path.open('r', encoding='utf-8') as reader:
        for raw_line in reader:
            sentence = []
            for item in raw_line.strip().split():
                if len(item.split(':')) == 2:
                    token, freq = item.split(':')
                try:
                    sentence.append((token, int(freq)))
                except ValueError:
                    exit(1)
            yield sentence


def update_counter(path: Path, counter: Counter):
    for sentence in sentence_stream(path):
        for token, freq in sentence:
            counter[token] += freq


def build_vocabulary():
    counter = Counter()
    update_counter(positive, counter)
    update_counter(negative, counter)

    vocabulary = {}
    for token, ix in counter.most_common():
        vocabulary[token] = ix

    return vocabulary


def prepare():
    vocabulary = build_vocabulary()
    pos_data, pos_targets = zip(*handle(positive, 1.0))
    neg_data, neg_targets = zip(*handle(positive, 0.0))
    dataset = list(zip(pos_data + neg_data, pos_targets + neg_targets))
    shuffle(dataset)
    data, targets = zip(*dataset)
    return data, targets

def handle(path: Path, target: float):
    for sentence in sentence_stream(path):
        datum = [0] * vocabulary.__len__()
        for token, freq in sentence:
            datum[vocabulary[token]] = 1
        yield datum, target

def prepare_test():
    vocabulary = build_vocabulary()
    test_data, test_targets = zip(*handle(positive, 1.0))
    dataset = list(zip(test_data, test_targets))
    shuffle(dataset)
    data, targets = zip(*dataset)
    return data, targets


def data_iteration(data, targets, batch_size: int):
    for index in range((data.__len__() + batch_size - 1) // batch_size):
        yield data[batch_size * index:batch_size * (index + 1)], \
              targets[batch_size * index:batch_size * (index + 1)]


if __name__ == '__main__':
    data, targets = prepare()
    print('data', data.__len__())
    print('targets', targets.__len__())
