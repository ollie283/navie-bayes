def extract_vocab():
    with open('sampleTrain.vocab.txt') as f:
        words = f.read().splitlines()
        print ({word: index for index, word in enumerate(words)})


extract_vocab()
