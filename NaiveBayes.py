def extract_vocab():
    with open('sampleTrain.vocab.txt') as f:
        words = f.read().splitlines()
        return {word: index for index, word in enumerate(words)}

def read_samples():
    with open("sampleTrain.txt") as f:
        string_samples = f.read().splitlines()
        for sample in string_samples:
            sample = sample.strip()
            sample_elements = sample.split('\t')
            print (sample_elements)


read_samples()
