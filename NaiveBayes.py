def extract_vocab():
    with open('sampleTrain.vocab.txt') as f:
        words = f.read().splitlines()
        return {word: index for index, word in enumerate(words)}

def read_samples():
    with open("sampleTrain.txt") as f:
        samples = []
        string_samples = f.read().splitlines()
        for sample in string_samples:
            sample = sample.strip()
            sample_elements = sample.split('\t')
            document_name = sample_elements[0]
            real_class = int(sample_elements[1])
            words = sample_elements[2].split(' ')
            samples.append((document_name, real_class, words))
        print (samples)

read_samples()
