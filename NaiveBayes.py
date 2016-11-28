def extract_vocab():
    with open('sampleTrain.vocab.txt') as f:
        words = f.read().splitlines()
        return {word: index for index, word in enumerate(words)}

def read_samples(path):
    with open(path) as f:
        samples = []
        string_samples = f.read().splitlines()
        for sample in string_samples:
            sample = sample.strip()
            sample_elements = sample.split('\t')
            document_id = sample_elements[0]
            real_class = int(sample_elements[1])
            words = sample_elements[2].split(' ')
            samples.append((document_id, real_class, words))
        return samples

def train_model():
    prior_probabilities = {}
    samples = read_samples('sampleTrain.txt')
    for sample in samples:
        prior_probabilities[sample[1]] = prior_probabilities.get(sample[1], 0.0) + 1.0
        
    print (prior_probabilities)

train_model()
