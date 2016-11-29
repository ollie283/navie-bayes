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

def train_model(word_2_num):
    prior_probabilities = {}
    word_likelihoods_per_class = {}
    samples = read_samples('sampleTrain.txt')
    
    for sample in samples:
        prior_probabilities[sample[1]] = prior_probabilities.get(sample[1], 0.0) + 1.0
        word_likelihoods = word_likelihoods_per_class.setdefault(sample[1], {})
        for word in sample[2]:
            word_likelihoods[word_2_num[word]] = word_likelihoods.get(word_2_num[word], 0.0) + 1.0
    number_of_documents = sum(prior_probabilities.values())    

    for class_index in prior_probabilities:
        prior_probabilities[class_index] /= number_of_documents

    for class_index in word_likelihoods_per_class:
        word_likelihoods = word_likelihoods_per_class[class_index]
        per_class_words = sum(word_likelihoods.values())

        for word_num in word_2_num.values():
            word_likelihoods[word_num] = (1.0 + word_likelihoods.get(word_num, 0.0)) / (
            per_class_words + float(len(word_2_num)))
    return prior_probabilities, word_likelihoods_per_class


if __name__ == '__main__':
    word_2_num = extract_vocab()
    prior_probabilities, word_likelihoods_per_class = train_model(word_2_num)
    print('Prior probabilities')
    for class_index in prior_probabilities:
        print('class {} = {}'.format(class_index, prior_probabilities[class_index]))
    print()
