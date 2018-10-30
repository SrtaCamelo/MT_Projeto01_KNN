import nltk
import pickle
from glob import glob

vocabulary = {}
i = 1
for filepath in glob('pos\\**'):
    arq = open(filepath,'r')
    text = arq.read()
    text = nltk.word_tokenize(text)
    text = [word.lower() for word in text if word.isalpha()]

    for token in text:
        if token in vocabulary:
            j = vocabulary[token]
            j += 1
            vocabulary[token] = j
        elif token not in vocabulary:
            vocabulary[token] = 1
    print('loading... %.2f%%' %(i*100/1000))
    i += 1

for filepath in glob('neg\\**'):
    arq = open(filepath, 'r')
    text = arq.read()
    text = nltk.word_tokenize(text)
    text = [word.lower() for word in text if word.isalpha()]

    for token in text:
        if token in vocabulary:
            j = vocabulary[token]
            j += 1
            vocabulary[token] = j
        elif token not in vocabulary:
            vocabulary[token] = 1
    print('loading... %.2f%%' % (i * 100 / 1000))
    i += 1

print(vocabulary)

print(len(vocabulary))

print('saving vocabulary')
with open('vocabulary', 'wb') as fp:
    pickle.dump(vocabulary, fp)
