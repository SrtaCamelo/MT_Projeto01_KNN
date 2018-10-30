from sklearn.model_selection import KFold
import nltk
from glob import glob
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

X_matriz = []

X_data = []
X_target = []

lemmatizer = WordNetLemmatizer()


#processa o vocabulário, gera os tokens, lemas e filtra o texto, retirando as stopwords
def toProcess(vocabulary, n):
    print('[INFO] processing vocabulary')
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = word_tokenize(stopWords)
    wordsFiltered = {}

    filteredVocabylary = []
    for w in vocabulary:
        if w not in stopWords:
            filteredVocabylary.append([w, vocabulary[w]])

    filteredVocabylary.sort(key=lambda x: x[1], reverse=True)
    while filteredVocabylary.__len__() > n:
        filteredVocabylary.pop()

    return filteredVocabylary

#gera a matriz de classificação, onde cada texto vira um array do tipo
#texto1 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
#onde cada indice 'i' do array texto1 indica se o text possui a palavra 'i' do vocabulário
#e o ultimo índice indica se o texto é positivo (1) ou negativo (0)
#a matriz gerada não é nada mais que uma lista de listas com todos os arrays
#matriz = [texto1, texto2, texto3, ...]
def generateMatrix(vocabulary):
    i = 1
    for filepath in glob('Base2\\pos\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in vocabulary:
            if word[0] in text:
                textLine.append(1)
            else:
                textLine.append(0)
        X_data.append(textLine)
        X_target.append(1)

        if i%200 == 0:
            print('[INFO] loading matrix... %.1f%%' % (i * 100 / 1000))
        i += 1

    for filepath in glob('Base2\\neg\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in vocabulary:
            if word[0] in text:
                textLine.append(1)
            else:
                textLine.append(0)

        X_data.append(textLine)
        X_target.append(0)

        if i%200 == 0:
            print('[INFO] loading matrix... %.1f%%' % (i * 100 / 1000))
        i += 1

def printMatrix(matrix):
    for i in matrix:
        for j in i:
            print(j, end='')
        print()

#calcula a similaridade baseada na quantidade de palavras que as duas linhas (textos) tem em comum
def getSimilarity(lineA, lineB):
    n = lineA.__len__()
    similarity = 0
    for i in range(0, n-1):
        if lineA[i] == 1 and lineB[i] == 1:
            similarity += 1
    return similarity


def defineSentiment(matrix,distanceVector):
    positive = 0
    negative = 0
    while positive == negative:
        for i in range(0, distanceVector.__len__()):
            j = distanceVector[i][0]
            n = matrix[0].__len__()
            m = matrix.__len__()
            if matrix[j][n - 1] == 1:
                positive += 1
            else:
                negative += 1
        distanceVector.pop()

    if positive > negative:
        return 1
    else:
        return 0

def knn(vocabulary, k):
    test = [1, 80, 120, 340, 525, 534, 680, 715, 750, 777, 830, 915, 999]
    result = []
    print('[INFO] defining sentiment...')
    for i in test:
        distanceVector = []
        for j in range(0, X_matriz.__len__() - 1):
            if j not in test:
                distanceVector.append([j, getSimilarity(X_matriz[i], X_matriz[j])])
        distanceVector.sort(key=lambda x: x[1], reverse = True)

        while distanceVector.__len__() > k:
            distanceVector.pop()
        result.append([X_matriz[i][vocabulary.__len__()], defineSentiment(X_matriz, distanceVector)])

    return result


def main():
    print('[INFO]loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        vocabulary = pickle.load(fp)
    print('[INFO]vocabulary has been loaded.')

    k = input('\n\t [INPUT] digite o k a ser utilizado: ')
    n = input('\t [INPUT] digite o tamanho do vocabulario: ')
    print('\n')

    vocabulary = toProcess(vocabulary, int(n))

    generateMatrix(vocabulary)

    print("len Xdata ", len(X_data))
    kf = KFold(n_splits=2)
    for train, test in kf.split(X_data):
        print("len train %d len test %d" % (len(train), len(test)))
        print("Train \n%s \n Test \n %s" % (train, test))


    '''result = knn(vocabulary, int(k))
    print(result)
    acc = 0
    for i in result:
        if i[0] == i[1]:
            acc += 1
    print('[INFO] accuracy: %.2f%%' % (acc/result.__len__()*100))'''


if __name__ == "__main__":
    main()