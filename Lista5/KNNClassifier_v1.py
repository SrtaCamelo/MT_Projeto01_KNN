import nltk
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

matrix = []
lemmatizer = WordNetLemmatizer()
#processa o vocabulário, gera os tokens, lemas e filtra o texto, retirando as stopwords
def toProcess(vocabulary,n):
    print('processing vocabulary')
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = word_tokenize(stopWords)

    filteredVocabylary = []
    for w in vocabulary:
        if w not in stopWords:
            filteredVocabylary.append([w,vocabulary[w]])

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
    for filepath in glob('pos\\**'):
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
        textLine.append(1)
        matrix.append(textLine)
        if i%200 == 0:
            print('loading matrix... %f%%' % (i * 100 / 1000))
        i += 1

    for filepath in glob('neg\\**'):
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
        textLine.append(0)
        matrix.append(textLine)
        if i%200 == 0:
            print('loading matrix... %f%%' % (i * 100 / 1000))
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
    for i in range (0,n-1):
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

    generateMatrix(vocabulary)

    # K-folder:
    kf = KFold(n_splits=10)
    for train, test in kf.split(matrix):
        # gera dois vetores que armazenam as posições do vetor MATRIX que serão utilizadas
        # para teste e treinamento
        # Primeiro realizar o treinamento e depois o teste
        # Calcular Métricas:  Precision, Recall, F-measure e Accuracy
        # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics (tem na biblioteca do sklearn)

        test = [1,80,120,340,525,534,680,715,750,777,830,915,999]
        result = []
        print('defining sentiment...')
        for i in test:
            distanceVector = []
            for j in range (0, matrix.__len__()-1):
                if j not in test:
                    distanceVector.append([j,getSimilarity(matrix[i],matrix[j])])
            distanceVector.sort(key=lambda x: x[1], reverse = True)

            while distanceVector.__len__() > k:
                distanceVector.pop()
            result.append([matrix[i][vocabulary.__len__()], defineSentiment(matrix,distanceVector)])

    return result


def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    k = input('digite o k a ser utilizado: ')
    n = input('digite o tamanho do vocabulario: ')

    vocabulary = toProcess(vocabulary,int(n))
    result = knn(vocabulary,int(k))
    print(result)
    acc = 0
    for i in result:
        if i[0] == i[1]:
            acc += 1
    print('accuracy: %f%%' % (acc/result.__len__()*100))


if __name__ == "__main__":
    main()