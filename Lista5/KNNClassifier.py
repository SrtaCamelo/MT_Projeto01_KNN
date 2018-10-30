import nltk
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
from math import log2
#import xlwt, xlrd
#from xlutils.copy import copy

lemmatizer = WordNetLemmatizer()
dictionary = []
vocabulary = {}

#processa o vocabulário, gera os tokens, lemas e filtra o texto, retirando as stopwords
def toProcess(vocabulary,n):
    print('processing vocabulary')
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = word_tokenize(stopWords)

    tfVector = []
    for w in vocabulary:
        if w not in stopWords:
            tfVector.append([w,vocabulary[w]])

    total_N = len(tfVector)

    idfVetor = []
    for i in tfVector:
        idfVetor.append([i[0],log2(total_N/i[1])])

    tf_idf_vector = []
    for i in range(len(idfVetor)):
        tupla_idf = idfVetor[i]
        tupla_tf = tfVector[i]
        tf_idf_vector.append([tupla_idf[0], tupla_idf[1] * tupla_tf[1]])


    tf_idf_vector.sort(key=lambda x: x[1], reverse=True)

    arq.close()
    return tf_idf_vector

#gera a matriz de classificação, onde cada texto vira um array do tipo
#texto1 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
#onde cada indice 'i' do array texto1 indica se o text possui a palavra 'i' do vocabulário
#e o ultimo índice indica se o texto é positivo (1) ou negativo (0)
#a matriz gerada não é nada mais que uma lista de listas com todos os arrays
#matriz = [texto1, texto2, texto3, ...]
def generateMatrix(vocabulary):
    matrix = []
    i = 1

    for filepath in glob('competition_final_test\\pos\\**'):
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
            print('loading matrix... %.2f%%' % (i * 100 / 1000))
        i += 1

    for filepath in glob('competition_final_test\\neg\\**'):
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
            print('loading matrix... %.2f%%' % (i * 100 / 1000))
        i += 1

    return matrix

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
        df = vocabulary[dictionary[i][0]]
        if lineA[i] == 1 and lineB[i] == 1:
            similarity = similarity + 1 + (1/df)
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

def knn(dictionary, k):

    matrix = generateMatrix(dictionary)

    y_esperado = []
    y_obtido = []

    # K-folder:
    kf = KFold(n_splits=10)
    print('defining sentiment...')
    for train, test in kf.split(matrix):
        # gera dois vetores que armazenam as posições do vetor MATRIX que serão utilizadas
        # para teste e treinamento
        # Primeiro realizar o treinamento e depois o teste
        # Calcular Métricas:  Precision, Recall, F-measure e Accuracy
        # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics (tem na biblioteca do sklearn)

        for i in test:
            distanceVector = []
            for j in train:
                distanceVector.append([j, getSimilarity(matrix[i],matrix[j])])

            distanceVector.sort(key=lambda x: x[1], reverse = True)

            while distanceVector.__len__() > k:
                distanceVector.pop()

            y_esperado.append(matrix[i][dictionary.__len__()])
            y_obtido.append(defineSentiment(matrix, distanceVector))

    return y_esperado, y_obtido

def criar_arquivo():
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet("resultados", True)

    worksheet.write(0, 0, 'Test')
    worksheet.write(0, 1, 'Precision')
    worksheet.write(0, 2, 'Recall')
    worksheet.write(0, 3, 'F-Measure')
    worksheet.write(0, 4, 'Accuracy')

    workbook.save("resultados.xls")


def adicionar_arquivo(saida):
    workbook = xlrd.open_workbook("resultados.xls")
    worksheet = workbook.sheet_by_index(0)

    wb = copy(workbook)

    linha = worksheet.nrows
    sheet = wb.get_sheet(0)

    for col in range(5):
        sheet.write(linha, col, saida[col])

    try:
        wb.save("resultados.xls")
    except IOError:
        wb.save("resultados.xls")


def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        global vocabulary
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    fp.close()

    global dictionary

    # criar_arquivo()

    '''for n in range(1300, 3000, 400):
        for k in range(2, 11):
            print("N %d K %d" %(n, k))

            dictionary = toProcess(vocabulary, n)
            y_esperado, y_obtido = knn(dictionary, k)

            saida = []

            saida.append("N - " + str(n) + " K - " + str(k))

            precision = recall_score(y_esperado, y_obtido)
            saida.append(precision)
            recall = precision_score(y_esperado, y_obtido)
            saida.append(recall)
            f_measure = f1_score(y_esperado, y_obtido)
            saida.append(f_measure)
            accuracy = accuracy_score(y_esperado, y_obtido)
            saida.append(accuracy)

            print(saida)

            adicionar_arquivo(saida)'''

    n = 100000
    k = 5

    print("N %d K %d" % (n, k))

    dictionary = toProcess(vocabulary, n)
    print(len(dictionary))
    y_esperado, y_obtido = knn(dictionary, k)

    saida = []

    saida.append("N - " + str(n) + " K - " + str(k))

    precision = recall_score(y_esperado, y_obtido)
    saida.append(precision)
    recall = precision_score(y_esperado, y_obtido)
    saida.append(recall)
    f_measure = f1_score(y_esperado, y_obtido)
    saida.append(f_measure)
    accuracy = accuracy_score(y_esperado, y_obtido)
    saida.append(accuracy)

    print(saida)


if __name__ == "__main__":
    main()