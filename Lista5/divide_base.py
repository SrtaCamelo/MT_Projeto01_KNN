import os
import shutil
import random

positivos = os.listdir('pos/')
negativos = os.listdir('neg/')

train_pos = []
train_neg = []
test_pos = []
test_neg = []

tamanho_train = (len(positivos) * 80) / 100
tamanho_test = (len(positivos) * 20) / 100

while len(train_neg) < tamanho_train:
    seed = random.randint(0, len(negativos)-1)
    if seed not in train_neg:
        train_neg.append(seed)

        try:
            antigo = 'neg/' + negativos[seed]
            novo = 'Base/Train/neg/' + negativos[seed]
            shutil.move(antigo, novo)
        except IndexError:
            print(seed)

while len(train_pos) < tamanho_train:
    seed = random.randint(0, len(positivos)-1)
    if seed not in train_pos:
        train_pos.append(seed)

        try:
            antigo = 'pos/' + positivos[seed]
            novo = 'Base/Train/pos/' + positivos[seed]
            shutil.move(antigo, novo)
        except IndexError:
            print(seed)

for arquivo in os.listdir('pos/'):
    antigo = 'pos/' + arquivo
    novo = 'Base/Test/pos/' + arquivo

    shutil.move(antigo, novo)

for arquivo in os.listdir('neg/'):
    antigo = 'neg/' + arquivo
    novo = 'Base/Test/neg/' + arquivo

    shutil.move(antigo, novo)