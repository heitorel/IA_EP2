# IMPORTANDO AS BIBLIOTECAS E DEFININDO AS FUNÇÕES QUE SERÃO UTILIZADAS:

import numpy as np
import pandas as pd

def kfold(X, y, K): # FUNÇÃO QUE RECEBE UMA MATRIZ X DE PREDITORES, UM VETOR  ALVO y E O K NÚMERO DE GRUPOS A SEREM DEFINIDOS
    
    # DEFININDO O NÚMERO DE ELEMENTOS NOS CONJUNTOS DE TREINAMENTO E DE TESTE E CRIANDO OS RESPECTIVOS GRUPOS
    n_test = int(len(y)/K)
    n_train = len(y) - n_test
    X_train = np.zeros((K, n_train, np.shape(X)[1]))
    X_test = np.zeros((K, n_test, np.shape(X)[1]))
    y_train = np.zeros((K, n_train))
    y_test = np.zeros((K, n_test))
    
    # CRIANDO UM VETOR RANDOMIZADO (0-K) COM O GRUPO REFERENTE A CADA ELEMENTO DA AMOSTRA:
    A = np.arange(0, len(y))
    kk = 0
    for i in range(0, len(y)):
        A[i] = kk
        if i == int((len(y)/K)*(kk+1)) - 1:
            kk = kk + 1
    A = np.random.permutation(A)
    
    # PREENCHENDO AS K MATRIZES DE PREDITORES E OS K VETORES DE ALVO 
    for k in range(0, K):
        c_test = 0
        c_train = 0
        
        for i in range(0, len(A)):
            if A[i] == k:
                y_test[k][c_test] = y[i]
                for j in range(0, np.shape(X)[1]):
                    X_test[k][c_test][j] = X[i][j]
                c_test = c_test + 1
                
            else:
                y_train[k][c_train] = y[i]
                for j in range(0, np.shape(X)[1]):
                    X_train[k][c_train][j] = X[i][j]
                c_train = c_train + 1
                        
    return X_train, X_test, y_train, y_test # RETORNANDO OS GRUPOS PRONTOS PARA SEREM APLICADOS AO CLASSIFICADOR

def holdout(X, y, p): # FUNÇÃO QUE RECEBE UMA MATRIZ X DE ATRIBUTOS, UM VETOR  ALVO y E A PROPORÇÃO p DE TREINAMENTO
    
    # DEFININDO O NÚMERO DE ELEMENTOS NO CONJUNTO DE TREINAMENTO E DE TESTE COM BASE NA PROPORÇÃO
    n_train = int(len(y)*p)
    n_test = len(y) - n_train
    K = 1
    X_train = np.zeros((K, n_train, np.shape(X)[1]))
    X_test = np.zeros((K, n_test, np.shape(X)[1]))
    y_train = np.zeros((K, n_train))
    y_test = np.zeros((K, n_test))
    
    # CRIANDO UM VETOR BINÁRIO COM AS POSIÇÕES ALEATORIAMENTE PERMUTADAS
    A = np.zeros(len(y))
    for i in range(0, n_test):
        A[i] = 1
    A = np.random.permutation(A)
    
    # PREENCHENDO AS SAÍDAS COM OS ELEMENTOS DE CADA UM DOS DOIS GRUPOS (TREINAMENTO E TESTE)
    for k in range(0, K):
        c_test = 0
        c_train = 0
        
        for i in range(0, len(A)):
            if A[i] != k:
                y_test[k][c_test] = y[i]
                for j in range(0, np.shape(X)[1]):
                    X_test[k][c_test][j] = X[i][j]
                c_test = c_test + 1
                
            else:
                y_train[k][c_train] = y[i]
                for j in range(0, np.shape(X)[1]):
                    X_train[k][c_train][j] = X[i][j]
                c_train = c_train + 1
                        
    return X_train, X_test, y_train, y_test # RETORNANDO OS GRUPOS PRONTOS PARA SEREM APLICADOS AO CLASSIFICADOR

def likelyhood(y, Z): # FUNÇÃO DE VEROSSIMILHANÇA
    
    def gaussian(x, mu, sig): # FUNÇÃO QUE CALCULA A GAUSSIANA DA AMOSTRA x, DADOS SUA MÉDIA E SEU DESVIO
        res = 0
        if np.power(sig, 2.) != 0: # RESTRIÇÃO PARA O CASO EM QUE A O DESVIO É 0, DANDO DIVISÃO INDEFINIDA NO VALOR DE res
            res = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        return res
    
    prob = 1 # CÁLCULO DAS PROBABILIDADES ASSOCIADAS AOS GRUPOS DO NAIVE-BAYES 
    for j in np.arange(0, Z.shape[1]):
        m = np.mean(Z[:,j])
        s = np.std(Z[:,j])
        prob = prob*gaussian(y[j], m, s)

    return prob # RETORNO DA PROBABILIDADE CALCULADA


with open('Sentiment.txt', encoding='utf-8') as f: # LENDO O ARQUIVO PROVENIENTE DA BASE DE DADOS
    lines = f.readlines()

c = '.,!?' # CRIANDO UM STRING DE PONTUAÇÕES QUE DEVEM SER REMOVIDAS

# INICIALIZANDO O VOCABULÁRIO DE PALAVRAS E O VETOR COM OS RÓTULOS DAS FRASES "GOOD" OU "BAD":

LISTA = []
Y = []

# ITERANDO SOBRE TODOS OS TWEETS LIDOS E SEPARANDO CADA PALAVRA

maxi = 500 # NÚMERO MÁXIMO DE TWEETS QUE DEVEM SER CONSIDERADOS (RESTRIÇÃO QUE EVITA GRANDE CUSTO COMPUTACIONAL)
n_n = 0 # NÚMERO DE TWEETS COM RÓTULO "BAD"
n_p = 0 # NÚMERO DE TWEET COM RÓTULO "GOOD"

for i in range(0, maxi):
    print('Lendo o tweet', i+1)
    lista = lines[i].split()
    
    # REMOVENDO AS PONTUAÇÕES DO TWEET LIDO
    
    for j in range(0, len(lista)):
        for x in range(len(c)):
            lista[j] = lista[j].replace(c[x], "")
    
    # REMOVENDO O TERMO "10Sentiment140" DE CADA FRASE E CONSTRUINDO O VETOR ALVO:
    
    if lista[0][1] == '0':
        Y.append(0)
        n_n = n_n + 1
        
    else:
        Y.append(1)
        n_p = n_p + 1
    
    del lista[0]
    for k in range(0, len(lista)):
        aux = 0
        
        for j in range(0, len(LISTA)):
            if lista[k] == LISTA[j]:
                aux = 1
                break
        
        # CONSTRUINDO O VOCABULÁRIO DE TODAS AS PALAVRAS PRESENTES NOS TWEETS:
        
        if aux == 0:
            LISTA.append(lista[k])

# CRIANDO UMA MATRIZ COM OS PREDITORES DE CADA FRASE DE ACORDO COM A FREQUÊNCIA DE PALAVRAS:

X = np.zeros((maxi, len(LISTA)))

for i in range(0, maxi):
    print('Caracterizando o tweet', i+1)
    lista = lines[i].split()
    
    # REMOVENDO AS PONTUAÇÕES DA LISTA DE PALAVRAS
    
    for j in range(0, len(lista)):
        for x in range(len(c)):
            lista[j] = lista[j].replace(c[x], "")
    
    del lista[0]
    
    for k in range(0, len(lista)):
        for kk in range(0, len(LISTA)):
            if lista[k] == LISTA[kk]:
                X[i][kk] = X[i][kk] + 1
                break

# ORGANIZANDO OS DADOS OBTIDOS DA LEITURA EM TABELAS E ARRANJOS PARA POSTERIOR CLASSIFICAÇÃO

X = pd.DataFrame(X)
data = X
data['labels'] = Y
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
data = data.to_numpy()
nrow, ncol = data.shape
y = data[:, -1]
X = data[:, 0:ncol-1]

# VALIDAÇÃO PELA TÉCNICA K-FOLD:

K = 5 # NÚMERO DE FOLDS
X_train, X_test, y_train, y_test = kfold(X, y, K)
vp = 0 # Número de verdadeiros positivos
vn = 0 # Número de verdadeiros negativos
fp = 0 # Número de falsos positivos
fn = 0 # Número de falsos negativos

# ITERANDO SOBRE OS K GRUPOS
for k in range(0, K):
    
    P = pd.DataFrame(data=np.zeros((X_test.shape[1], len(classes))), columns = classes) # TABELA COM OS DADOS DE TESTE
    for i in np.arange(0, len(classes)):
        elements = tuple(np.where(y_train[k] == int(classes[i]))) # LOCALIZAÇÃO DOS ELEMENTOS PERTENCENTES A CADA CLASSE
        Z = X_train[k][elements,:][0]
        
        # ITERANDO SOBRE TODOS OS ELEMENTOS DE TESTE
        for j in np.arange(0,X_test.shape[1]):
            x = X_test[k][j,:]
            pj = likelyhood(x, Z) # CALCULANDO A FUNÇÃO DE VEROSSIMILHANÇA DOS PREDITORES DO ELEMENTO DE TESTE VS TREINAMENTO
            P[classes[i]][j] = pj*len(elements)/X_train.shape[1] # PREENCHIMENTO DA MATRIZ DE VEROSSIMILHANÇAS
    
    # CRIANDO O VETOR RESULTANTE DA CLASSIFICAÇÃO COM OS RÓTULOS PREDITOS
    y_pred = []
    for i in np.arange(0, P.shape[0]):
        c = np.argmax(np.array(P.iloc[[i]]))
        y_pred.append(P.columns[c]) # PREENCHENDO 
    y_pred = np.array(y_pred, dtype=int)

    for i in range(0, len(y_pred)):
        if y_pred[i] == 0 and y_test[k][i] == 0: # CASO SEJA VERDADEIRO NEGATIVO
            vn = vn + 1
        if y_pred[i] == 1 and y_test[k][i] == 1: # CASO SEJA VERDADEIRO POSITIVO
            vp = vp + 1
        if y_pred[i] == 0 and y_test[k][i] == 1: # CASO SEJA FALSO NEGATIVO
            fn = fn + 1
        if y_pred[i] == 1 and y_test[k][i] == 0: # CASO SEJA FALSO POSITIVO
            fn = fn + 1

print("Matriz de confusão:\n%d %d\n%d %d"%(vp, vn, fp, fn))
acu = (vp + vn)/(vp + vn + fp + fn)
print("Acurácia = %f"%(acu))

# VALIDAÇÃO PELA TÉCNICA HOLDOUT:

p = 0.7 # fracao de elementos no conjunto de treinamento
X_train, X_test, y_train, y_test = holdout(X, y, p)
vp = 0 # Número de verdadeiros positivos
vn = 0 # Número de verdadeiros negativos
fp = 0 # Número de falsos positivos
fn = 0 # Número de falsos negativos

# REALIZANDO O PROCESSO DE CLASSIFICAÇÃO 
for k in range(0, 1):
    
    P = pd.DataFrame(data=np.zeros((X_test.shape[1], len(classes))), columns = classes) # TABELA COM OS DADOS DE TESTE
    for i in np.arange(0, len(classes)):
        elements = tuple(np.where(y_train[k] == int(classes[i]))) # LOCALIZAÇÃO DOS ELEMENTOS PERTENCENTES A CADA CLASSE
        Z = X_train[k][elements,:][0]
        
        # ITERANDO SOBRE TODOS OS ELEMENTOS DE TESTE
        for j in np.arange(0,X_test.shape[1]):
            x = X_test[k][j,:]
            pj = likelyhood(x, Z) # CALCULANDO A FUNÇÃO DE VEROSSIMILHANÇA DOS PREDITORES DO ELEMENTO DE TESTE VS TREINAMENTO
            P[classes[i]][j] = pj*len(elements)/X_train.shape[1] # PREENCHIMENTO DA MATRIZ DE VEROSSIMILHANÇAS
    
    # CRIANDO O VETOR RESULTANTE DA CLASSIFICAÇÃO COM OS RÓTULOS PREDITOS
    y_pred = []
    for i in np.arange(0, P.shape[0]):
        c = np.argmax(np.array(P.iloc[[i]]))
        y_pred.append(P.columns[c]) # PREENCHENDO O VETOR DE RESPOSTA
    y_pred = np.array(y_pred, dtype=int)

    for i in range(0, len(y_pred)):
        if y_pred[i] == 0 and y_test[k][i] == 0: # CASO SEJA VERDADEIRO NEGATIVO
            vn = vn + 1
        if y_pred[i] == 1 and y_test[k][i] == 1: # CASO SEJA VERDADEIRO POSITIVO
            vp = vp + 1
        if y_pred[i] == 0 and y_test[k][i] == 1: # CASO SEJA FALSO NEGATIVO
            fn = fn + 1
        if y_pred[i] == 1 and y_test[k][i] == 0: # CASO SEJA FALSO POSITIVO
            fn = fn + 1
print("Matriz de confusão:\n%d %d\n%d %d"%(vp, vn, fp, fn))
acu = (vp + vn)/(vp + vn + fp + fn)
print("Acurácia = %f"%(acu))

# REMOVENDO AS STOP WORDS DO GOOGLE E CALCULANDO TREINANDO NOVAMENTE O CLASSIFICADOR:

c = '.,!?'
print("Quantidade antiga de palavras no vocabulário: %d"%(len(LISTA)))

with open('stopwords_en.txt', encoding='utf-8') as f: # LEITURA DO ARQUIVO QUE CONTÉM AS STOP WORDS
    liness = f.readlines()

for i in range(0, len(liness)):
    stop = liness[i].split() # SEPARAÇÃO DE CADA STOP WORD
    
    # PROCURA NA LISTA ORIGINAL SE EXISTE A STOP WORD E, SE SIM, APAGA A MESMA
    for j in range(0, len(LISTA)):
        if LISTA[j] == stop[0]:
            del(LISTA[j])
            break

# CRIA E PREENCHE UMA NOVA MATRIZ DE PREDITORES
X = np.zeros((maxi, len(LISTA)))
    
for i in range(0, maxi):
    lista = lines[i].split()
    
    # REMOVENDO AS PONTUAÇÕES DA LISTA DE PALAVRAS
    
    for j in range(0, len(lista)):
        for x in range(len(c)):
            lista[j] = lista[j].replace(c[x], "")
    
    del lista[0]
    
    for k in range(0, len(lista)):
        for kk in range(0, len(LISTA)):
            if lista[k] == LISTA[kk]:
                X[i][kk] = X[i][kk] + 1
                break

# CRIANDO UMA NOVA TABELA COM OS DADOS DE ENTRADA DO CLASSIFICADOR                
X = pd.DataFrame(X)
data = X
data['labels'] = Y
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
data = data.to_numpy()
nrow, ncol = data.shape
y = data[:, -1]
X = data[:, 0:ncol-1]
print("Quantidade nova de palavras no vocabulário: %d"%(len(LISTA)))