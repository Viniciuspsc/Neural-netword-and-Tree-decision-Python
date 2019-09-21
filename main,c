#arvore de decisão, rede neural e bases de dados
from sklearn import tree, neural_network, datasets, metrics
import random 
from sklearn.tree import _tree

print()
print('Programa de comparação de aprendizado classificatório')

matricula=34706
random.seed(matricula)
nome = random.choice(['iris','wine','breast_cancer']) 
print('Base escolhida=', nome)

#base de testes
if nome == 'iris':
  banco = datasets.load_iris()
elif nome == 'wine':
  banco = datasets.load_wine()
else:
  banco = datasets.load_breast_cancer()

print()
print('Rede Neural')
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
rna = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,
                                  hidden_layer_sizes=(10), random_state=100)
#treinando rede
rna.fit(banco.data, banco.target)
#verificando
previsto= rna.predict(banco.data)
print(' Matriz de Confusão')
print(metrics.confusion_matrix(banco.target,previsto))
print(' Acurácia=',metrics.accuracy_score(banco.target,previsto))

print()
print('Nº de neurônios:')
print(' Input: 4')
print(' Hidden layer: 10')
print(' Output: 3')

print()
print('Pesos entre os Inputs e a Hidden layer')
for i in range(0,4):
  print()
  for j in range(0,10):
    w = i*10 + j
    print('In%i' %i, '--W%i-->' %w ,'H%i:' %j,' W%i ='%w, rna.coefs_[0][i][j])

print()
print('Bias da Hidden Layer')
for i in range(0,10):
  print('Bias H%i:'%i, rna.intercepts_[0][i])

print()
print()
print('Pesos entre a Hidden layer e as Outputs')
for i in range(0,10):
  print()
  for j in range(0,3):
    w = i*10 + j
    print('H%i' %i, '--W%i-->' %w ,'Out%i:' %j,' W%i ='%w, rna.coefs_[1][i][j])

print()
print('Bias da Output')
for i in range(0,3):
  print('Bias Out%i:'%i, rna.intercepts_[1][i])


print()
print()
print()
print('\nÁrvore de Decisão')
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dtree = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=5)
#treinando arvore
dtree = dtree.fit(banco.data, banco.target)
#verificando
previsto= dtree.predict(banco.data)
print(' Matriz de Confusão')
print(metrics.confusion_matrix(banco.target,previsto))
print(' Acurácia=',metrics.accuracy_score(banco.target,previsto))
print()

import numpy as np
from sklearn import tree

print('Entradas:')
for i in range(0,4):
  print(' ',banco.feature_names[i])
print('\nClasses saída:')  
for j in range(0,3):
  print(' ',banco.target_names[j])
print()  


#Geração da visão da arvore de decisão em forma de IF-ELSE
#Inspirado em: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
def get_code(tree, feature_names, target_names,
             spacer_base="    "):
   
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if (" + features[node] + " <= " + \
                  str(threshold[node]) + ") {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "Classe: " + str(target_name) + \
                      " ("+ str(target_count) +"examples)")

    recurse(left, right, threshold, features, 0, 0)
print()
print('Decisões dos nós em forma de if-else:\n')
print(get_code(dtree, banco.feature_names, banco.target_names))

#Decisoes que contem mais de uma classe, adotará a com mais exemplos



