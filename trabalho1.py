# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:05:44 2019

@author: Marcelo Guedes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import argparse

 
class Perceptron():
    #método construtor 	
    def __init__(self, n_inputs, lr, epochs,weights):
        self.n_inputs=n_inputs
        self.lr =lr
        self.acuracy=[]
        self.loss=[]
        self.epochs =epochs
        self.weights=weights
        
    def predict(self, inputs):
        prod_escalar = np.dot(inputs, self.weights[1:]) + self.weights[0]*1
        if prod_escalar > 0:
            y = 1
        else:
            y= 0            
        return y
    
    
    def training(self,inputs,targets):
        for i in range(self.epochs):
            E=0
            hits=0
            for j in range (len(targets)):
                y= self.predict(inputs[j])
                self.weights[1:] =self.weights[1:] + self.lr*(targets[j]-y)*inputs[j]
                self.weights[0] = self.weights[0] + self.lr*(targets[j]-y)*1                
                E+=(y-targets[j])**2

                if(y==targets[j]):
                    hits=hits+1
            ac=100*hits/len(targets)
            self.acuracy.append(ac)
            self.loss.append(E)
            print('  Época {0:d} Acurácia {1:.3f}% Erro {2:d}'.format((1+i),ac,E))
           
    def plot_result_training(self):
        x = range(1, 1+self.epochs)
        y1 =self.acuracy
        y2 = self.loss
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(x, y1, 'g-')
        ax2.plot(x, y2, 'b-')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Y1 Acurácia', color='g')
        ax2.set_ylabel('Y2 Erro', color='b')
        string='Acurácia e Erro ao longo das épocas \n Taxa de aprendizagem:'+str(self.lr)+" Número de épocas: "+str(self.epochs)
        plt.title(string)
        plt.show()

    def test(self,inputs,targets):
        E=0
        hits=0
        for j in range (len(targets)):
            y= self.predict(inputs[j])
            E+=(y-targets[j])**2
            if(y==targets[j]):
                    hits+=1
        print(' Acurácia: {0:.3f}% Erro: {1:d}'.format(100*hits/len(targets),E))

#função para salvar o modelo se o usuário assim o desejar 
    def save_model(self,path):
        np.save(file=path,arr=self.weights)


#função para abrir um dataset e retorna os atributos e as classes em numpy
def open_dataset(path):
        if(os.path.isfile(path)):
		#Rocha e mina são categorizados para 0 e 1 respectivamente
            data= pd.read_csv(path,delimiter =',',header=None)
            data=data.replace(['R','M'],[0,1])
            features=data.iloc[:,:-1].values
            targets=data.iloc[:,-1].values
        else:
            print('Não foi encontrado dataset em', path)
            features,targtes=([],[])
        return (features, targets)
    

#carrega um modelo salvo e armazena na matriz de pesos ou cria uma matriz de pesos aleatoriamente
def load_model(path,op,n_inputs):
    if(os.path.isfile(path) and (op!=1)):
        weights=np.load(file=path)
        print('here')
    else:
        weights=np.random.random(n_inputs+1)*2 -1
        
    return weights


#função principal

def main(args):
    features_train, targets_train= open_dataset(args.train_path)
    features_test, targets_test= open_dataset(args.test_path)
    n_inputs=features_train.shape[1]
    weights=load_model(args.model_path+'weights.npy',args.op,n_inputs)  
	#cria um objeto da classe Perceptron passando-o os parâmetros do modelo
    perceptron = Perceptron(n_inputs,args.lr, args.n_epochs,weights)

    if(args.op!=3):
        print('============================================Training Model============================================')
        perceptron.training(features_train, targets_train)
        perceptron.plot_result_training()
        print('============================================Training Done============================================\n')
        if(args.ow_model):
            perceptron.save_model(args.model_path+'weights')

    print('================================================Test=================================================')
    
    perceptron.test(features_test, targets_test)

    


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # =========================================
        # MODEL PARAMETERS
    # -----------------------------------------
    #conjuntos de opções e parâmetros do modelo que o usuário dispõe para alterar
    parser.add_argument('--op', type=int, default=1, help='Para treinar e testar um novo modelo digite 1, Para retreinar um modelo salvo e testá-lo digite 2,  Para somente testar/avaliar um modelo salvo Digite 3')
    parser.add_argument('--ow_model', type=bool, 
        help='Sobrescrever modelo treinado: True, não substituir: False',default=False)
    parser.add_argument('--n_epochs', type=int, default=500,
        help='número de epocas a serem treinadas')
    parser.add_argument('--lr', type=float, 
        help="taxa de aprendizagem a ser utilizada",default=0.001)
    parser.add_argument('--model_path', type=str, default='saved_model/',
        help='caminho onde o modelo será salvo e importado')
    parser.add_argument('--train_path', type=str, default='DadosSonar/sonar.train-data.csv',
        help='caminho no qual os dados de treino serão importados')
    parser.add_argument('--test_path', type=str, default='DadosSonar/sonar.test-data.csv',
        help='caminho no qual os dados de teste serão importados')


    args = parser.parse_args()
    print(args)
    main(args)   
