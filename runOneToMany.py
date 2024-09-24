#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork,PrintMetricsList

import os

def nomes_arquivos_sem_extensao(caminho_pasta):
    # Lista todos os arquivos no diretório
    arquivos = os.listdir(caminho_pasta)
    
    # Remove a extensão .xlsx de cada nome de arquivo
    nomes_sem_extensao = [arquivo.removesuffix('.xlsx') for arquivo in arquivos]
    
    return nomes_sem_extensao

citiesList = nomes_arquivos_sem_extensao('./Data')
for city in citiesList:
  caminho = './Images/'+city
  if not os.path.exists(caminho):
    # Cria o diretório se ele não existir
    os.makedirs(caminho)
    print(f"Pasta '{caminho}' criada com sucesso.")

      
showImages = False

city = 'SÃO JOÃO DA PONTE'
model = FitNeuralNetwork('./Data/'+city+'.xlsx', city,showImages)
citiesListAux = citiesList.copy()
citiesListAux.remove(city)

for cityAux in citiesListAux: 
  #Deve-se passar o caminho para o xlsx, o nome da Região e o modelo treinado;
  ApplyTraining("./Data/"+cityAux+".xlsx", cityAux, model,showImages,city)
  
PrintMetricsList()
