#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork,PrintMetricsList

import os

citiesList = [os.path.splitext(arquivo)[0] for arquivo in os.listdir('./Data') if os.path.isfile(os.path.join('./Data', arquivo))]
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
