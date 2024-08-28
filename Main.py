#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork

model = FitNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo")

#Deve-se passar o caminho para o xlsx, o nome da Região e o modelo treinado;
ApplyTraining("./Data/spei12_FranciscoSá.xlsx", "Francisco Sá", model)
#ApplyTraining("./Data/spei12_GrãoMogol.xlsx", "Grão Mogol", model)
#ApplyTraining("./Data/spei12_Josenopolis.xlsx", "Josenópolis", model)


