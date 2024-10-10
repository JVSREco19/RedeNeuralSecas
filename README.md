# Rede Neural Secas



## Configuração

Insira os arquivos de dados das cidades no formato `.xlsx` na pasta `Data`. Cada arquivo deve estar no seguinte formato:

![alt text](image.png)

Onde a primeira coluna é o valor do SPEI e a segunda coluna é a data correspondente.

Os dados podem ser adquiridos através deste repositório: [GenerateCitiesSPEI](https://github.com/JVSREco19/GenerateCitiesSPEI).

A estrutura do diretório "Data" deve ser similar à seguinte, para rodar `OneToMany.py`:
```
Data
├───ESPINOSA
│       ESPINOSA.xlsx
│       GAMELEIRAS.xlsx
│       MAMONAS.xlsx
│       MONTE AZUL.xlsx
│       MONTEZUMA.xlsx
│       SANTO ANTÔNIO DO RETIRO.xlsx
│
├───LASSANCE
│       AUGUSTO DE LIMA.xlsx
│       BUENÓPOLIS.xlsx
│       BURITIZEIRO.xlsx
│       CORINTO.xlsx
│       FRANCISCO DUMONT.xlsx
│       JOAQUIM FELÍCIO.xlsx
│       LASSANCE.xlsx
│       TRÊS MARIAS.xlsx
│       VÁRZEA DA PALMA.xlsx
│
├───RIO PARDO DE MINAS
│       FRUTA DE LEITE.xlsx
│       GRÃO MOGOL.xlsx
│       INDAIABIRA.xlsx
│       MATO VERDE.xlsx
│       MONTEZUMA.xlsx
│       NOVORIZONTE.xlsx
│       PADRE CARVALHO.xlsx
│       PORTEIRINHA.xlsx
│       RIACHO DOS MACHADOS.xlsx
│       RIO PARDO DE MINAS.xlsx
│       SALINAS.xlsx
│       SANTO ANTÔNIO DO RETIRO.xlsx
│       SERRANÓPOLIS DE MINAS.xlsx
│       TAIOBEIRAS.xlsx
│       VARGEM GRANDE DO RIO PARDO.xlsx
│
├───SÃO FRANCISCO
│       BRASÍLIA DE MINAS.xlsx
│       CHAPADA GAÚCHA.xlsx
│       ICARAÍ DE MINAS.xlsx
│       JANUÁRIA.xlsx
│       JAPONVAR.xlsx
│       LUISLÂNDIA.xlsx
│       PEDRAS DE MARIA DA CRUZ.xlsx
│       PINTÓPOLIS.xlsx
│       SÃO FRANCISCO.xlsx
│
└───SÃO JOÃO DA PONTE
        CAPITÃO ENÉAS.xlsx
        IBIRACATU.xlsx
        JANAÚBA.xlsx
        JAPONVAR.xlsx
        LONTRA.xlsx
        MONTES CLAROS.xlsx
        PATIS.xlsx
        SÃO JOÃO DA PONTE.xlsx
        VARZELÂNDIA.xlsx
        VERDELÂNDIA.xlsx
```

## Saídas
O programa `OneToMany.py` tem duas saídas: uma planilha `metricas_modelo.xlsx` e inúmeros arquivos de gráficos de métricas de desempenho.

Os arquivos são salvos numa árvore de diretórios que possui a seguinte estrutura:
```
Images
├───cluster ESPINOSA
│   └───model ESPINOSA
│       ├───city ESPINOSA
│       ├───city GAMELEIRAS
│       ├───city MAMONAS
│       ├───city MONTE AZUL
│       ├───city MONTEZUMA
│       └───city SANTO ANTÔNIO DO RETIRO
├───cluster LASSANCE
│   └───model LASSANCE
│       ├───city AUGUSTO DE LIMA
│       ├───city BUENÓPOLIS
│       ├───city BURITIZEIRO
│       ├───city CORINTO
│       ├───city FRANCISCO DUMONT
│       ├───city JOAQUIM FELÍCIO
│       ├───city LASSANCE
│       ├───city TRÊS MARIAS
│       └───city VÁRZEA DA PALMA
├───cluster RIO PARDO DE MINAS
│   └───model RIO PARDO DE MINAS
│       ├───city FRUTA DE LEITE
│       ├───city GRÃO MOGOL
│       ├───city INDAIABIRA
│       ├───city MATO VERDE
│       ├───city MONTEZUMA
│       ├───city NOVORIZONTE
│       ├───city PADRE CARVALHO
│       ├───city PORTEIRINHA
│       ├───city RIACHO DOS MACHADOS
│       ├───city RIO PARDO DE MINAS
│       ├───city SALINAS
│       ├───city SANTO ANTÔNIO DO RETIRO
│       ├───city SERRANÓPOLIS DE MINAS
│       ├───city TAIOBEIRAS
│       └───city VARGEM GRANDE DO RIO PARDO
├───cluster SÃO FRANCISCO
│   └───model SÃO FRANCISCO
│       ├───city BRASÍLIA DE MINAS
│       ├───city CHAPADA GAÚCHA
│       ├───city ICARAÍ DE MINAS
│       ├───city JANUÁRIA
│       ├───city JAPONVAR
│       ├───city LUISLÂNDIA
│       ├───city PEDRAS DE MARIA DA CRUZ
│       ├───city PINTÓPOLIS
│       └───city SÃO FRANCISCO
└───cluster SÃO JOÃO DA PONTE
    └───model SÃO JOÃO DA PONTE
        ├───city CAPITÃO ENÉAS
        ├───city IBIRACATU
        ├───city JANAÚBA
        ├───city JAPONVAR
        ├───city LONTRA
        ├───city MONTES CLAROS
        ├───city PATIS
        ├───city SÃO JOÃO DA PONTE
        ├───city VARZELÂNDIA
        └───city VERDELÂNDIA
```
