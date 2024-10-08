# Rede Neural Secas



## Configuração

### 1. Preparar os Dados

Insira os arquivos de dados das cidades no formato `.xlsx` na pasta `Data`. Cada arquivo deve estar no seguinte formato:

![alt text](image.png)

Onde a primeira coluna é o valor do SPEI e a segunda coluna é a data correspondente.

Os dados podem ser adquiridos através deste repositório: [GenerateCitiesSPEI](https://github.com/JVSREco19/GenerateCitiesSPEI).

A estrutura do diretório "Data" deve ser a seguinte, para rodar `OneToMany.py`:

  ├───Data
  │   ├───ESPINOSA
  │   │       ESPINOSA.xlsx
  │   │       GAMELEIRAS.xlsx
  │   │       MAMONAS.xlsx
  │   │       MONTE AZUL.xlsx
  │   │       MONTEZUMA.xlsx
  │   │       SANTO ANTÔNIO DO RETIRO.xlsx
  │   │
  │   ├───LASSANCE
  │   │       AUGUSTO DE LIMA.xlsx
  │   │       BUENÓPOLIS.xlsx
  │   │       BURITIZEIRO.xlsx
  │   │       CORINTO.xlsx
  │   │       FRANCISCO DUMONT.xlsx
  │   │       JOAQUIM FELÍCIO.xlsx
  │   │       LASSANCE.xlsx
  │   │       TRÊS MARIAS.xlsx
  │   │       VÁRZEA DA PALMA.xlsx
  │   │
  │   ├───RIO PARDO DE MINAS
  │   │       FRUTA DE LEITE.xlsx
  │   │       GRÃO MOGOL.xlsx
  │   │       INDAIABIRA.xlsx
  │   │       MATO VERDE.xlsx
  │   │       MONTEZUMA.xlsx
  │   │       NOVORIZONTE.xlsx
  │   │       PADRE CARVALHO.xlsx
  │   │       PORTEIRINHA.xlsx
  │   │       RIACHO DOS MACHADOS.xlsx
  │   │       RIO PARDO DE MINAS.xlsx
  │   │       SALINAS.xlsx
  │   │       SANTO ANTÔNIO DO RETIRO.xlsx
  │   │       SERRANÓPOLIS DE MINAS.xlsx
  │   │       TAIOBEIRAS.xlsx
  │   │       VARGEM GRANDE DO RIO PARDO.xlsx
  │   │
  │   ├───SÃO FRANCISCO
  │   │       BRASÍLIA DE MINAS.xlsx
  │   │       CHAPADA GAÚCHA.xlsx
  │   │       ICARAÍ DE MINAS.xlsx
  │   │       JANUÁRIA.xlsx
  │   │       JAPONVAR.xlsx
  │   │       LUISLÂNDIA.xlsx
  │   │       PEDRAS DE MARIA DA CRUZ.xlsx
  │   │       PINTÓPOLIS.xlsx
  │   │       SÃO FRANCISCO.xlsx
  │   │
  │   └───SÃO JOÃO DA PONTE
  │           CAPITÃO ENÉAS.xlsx
  │           IBIRACATU.xlsx
  │           JANAÚBA.xlsx
  │           JAPONVAR.xlsx
  │           LONTRA.xlsx
  │           MONTES CLAROS.xlsx
  │           PATIS.xlsx
  │           SÃO JOÃO DA PONTE.xlsx
  │           VARZELÂNDIA.xlsx
  │           VERDELÂNDIA.xlsx

2. **Executar os Scripts:**

Existem dois scripts principais que você pode usar para treinar e testar o modelo:

- **`runOneToMany.py`**: Este script usa uma cidade como modelo para prever o SPEI das demais cidades. Para usá-lo, defina o nome da cidade na variável `city` na linha 35 do script. O modelo será treinado usando essa cidade e fará previsões para todas as outras cidades na pasta `Data`.

  ```python
  # Defina o nome da cidade aqui
  city = "NomeDaCidade"
  ```

- **`runManyToMany.py`**: Este script compara todas as cidades da pasta `Data` entre si. Por exemplo, se houver 5 cidades, você obterá 25 resultados (cada cidade sendo usada como modelo para todas as outras).

