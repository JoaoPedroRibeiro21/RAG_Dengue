# 🦟 Assistente Virtual sobre Dengue (RAG + LLM)

Este projeto é um **assistente virtual sobre Dengue**, desenvolvido como prática em **LLMs, RAG (Retrieval-Augmented Generation)** e bancos vetoriais.  

O sistema responde dúvidas sobre **sintomas, sinais de alarme, prevenção e cuidados**, a partir de informações do SUS retirados da internet e reunidos em um PDF.  
Também possui uma função de **cadastro simples** (nome e idade), registrando os dados em um arquivo `.csv`.

---

## 🚀 Tecnologias utilizadas

- [**LangChain**](https://www.langchain.com/) → orquestração de chains e histórico de conversas.  
- [**ChromaDB**](https://www.trychroma.com/) → banco vetorial para indexação e busca semântica.  
- [**Google Gemini**](https://ai.google.dev/) → LLM para respostas e embeddings.  
- [**Chainlit**](https://docs.chainlit.io/) → interface de chat para interação com o usuário.  
- **PyPDF / PyMuPDF** → leitura de documentos PDF.  
- **CSV** → persistência simples dos cadastros.  

---

## 📂 Estrutura do projeto

```
RAG_Dengue/
├── chains/                  # Chains de classificação, RAG, cadastro, etc.
│   ├── chain_classifica.py
│   ├── chain_rag_duvidas.py
│   ├── chain_geral.py
│   └── chain_registro_ocorrencia.py
├── memorias/
│   └── memoria.py           # Histórico de conversas
├── files/
│   ├── cadastros.csv        # Arquivo de saída dos cadastros (gerado em runtime)
│   └── DENGUE.PDF  # PDFs usados no RAG
├── db_dengue/               # Persistência do ChromaDB (ignorado no git)
├── main.py                  # Ponto de entrada do Chainlit
├── indexa_informacao.py     # Script para indexar documentos no ChromaDB
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Instalação

Clone o repositório e entre na pasta:

```bash
git clone https://github.com/seuusuario/RAG_Dengue.git
cd RAG_Dengue
```

Crie o ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Configuração

Crie um arquivo `.env` na raiz com a sua chave do **Google Gemini**:

```
GOOGLE_API_KEY=suachave_aqui
```

---

## 📄 Indexação dos documentos

Coloque o PDF base em `files/BaseDeConhecimento_PDF/`.

Depois, rode:

```bash
python3 indexa_informacao.py
```

Isso cria/atualiza o banco vetorial **ChromaDB** em `db_dengue/`.

---

## 💬 Executando o assistente

Inicie a interface Chainlit:

```bash
chainlit run main.py -w
```

O app ficará disponível em:  
👉 [http://localhost:8000](http://localhost:8000)

---

## 🧾 Funcionalidades

- **Chat com RAG**: perguntas sobre Dengue são respondidas com base no PDF indexado.  
- **Respostas gerais**: saudações e temas fora de escopo são tratados de forma amigável.  
- **Cadastro simples**: usuário informa **nome e idade**, e os dados são salvos em `files/cadastros.csv`.  
- **Fluxo inteligente**: caso sintomas sejam relatados, o assistente sugere o cadastro.  

---

## ⚠️ Aviso importante

Este projeto tem caráter **educacional e experimental**.  
As informações fornecidas não substituem **orientação médica profissional**.  
Em caso de sintomas, procure uma **UBS/UPA ou profissional de saúde**.

