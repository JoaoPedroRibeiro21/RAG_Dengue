# RAG_Dengue/indexa_informacao.py
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ------------------------------
# Configurações
# ------------------------------
PDF_PATH = "files/DENGUE.PDF"   # caminho fixo do PDF
DB_DIR = "db_dengue"
COLLECTION = "dengue"

EMBEDDING_MODEL = "models/text-embedding-004"

# ------------------------------
# Funções auxiliares
# ------------------------------
def carregar_pdf(caminho: str):
    """Carrega o PDF especificado."""
    if not os.path.exists(caminho):
        print(f"⚠️ Arquivo não encontrado: {caminho}")
        return []

    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        print(f"📄 Carregado: {caminho}")
        return documentos
    except Exception as e:
        print(f"❌ Erro ao carregar {caminho}: {e}")
        return []


def dividir_em_chunks(documentos, chunk_size=1000, chunk_overlap=100):
    """Divide documentos em pedaços menores."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documentos)


def criar_embeddings():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY não definido no .env")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=key)


def salvar_no_chroma(docs, embeddings):
    """Salva os documentos no ChromaDB persistente."""
    if not docs:
        print("⚠️ Nenhum documento para salvar.")
        return

    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    db.add_documents(docs)
    print(f"✅ Indexação concluída. Chunks: {len(docs)} | DB: {DB_DIR}")


# ------------------------------
# Execução principal
# ------------------------------
def main():
    print("📄 Lendo PDF...")
    docs = carregar_pdf(PDF_PATH)

    if not docs:
        return

    print("✂️ Dividindo em chunks...")
    chunks = dividir_em_chunks(docs)

    print("🧮 Gerando embeddings e salvando no ChromaDB...")
    embeddings = criar_embeddings()
    salvar_no_chroma(chunks, embeddings)


if __name__ == "__main__":
    main()
