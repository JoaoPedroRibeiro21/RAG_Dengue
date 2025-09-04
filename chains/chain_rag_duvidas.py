import os
import re
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

__all__ = ["chain_orientador"]

EMBEDDING_MODEL = "models/text-embedding-004"
DB_DIR = "db_dengue"
COLLECTION = "dengue"


SINTOMAS_PADRAO = [
    r"febre(?: (?:alta|repentina))?",
    r"dor(?:es)? de cabeça",
    r"dor(?:es)? (?:no corpo|musculares|nas articula(?:ç|c)ões)",
    r"mialgia",
    r"artralgia",
    r"dor(?:es)? (?:atr[aá]s|retro) dos olhos",
    r"manchas (?:vermelhas|na pele)|exantema",
    r"cansa[çc]o|fadiga|prostra[çc][aã]o",
    r"n[áa]usea[s]?|enjoo",
    r"v[oó]mito[s]?",
    r"diarreia",
    r"perda de apetite",
]
REGEX_SINTOMAS = re.compile(r"(?i)\b(" + r"|".join(SINTOMAS_PADRAO) + r")\b")


SINAIS_ALARME = [
    r"dor abdominal (?:intensa|forte) (?:e )?cont[ií]nua",
    r"v[oó]mitos? persistentes?",
    r"sangramento (?:nasal|gengival|vaginal|de pele)|hematomas? f[áa]ceis|pet[eé]quias",
    r"tontura|desmaio|hipotens[aã]o|queda de press[aã]o",
    r"letargia|irritabilidade",
    r"hepatomegalia|f[íi]gado aumentado|dor no f[íi]gado",
    r"hemorragi(?:a|as)|hemat[ée]mese|melena",
]
REGEX_ALARME = re.compile(r"(?i)\b(" + r"|".join(SINAIS_ALARME) + r")\b")

def _tem_sintomas(texto: str) -> bool:
    return bool(texto and REGEX_SINTOMAS.search(texto))

def _tem_alarme(texto: str) -> bool:
    return bool(texto and REGEX_ALARME.search(texto))

def _cta(pergunta: str, resposta: str) -> str:
    """CTA curto e cordial conforme detecção."""
    texto = f"{pergunta}\n\n{resposta or ''}"
    if _tem_alarme(texto):
        return (
            "\n\n⚠️ **Atenção:** há sinais que podem indicar **gravidade**. "
            "Procure avaliação **imediata** em uma UBS/UPA. "
            "Se preferir, posso **registrar seus dados** para acompanhamento — informe **nome** e **idade**, "
            "e diga **concluir** ao terminar."
        )
    if _tem_sintomas(texto):
        return (
            "\n\n📝 Se você está com esses sintomas, posso **registrar seus dados** para acompanhamento. "
            "Digite seu **nome** e **idade**; ao finalizar, escreva **concluir**."
        )
    return ""


def _embeddings():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY não definido.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=key)

def _chroma(emb):
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=emb,
    )

def _fmt_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(
        d.page_content.strip() for d in docs
        if d and isinstance(d.page_content, str) and d.page_content.strip()
    )


rag_system = """
Você é um assistente de saúde. Responda SOMENTE com base no contexto abaixo sobre Dengue.
Se a informação não estiver no contexto, diga que não encontrou no material e oriente procurar uma UBS.
Produza uma resposta clara, **completa** e **organizada** em seções, sem referências, links, IDs de trechos ou citações de fonte.
Evite falar com base no material fornecido ou similares informando, apresente a informação sem essas citações.

Estrutura recomendada (use apenas as partes suportadas pelo contexto):
- **Sintomas típicos:** detalhe os principais (ex.: febre alta de início súbito, dor de cabeça, dores musculares/articulares, dor atrás dos olhos, manchas, náuseas/vômitos, etc.).
- **Evolução temporal da doença:** em linhas gerais (ex.: período febril, possíveis sinais que surgem após queda da febre, duração aproximada).
- **Sinais de alarme:** liste claramente e destaque que indicam gravidade e exigem avaliação imediata.
- **Quando procurar atendimento:** critérios práticos.
- **Cuidados em casa e hidratação:** recomendações objetivas.
- **Populações especiais:** gestantes, crianças, idosos — ressalte cuidado e procura precoce.

Contexto:
{contexto_obtido}
""".strip()

rag_human = """
Pergunta: {pergunta_usuario}

Histórico (pode estar vazio):
{history}
""".strip()

prompt_template_orientador = ChatPromptTemplate(
    [("system", rag_system), ("human", rag_human)]
)


model_atendimento_orientador = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)

try:
    _emb = _embeddings()
    _db = _chroma(_emb)


    retriever_mmr = _db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 36}
    )
    retriever_bm25 = _db.as_retriever(search_kwargs={"k": 12})

    def _busca_contexto(pergunta: str):
        docs = retriever_mmr.invoke(pergunta or "")
        if not docs:
            docs = retriever_bm25.invoke(pergunta or "")
        return _fmt_docs(docs)

    def _append_cta(payload: dict, resposta_base: str) -> str:
        pergunta = payload.get("pergunta_usuario", "")
        extra = _cta(pergunta, resposta_base)
        return (resposta_base or "") + extra

    chain_orientador = (
        RunnableParallel({
            "pergunta_usuario": itemgetter("pergunta_usuario"),
            "history": itemgetter("history"),
            "contexto_obtido": itemgetter("pergunta_usuario")
                | RunnableLambda(_busca_contexto),
        })
        | prompt_template_orientador
        | model_atendimento_orientador
        | StrOutputParser()
        | RunnableLambda(lambda out, **kw: _append_cta(kw.get("input", {}) if isinstance(kw, dict) else {}, out))
    )

except Exception as e:
    def _erro(_):
        return (
            "⚠️ RAG indisponível no momento.\n\n"
            f"Motivo: {type(e).__name__}: {e}\n"
            "Verifique a indexação (db_dengue) e GOOGLE_API_KEY."
        )
    chain_orientador = RunnableLambda(_erro) | StrOutputParser()
