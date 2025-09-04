import os
import csv
import re
from typing import Optional
from operator import itemgetter
from dotenv import load_dotenv

from pydantic import BaseModel, Field, field_validator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class CadastroPessoa(BaseModel):
    """
    Estrutura do cadastro: apenas nome e idade, e sinalização de conclusão.
    """
    nome: Optional[str] = Field(default=None, description="Nome completo do usuário, se informado.")
    idade: Optional[int] = Field(default=None, description="Idade em anos, se informada.")
    concluir: bool = Field(default=False, description="True quando o usuário disser 'concluir', 'finalizar', 'enviar', etc.")

    @field_validator("idade", mode="before")
    @classmethod
    def _coerce_idade(cls, v):
        
        if v is None:
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            m = re.search(r"\b(\d{1,3})\b", v)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return v

SISTEMA_CADASTRO = """
Você extrairá informações para um cadastro simples de pessoa.
Capture SOMENTE:
- nome
- idade (em anos)
- concluir (true/false) → true quando o usuário indicar claramente "concluir", "finalizar", "enviar", "pode registrar", etc.

Regras:
- NÃO invente dados. Se não houver no texto, deixe nulo.
- O campo idade deve ser um número inteiro (anos).
- Se o usuário só disser "concluir" sem ter informado os dados, apenas marque concluir=true e deixe os campos ausentes como nulos.

A saída DEVE seguir exatamente o schema definido (nome, idade, concluir).
""".strip()

HUMANO_CADASTRO = """
Mensagem do usuário:
{pergunta_usuario}

Histórico (pode estar vazio):
{history}
""".strip()

cadastro_prompt = ChatPromptTemplate(
    messages=[
        ("system", SISTEMA_CADASTRO),
        ("human", HUMANO_CADASTRO),
    ]
)

SISTEMA_FINAL = """
Você é um atendente cordial. Responda de forma breve e clara.
- Se CADASTRO_OK: confirme que foi registrado com sucesso (sem citar fontes ou IDs).
- Se CADASTRO_PENDENTE: peça apenas os campos faltantes (nome e/ou idade), educadamente.
Evite textos longos e termos técnicos.
""".strip()

HUMANO_FINAL = """
Resultado (sistema):
{acao_executada}

Mensagem original:
{pergunta_usuario}

Histórico:
{history}
""".strip()

chat_prompt_final = ChatPromptTemplate(
    messages=[("system", SISTEMA_FINAL), ("human", HUMANO_FINAL)]
)

_model_extracao_base = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, 
)

model_extracao = _model_extracao_base.with_structured_output(CadastroPessoa)


model_final = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


CSV_PATH = os.path.join("files", "cadastros.csv")
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def _csv_tem_cabecalho(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def _persistir(cad: CadastroPessoa) -> None:
    write_header = not _csv_tem_cabecalho(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nome", "idade"])
        if write_header:
            w.writeheader()
        w.writerow({
            "nome": cad.nome or "",
            "idade": cad.idade if cad.idade is not None else "",
        })


def processa_cadastro(cad: CadastroPessoa) -> str:
    """
    Se concluir==True e nome/idade presentes → persiste e retorna CADASTRO_OK.
    Caso contrário, informa o que falta e pede.
    """
    tem_nome = bool(cad.nome and cad.nome.strip())
    tem_idade = cad.idade is not None

    if cad.concluir and tem_nome and tem_idade:
        _persistir(cad)
        return (
            "CADASTRO_OK\n"
            f"Nome: {cad.nome}\n"
            f"Idade: {cad.idade}"
        )

    faltas = []
    if not tem_nome:
        faltas.append("nome")
    if not tem_idade:
        faltas.append("idade")

    faltantes = ", ".join(faltas) if faltas else "nenhum"
    return (
        "CADASTRO_PENDENTE\n"
        f"Concluir sinalizado: {'sim' if cad.concluir else 'não'}\n"
        f"Campos faltantes: {faltantes}\n"
        f"Parciais capturados — Nome: {cad.nome or '-'} | Idade: {cad.idade if cad.idade is not None else '-'}\n"
        "Por favor, informe os campos faltantes. Quando terminar, digite **concluir**."
    )

chain_de_cadastro = (
    RunnableParallel({
        
        "acao_executada": cadastro_prompt | model_extracao | RunnableLambda(processa_cadastro),
        "history": itemgetter("history"),
        "pergunta_usuario": itemgetter("pergunta_usuario"),
    })
    
    | chat_prompt_final
    | model_final
    | StrOutputParser()
)
