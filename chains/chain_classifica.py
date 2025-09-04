from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class RotaResposta(BaseModel):
    opcao: int = Field(
        description="1=Dúvidas sobre Dengue (RAG), 2=Saudações/gerais, 3=Cadastro (NOME e IDADE)"
    )
    justificativa: str = Field(default="", description="Breve justificativa")

parser_classifica = PydanticOutputParser(pydantic_object=RotaResposta)

EXEMPLOS = """
[EXEMPLOS]
Usuário: "Quais os sintomas da dengue e como prevenir?"
→ 1

Usuário: "oi, tudo bem? bom dia"
→ 2

Usuário: "quero me cadastrar: meu nome é Ana"
→ 3

Usuário: "tenho 22 anos, posso continuar o cadastro?"
→ 3

Usuário: "obrigado!"
→ 2
"""

sys_prompt_rota = f"""
Você é um classificador. Escolha exatamente UMA opção:

1 = Dúvidas sobre Dengue (usar RAG)
2 = Saudações/assuntos gerais (sem RAG, resposta curta)
3 = Cadastro (coletar apenas NOME e IDADE do usuário; conclusão quando ele disser "concluir"/"finalizar")

Retorne SOMENTE o JSON no formato:
{{format_instructions}}

{EXEMPLOS}
""".strip()

hum_prompt = """
Pergunta do usuário:
{input}

Histórico (pode estar vazio):
{history}
""".strip()

rota_prompt_template = ChatPromptTemplate(
    [("system", sys_prompt_rota), ("human", hum_prompt)],
    partial_variables={"format_instructions": parser_classifica.get_format_instructions()},
)

model_classificador = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=200,  
)

chain_de_roteamento = rota_prompt_template | model_classificador | parser_classifica