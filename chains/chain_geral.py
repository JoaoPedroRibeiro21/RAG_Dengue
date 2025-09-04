from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

sys_prompt = """
Você é um assistente simpático e breve para saudações/assuntos gerais.
Evite falar sobre temas médicos específicos; se perguntarem sobre Dengue, diga que há um modo próprio para isso.

Responda de forma calorosa e natural, mantendo um tom conversacional e amigável.
""".strip()

hum_prompt = """
Usuário: {pergunta_usuario}

Histórico (pode estar vazio):
{history}
""".strip()

prompt_geral = ChatPromptTemplate(
    [("system", sys_prompt), ("human", hum_prompt)]
)

model_geral = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,  
    max_output_tokens=1024,  
)

chain_temas_nao_relacionados = prompt_geral | model_geral | StrOutputParser()