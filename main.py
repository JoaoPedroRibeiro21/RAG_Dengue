import os
import asyncio
import chainlit as cl
from operator import itemgetter

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["LC_LOG_LEVEL"] = "ERROR"

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHAINLIT_LOG_LEVEL"] = "ERROR"

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from memorias.memoria import get_session_history, trimmer
from chains.chain_classifica import chain_de_roteamento
from chains.chain_rag_duvidas import chain_orientador
from chains.chain_geral import chain_temas_nao_relacionados
from chains.chain_registro_ocorrencia import chain_de_cadastro


async def simulate_streaming(text: str, message: cl.Message, chunk_size: int = 3) -> None:
    """
    Simula streaming dividindo o texto em chunks de palavras
    """
    if not text or len(text) < 20:
        await message.stream_token(text)
        return
    
    words = text.split(' ')
    current_chunk = ""
    
    for i, word in enumerate(words):
        if i > 0:
            current_chunk += " "
        current_chunk += word
        
    
        if (i + 1) % chunk_size == 0 or i == len(words) - 1:
            await message.stream_token(current_chunk)
            current_chunk = ""
            
            
            await asyncio.sleep(0.02)

def _escolhe_rota(entrada: dict):
    opcao = entrada["resposta_pydantic"].opcao
    if opcao == 1:
        rota = chain_orientador                 
    elif opcao == 2:
        rota = chain_temas_nao_relacionados     
    elif opcao == 3:
        rota = chain_de_cadastro                
    else:
        rota = chain_temas_nao_relacionados

    return RunnableLambda(lambda x: {
        "pergunta_usuario": x["input"],
        "history": x["history"],
    }) | rota

chain_principal = (
    RunnableParallel({
        "input": itemgetter("input"),
        "history": itemgetter("history"),
        "resposta_pydantic": chain_de_roteamento
    })
    | RunnableLambda(_escolhe_rota)
)

chain_principal_com_trimming = (
    RunnablePassthrough.assign(history=itemgetter("history") | RunnableLambda(trimmer))
    | chain_principal
)

runnable_with_history = RunnableWithMessageHistory(
    chain_principal_com_trimming,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

@cl.on_chat_start
async def start():
    await cl.Message(content=(
        "👋 Olá! Eu sou o **Assistente Virtual sobre Dengue**.\n\n"
        "Estou aqui para ajudar você com informações confiáveis sobre a dengue:\n"
        "• Tirar dúvidas sobre sintomas, transmissão e prevenção 🦟\n"
        "• Orientar sobre quando procurar atendimento médico 🏥\n"
        "• Apoiar no registro de informações básicas (nome e idade) 📋\n\n"
        "Como posso te ajudar hoje?"
    )).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_input = (message.content or "").strip()
    if not user_input:
        await cl.Message("⚠️ Digite uma mensagem.").send()
        return

    print(f"\n🔍 Processando mensagem: {user_input}")
    
    session_id = cl.user_session.get("id") or "default"
    try:
        print("🔄 Executando pipeline principal com streaming...")
        
        
        resp = await runnable_with_history.ainvoke(
            {"input": user_input, "history": []},
            config={"configurable": {"session_id": session_id}}
        )
        
        final_text = resp if isinstance(resp, str) else (
            resp.get("output") or resp.get("content") or str(resp)
        )

        
    
        response_msg = cl.Message(content="")
        await response_msg.send()
        await simulate_streaming(final_text, response_msg, chunk_size=3)
        
        print("✅ Resposta enviada com sucesso")

    except Exception as e:
        print(f"❌ Erro no pipeline principal: {e}")
        error_msg = cl.Message(content="")
        await error_msg.send()
        
        error_text = (
            f"❌ **Erro ao processar sua mensagem**\n\n"
            f"**Detalhes técnicos**: `{str(e)}`\n\n"
            f"💡 **Sugestões**:\n"
            f"• Tente reformular sua pergunta\n"
            f"• Seja mais específico sobre o que deseja saber\n"
            f"• Para dúvidas sobre dengue, use termos como: sintomas, prevenção, tratamento\n\n"
            f"🔄 Tente novamente em alguns instantes."
        )
        
        await simulate_streaming(error_text, error_msg, chunk_size=2)

@cl.on_chat_resume  
async def on_resume():
    """Executado quando o chat é retomado"""
    resume_msg = cl.Message(content="")
    await resume_msg.send()
    
    resume_text = (
        "🔄 **Chat retomado!**\n\n"
        "Continue fazendo suas perguntas sobre dengue baseadas no conteúdo do PDF.\n"
        "⚡ **Streaming ativo** - respostas aparecem em tempo real!"
    )
    
    await simulate_streaming(resume_text, resume_msg, chunk_size=4)

if __name__ == "__main__":
    print("🚀 Iniciando o assistente de dengue com STREAMING...")
    print("📚 Sistema focado em análise de texto (PDF)")
    print("🔗 Execute: chainlit run main.py")