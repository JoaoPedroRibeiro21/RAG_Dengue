from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory

_SESSIONS: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ChatMessageHistory()
    return _SESSIONS[session_id]

def _trim_history(messages, max_pairs: int = 4):
    if len(messages) <= max_pairs * 2:
        return messages
    return messages[-max_pairs*2:]

def trimmer(history: ChatMessageHistory):
    if hasattr(history, "messages"):
        history.messages = _trim_history(history.messages, max_pairs=4)
    return history
