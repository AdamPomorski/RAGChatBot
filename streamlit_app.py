import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Initialize memory store
store = {}

# Create conversation memory
window_memory = ConversationBufferWindowMemory(k=2)


contextualize_q_system_prompt = """Mając historię rozmowy oraz najnowsze pytanie użytkownika, \
które może odnosić się do kontekstu w historii rozmowy, sformułuj samodzielne pytanie, które będzie zrozumiałe bez historii rozmowy. \
NIE odpowiadaj na pytanie, po prostu je sformułuj w razie potrzeby i w przeciwnym razie zwróć je takie, jakie jest."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = docsearch.as_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """Jesteś asystentem do zadań związanych z odpowiadaniem na pytania. \
Wykorzystaj następujące fragmenty odzyskanego kontekstu, aby odpowiedzieć na pytanie. \
Jeśli nie znasz odpowiedzi, po prostu powiedz, że nie wiesz. Odpowiedz maksymalnie trzema zdaniami i zachowaj odpowiedź zwięzłą.\"

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



# Function to interact with the chat bot
def chat_with_bot(user_input):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )
    return response["answer"]

# Streamlit UI
st.title("Chat with the Bot")
user_input = st.text_input("You:")
if st.button("Send"):
    bot_response = chat_with_bot(user_input)
    st.write("Bot:", bot_response)
    # Add user input to chat history
    chat_history = get_session_history("abc123")
