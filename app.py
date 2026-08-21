import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de la página
st.set_page_config(
    page_title="RAG Académico - Asistente de Investigación",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Asistente RAG para Documentos Académicos")
st.write("Carga un documento PDF y realiza preguntas sobre su contenido utilizando Inteligencia Artificial.")

# 2. Barra lateral para credenciales y configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    openai_api_key = st.text_input("Ingresa tu OpenAI API Key:", type="password")
    
    st.markdown("---")
    st.markdown("### 🛠️ Tecnologías Utilizadas")
    st.markdown("- **Streamlit**: Interfaz web")
    st.markdown("- **LangChain**: Pipeline RAG")
    st.markdown("- **ChromaDB**: Base de datos vectorial")
    st.markdown("- **OpenAI**: Embeddings y LLM")

# 3. Estado de la sesión (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# 4. Carga y procesamiento del archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF académico", type=["pdf"])

if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if st.button("🚀 Procesar Documento"):
        with st.spinner("Procesando PDF, extrayendo texto y creando embeddings..."):
            # Guardar el PDF temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Cargar documento
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # Splitter: Dividir el texto en fragmentos (chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Embeddings y Base Vectorial (ChromaDB)
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            # Configurar Retriever y Modelo
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            # Prompt del Sistema
            system_prompt = (
                "Eres un asistente de investigación académica. Usa los siguientes "
                "fragmentos de contexto recuperados para responder a la pregunta. "
                "Si no sabes la respuesta, responde honestamente que no dispones de dicha información. "
                "Proporciona respuestas claras, formales y estructuradas.\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Cadena de RAG
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Limpiar archivo temporal
            os.remove(tmp_file_path)

            st.success("¡Documento indexado con éxito! Ya puedes hacer preguntas en el chat.")

elif uploaded_file and not openai_api_key:
    st.warning("⚠️ Por favor, ingresa tu OpenAI API Key en la barra lateral para continuar.")

# 5. Interfaz de Chat
st.markdown("---")
st.subheader("💬 Chat de Consultas")

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de la pregunta del usuario
if user_input := st.chat_input("Haz una pregunta sobre el PDF cargado..."):
    if not st.session_state.retrieval_chain:
        st.error("❌ Por favor, sube y procesa un archivo PDF primero.")
    else:
        # Guardar y mostrar el mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generar respuesta con la cadena RAG
        with st.chat_message("assistant"):
            with st.spinner("Buscando respuestas en el documento..."):
                response = st.session_state.retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
