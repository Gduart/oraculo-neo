# oraculo_optimized_complete.py
"""
# ============================================================================
# ORÁCULO - VERSÃO PROFISSIONAL REESTRUTURADA
# ============================================================================
"""
import os
import time
import tempfile
from typing import Tuple, Any, Dict
import asyncio
import re
import sys
import yt_dlp
import subprocess
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# --- CORREÇÃO DEFINITIVA ---
# Importamos as duas classes necessárias para o Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import FAISS

nest_asyncio.apply()
load_dotenv()

# ============================================================================
# CONFIGURAÇÃO CENTRAL DE MODELOS
# ============================================================================
MODEL_REGISTRY = {
    "OpenAI": {
        "api_key_env": "OPENAI_API_KEY",
        "class": ChatOpenAI,
        "models": {
            "GPT-4o": {"id": "gpt-4o"},
            "GPT-4o Mini": {"id": "gpt-4o-mini"},
            "GpT5": {"id": "gpt-5-2025-08-07"},
            "GpT5-Mini": {"id": "gpt-5-mini-2025-08-07"},
            "GpT5-Nano": {"id": "gpt-5-nano"},
            "Gpt-4.1": {"id": "gpt-4.1-2025-04-14"},
            "o4-mini-deep-research": {"id": "o4-mini-deep-research-2025-06-26"},
        }
    },
    "Google Gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "class": ChatGoogleGenerativeAI,
        "models": {
            "Gemini 1.5 Pro": {"id": "gemini-1.5-pro"},
            "Gemini 1.5 Flash": {"id": "gemini-1.5-flash"},
        }
    },
    "Hugging Face": {
        "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
        "class": HuggingFaceEndpoint, # Mantemos a classe base para identificação
        "models": {
            "Mistral 7B Instruct": {"id": "mistralai/Mistral-7B-Instruct-v0.3"},
            "Zephyr 7B": {"id": "HuggingFaceH4/zephyr-7b-beta"},
            "Qwen3 Coder 35B": {"id": "Qwen/Qwen3-Coder-480B-A35B-Instruct"},
            "Lhama 3.1": {"id": "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.1-Uncensored"},
            "Llama 3 70B Instruct": {"id": "meta-llama/Meta-Llama-3-70B-Instruct"},
        }
    },
    "Groq": {
        "api_key_env": "GROQ_API_TOKEN",
        "class": ChatGroq,
        "models": {
            "Llama3 70B (Groq)": {"id": "llama3-70b-8192"},
        }
    },
    "Deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "class": ChatDeepSeek,
        "models": {
            "DeepSeek Chat": {"id": "deepseek-chat"},
            "deepseek-reasoner": {"id": "deepseek-reasoner"},
        }
    },
    "ZhipuAI (GLM)": {
        "api_key_env": "GLMKEY",
        "class": ChatZhipuAI,
        "models": {
            "GLM 4.5": { "id": "glm-4.5" },
            "GLM 4.5 Flash": { "id": "glm-4.5-flash" }
        }
    }
}

# ... (O restante das classes DocumentLoader e RAGProcessor permanecem idênticas) ...
class DocumentLoader:
    """Classe robusta para carregar conteúdo de diversas fontes."""

    def _handle_error(self, msg: str) -> Tuple[str, bool]:
        st.error(msg)
        return "", False

    def load_pdf(self, file_upload: Any) -> Tuple[str, bool]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_upload.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
            os.remove(tmp.name)
            if not docs: return self._handle_error( "❌ PDF vazio ou ilegível." )
            content = "\n\n".join(doc.page_content for doc in docs)
            return content, True
        except Exception as e:
            return self._handle_error(f"❌ Erro crítico ao carregar PDF: {e}")

    def load_youtube(self, url: str) -> Tuple[str, bool]:
        try:
            st.info("📹 Etapa 1: Sondando legendas disponíveis...")
            ydl_opts_list = {'listsubtitles': True, 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
                info = ydl.extract_info(url, download=False)
                video_id = info['id']
                available_subs = info.get('subtitles', {})
                available_auto_subs = info.get('automatic_captions', {})

            if not available_subs and not available_auto_subs:
                return self._handle_error( "❌ Nenhuma legenda (manual ou automática) encontrada para este vídeo." )

            lang_to_download = next((lang for lang in ['pt', 'en'] if lang in available_subs), None) or \
                               next((lang for lang in ['pt', 'en'] if lang in available_auto_subs), None)
           
            if not lang_to_download:
                return self._handle_error( "❌ Nenhuma legenda em Português ou Inglês foi encontrada." )

            st.info(f"📹 Etapa 2: Baixando legenda selecionada ('{lang_to_download}')...")
            ydl_opts_download = {
                'writesubtitles': lang_to_download in available_subs,
                'writeautomaticsub': lang_to_download in available_auto_subs,
                'subtitleslangs': [lang_to_download],
                'skip_download': True,
                'outtmpl': f'{video_id}',
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
                ydl.download([url])

            subtitle_file = f"{video_id}.{lang_to_download}.vtt"
            if not os.path.exists(subtitle_file):
                return self._handle_error(f"❌ Falha ao salvar o arquivo de legenda '{subtitle_file}'.")

            with open(subtitle_file, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            os.remove(subtitle_file)

            lines = vtt_content.splitlines()
            text_lines = [line for line in lines if line and '-->' not in line and 'WEBVTT' not in line and not re.match(r'^\d+$', line) and not line.startswith('Kind:') and not line.startswith('Language:')]
            content = " ".join(text_lines)

            if not content.strip():
                return self._handle_error( "❌ Transcrição encontrada, mas a extração do texto falhou." )
           
            st.success( "✅ Transcrição do YouTube obtida com sucesso!" )
            return content, True
       
        except Exception as e:
            return self._handle_error(f"❌ Falha crítica ao processar transcrição do YouTube. Erro: {e}")

    def load_csv(self, file_upload: Any) -> Tuple[str, bool]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', encoding='utf-8') as tmp:
                tmp.write(file_upload.getvalue().decode('utf-8'))
                loader = CSVLoader(tmp.name)
                docs = loader.load()
            os.remove(tmp.name)
            if not docs: return self._handle_error( "❌ CSV vazio ou ilegível." )
            content = "\n\n".join(doc.page_content for doc in docs)
            return content, True
        except Exception as e:
            return self._handle_error(f"❌ Erro ao carregar CSV: {e}")
           
    def load_txt(self, file_upload: Any) -> Tuple[str, bool]:
        try:
            content = file_upload.getvalue().decode("utf-8")
            if not content.strip(): return self._handle_error( "❌ Arquivo TXT está vazio." )
            return content, True
        except Exception as e:
            return self._handle_error(f"❌ Erro ao carregar TXT: {e}")

    def load_website(self, url: str) -> Tuple[str, bool]:
        try:
            st.info("🤖 Robô de scraping em ação... Navegando e analisando o site. Isso pode levar um momento.")
           
            scraper_path = "scraper.py"
            if not os.path.exists(scraper_path):
                return self._handle_error(f"❌ Erro crítico: O arquivo '{scraper_path}' não foi encontrado.")

            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt', encoding='utf-8') as tmp_file:
                output_filename = tmp_file.name
           
            subprocess.run(
                [sys.executable, scraper_path, url, output_filename],
                capture_output=True, text=True, encoding='utf-8'
            )

            with open(output_filename, 'r', encoding='utf-8') as f:
                content = f.read()
           
            os.remove(output_filename)

            if "SCRAPER_ERROR" in content:
                return self._handle_error(f"❌ O robô de scraping falhou: {content.replace('SCRAPER_ERROR:', '')}")
           
            if not content or not content.strip():
                return self._handle_error( "❌ O robô não encontrou conteúdo de texto relevante no site." )
           
            st.success( "✅ Conteúdo do site extraído com sucesso pelo robô!" )
            return content, True
           
        except Exception as e:
            return self._handle_error(f"❌ Falha crítica ao processar o site. Erro: {e}")

class RAGProcessor:
    """Gerencia a criação do sistema RAG (Embeddings, Vectorstore, Retriever)."""
    def __init__(self):
        self.embedding_model = None

    def _initialize_embeddings(self) -> bool:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error( "❌ Chave GOOGLE_API_KEY não encontrada para os embeddings." )
            return False
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        return True

    def create_rag_system(self, doc_content: str) -> Any:
        if not self._initialize_embeddings(): return None
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            chunks = text_splitter.split_text(doc_content)
            if not chunks:
                st.error( "❌ Falha ao dividir o documento. O conteúdo pode ser muito curto." )
                return None
           
            vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embedding_model)
            return vectorstore.as_retriever(search_kwargs={"k": 7})
        except Exception as e:
            st.error(f"❌ Erro crítico ao criar o sistema RAG: {e}")
            return None

class OraculoApp:
    def __init__(self):
        self.loader = DocumentLoader()
        self.rag_processor = RAGProcessor()
        self._initialize_session_state()

    def _initialize_session_state(self):
        defaults = {'system_ready': False, 'chat_history': []}
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _initialize_chat_model(self, provider: str, model_name: str) -> Any:
        provider_info = MODEL_REGISTRY.get(provider)
        if not provider_info:
            st.error(f"Provedor '{provider}' não encontrado no registro.")
            return None
       
        api_key = os.getenv(provider_info["api_key_env"])
        if not api_key:
            st.error(f"❌ Chave de API '{provider_info['api_key_env']}' não encontrada no ambiente.")
            return None
       
        try:
            ModelClass = provider_info["class"]
            model_id = provider_info["models"][model_name]["id"]

            # --- CORREÇÃO DEFINITIVA ---
            # Lógica especial para o Hugging Face para garantir que ele funcione como um ChatModel
            if provider == "Hugging Face":
                llm = HuggingFaceEndpoint(
                    huggingfacehub_api_token=api_key,
                    repo_id=model_id,
                    temperature=0.7,
                    max_new_tokens=1024
                )
                return ChatHuggingFace(llm=llm)
            
            # Lógica padrão para todos os outros provedores
            init_params = {
                ChatOpenAI: {"openai_api_key": api_key, "model_name": model_id},
                ChatGoogleGenerativeAI: {"google_api_key": api_key, "model": model_id},
                ChatGroq: {"groq_api_key": api_key, "model_name": model_id},
                ChatDeepSeek: {"api_key": api_key, "model_name": model_id},
                ChatZhipuAI: {"zhipuai_api_key": api_key, "model_name": model_id},
            }
            params = init_params.get(ModelClass, {})
            return ModelClass(**params)

        except Exception as e:
            st.error(f"❌ Falha ao inicializar o modelo '{model_name}': {e}")
            return None

    def _run_initialization_process(self, source_type, source_data, provider, model_name):
        st.session_state.system_ready = False
        st.session_state.chat_history = []
       
        try:
            with st.status("Oráculo está se preparando...", expanded=True) as status:
                status.update(label="**Etapa 1/3:** Carregando e processando documento...")
                load_func = getattr(self.loader, f"load_{source_type.lower()}", None)
                if not load_func:
                    status.update(label="Falha na Etapa 1", state="error", expanded=True)
                    st.error(f"Tipo de fonte '{source_type}' não implementado.")
                    return
               
                doc_content, success = load_func(source_data)
                if not success:
                    status.update(label="Falha no Carregamento do Documento", state="error", expanded=True)
                    return
                status.update(label="**Etapa 1/3:** Documento processado com sucesso!", state="complete")

                status.update(label="**Etapa 2/3:** Inicializando modelo de IA...")
                chat_model = self._initialize_chat_model(provider, model_name)
                if not chat_model:
                    status.update(label="Falha na Inicialização do Modelo", state="error", expanded=True)
                    return
                st.session_state.chat_model = chat_model
                status.update(label="**Etapa 2/3:** Modelo de IA pronto!", state="complete")

                status.update(label="**Etapa 3/3:** Criando memória e sistema de busca...")
                retriever = self.rag_processor.create_rag_system(doc_content)
                if not retriever:
                    status.update(label="Falha na Criação do Sistema RAG", state="error", expanded=True)
                    return
                st.session_state.retriever = retriever
                status.update(label="**Etapa 3/3:** Sistema de busca pronto!", state="complete")

            st.session_state.system_ready = True
            st.success("🎉 Oráculo pronto para uso!")
            st.balloons()
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error("❌ Ocorreu um erro crítico durante a inicialização.")
            st.exception(e)

    def _render_sidebar(self):
        with st.sidebar:
            st.markdown("# 🛠️ Configurações do Oráculo")
           
            source_type = st.selectbox("Tipo de fonte", ['PDF', 'Youtube', 'CSV', 'TXT', 'Website'])
           
            source_data = None
            if source_type in ['Youtube', 'Website']:
                source_data = st.text_input(f"URL do {source_type}", placeholder="https://...")
            else:
                source_data = st.file_uploader(f"Upload do arquivo {source_type}", type=[source_type.lower()])

            st.divider()
            provider = st.selectbox("Provedor de IA", list(MODEL_REGISTRY.keys()))
            model_name = st.selectbox("Modelo", list(MODEL_REGISTRY[provider]["models"].keys()))
            st.divider()

            if st.button("🚀 Inicializar Oráculo", type="primary", use_container_width=True, disabled=not source_data):
                self._run_initialization_process(source_type, source_data, provider, model_name)
           
            if st.session_state.get('system_ready') and st.session_state.get('chat_history'):
                st.divider()
                st.markdown("### 📥 Download da Análise")

                txt_history = ""
                for msg in st.session_state.chat_history:
                    role = "Usuário" if isinstance(msg, HumanMessage) else "Oráculo"
                    txt_history += f"--- {role} ---\n{msg.content}\n\n"
               
                st.download_button(
                    label="Baixar como Texto (.txt)",
                    data=txt_history.encode('utf-8'),
                    file_name=f"analise_oraculo_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                md_history = ""
                for msg in st.session_state.chat_history:
                    role = "**Usuário**" if isinstance(msg, HumanMessage) else "**Oráculo**"
                    md_history += f"### {role}\n{msg.content}\n\n---\n\n"

                st.download_button(
                    label="Baixar como Markdown (.md)",
                    data=md_history.encode('utf-8'),
                    file_name=f"analise_oraculo_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )


    def _render_chat_interface(self):
        st.title("🔮 Oráculo - Análise de Documentos")
        st.caption("Faça upload de um documento e extraia insights com IA.")


        if not st.session_state.get('system_ready'):
            st.info("👈 Configure e inicialize o Oráculo na barra lateral para começar.")
            return

        for msg in st.session_state.chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)
       
        if prompt := st.chat_input("Faça sua pergunta sobre o documento..."):
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                self._process_user_query(prompt)

    def _process_user_query(self, query: str):
        with st.spinner("🔮 Consultando o oráculo..."):
            try:
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Dado o histórico da conversa, reformule a última pergunta para ser uma pergunta independente."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(
                    st.session_state.chat_model, st.session_state.retriever, contextualize_q_prompt
                )
               
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Você é o Oráculo. Responda à pergunta do usuário em Português do Brasil com base no seguinte contexto:\n\n{context}"),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ])
                question_answer_chain = create_stuff_documents_chain(st.session_state.chat_model, qa_prompt)
               
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
               
                response = rag_chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.chat_history
                })
                answer = response.get('answer', "Desculpe, não encontrei uma resposta.")
               
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))

            except Exception as e:
                st.error(f"❌ Erro ao processar sua pergunta: {e}")

    def run(self):
        st.set_page_config(page_title="Análise de Documentos", page_icon="🔮", layout="wide")
        self._render_sidebar()
        self._render_chat_interface()

if __name__ == "__main__":
    app = OraculoApp()
    app.run()
