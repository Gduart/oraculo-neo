# pages/02_Consulta_Livre.py
"""
# ============================================================================
# ORÁCULO - PÁGINA DE CONSULTA LIVRE
# ============================================================================
"""

import os
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# --- CORREÇÃO DEFINITIVA ---
# Importamos as duas classes necessárias para o Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

# ============================================================================
# REGISTRO CENTRAL DE MODELOS
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

# ============================================================================
# CLASSE DA APLICAÇÃO DE CONSULTA LIVRE
# ============================================================================
class ConsultaLivreApp:
    def __init__(self):
        """Inicializa o estado da sessão para a página de consulta livre."""
        if 'chat_history_livre' not in st.session_state:
            st.session_state.chat_history_livre = []
        if 'chat_model_livre' not in st.session_state:
            st.session_state.chat_model_livre = None
        if 'model_initialized_livre' not in st.session_state:
            st.session_state.model_initialized_livre = False

    def _initialize_chat_model(self, provider: str, model_name: str):
        """Inicializa o modelo de chat selecionado pelo usuário."""
        provider_info = MODEL_REGISTRY.get(provider)
        if not provider_info:
            st.error(f"Provedor '{provider}' não encontrado.")
            return
       
        api_key = os.getenv(provider_info["api_key_env"])
        if not api_key:
            st.error(f"❌ Chave de API '{provider_info['api_key_env']}' não encontrada.")
            return
       
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
                st.session_state.chat_model_livre = ChatHuggingFace(llm=llm)
            else:
                # Lógica padrão para todos os outros provedores
                init_params = {
                    ChatOpenAI: {"openai_api_key": api_key, "model_name": model_id},
                    ChatGoogleGenerativeAI: {"google_api_key": api_key, "model": model_id},
                    ChatGroq: {"groq_api_key": api_key, "model_name": model_id},
                    ChatDeepSeek: {"api_key": api_key, "model_name": model_id},
                    ChatZhipuAI: {"zhipuai_api_key": api_key, "model_name": model_id},
                }
                params = init_params.get(ModelClass, {})
                st.session_state.chat_model_livre = ModelClass(**params)

            st.session_state.model_initialized_livre = True
            st.success(f"Modelo '{model_name}' inicializado com sucesso!")

        except Exception as e:
            st.error(f"❌ Falha ao inicializar o modelo '{model_name}': {e}")
            st.session_state.model_initialized_livre = False

    def _render_sidebar(self):
        """Renderiza a barra lateral com as opções de modelo e download."""
        with st.sidebar:
            st.markdown("# 🧠 Seleção de Modelo")
            
            provider = st.selectbox("Provedor de IA", list(MODEL_REGISTRY.keys()), key="provider_livre")
            model_name = st.selectbox("Modelo", list(MODEL_REGISTRY[provider]["models"].keys()), key="model_livre")
            
            st.divider()

            if st.button("🚀 Iniciar Chat", type="primary", use_container_width=True):
                with st.spinner("Inicializando modelo..."):
                    self._initialize_chat_model(provider, model_name)
                    st.session_state.chat_history_livre = []

            if st.session_state.model_initialized_livre and st.session_state.chat_history_livre:
                st.divider()
                st.markdown("### 📥 Download do Chat")

                txt_history = ""
                for msg in st.session_state.chat_history_livre:
                    role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
                    txt_history += f"--- {role} ---\n{msg.content}\n\n"
               
                st.download_button(
                    label="Baixar como Texto (.txt)",
                    data=txt_history.encode('utf-8'),
                    file_name=f"chat_livre_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                md_history = ""
                for msg in st.session_state.chat_history_livre:
                    role = "**Usuário**" if isinstance(msg, HumanMessage) else "**Assistente**"
                    md_history += f"### {role}\n{msg.content}\n\n---\n\n"

                st.download_button(
                    label="Baixar como Markdown (.md)",
                    data=md_history.encode('utf-8'),
                    file_name=f"chat_livre_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    def _render_chat_interface(self):
        """Renderiza a interface principal do chat."""
        st.title("💬 Oráculo - Consulta Livre")
        st.caption("Converse diretamente com os modelos de Inteligência Artificial.")

        if not st.session_state.model_initialized_livre:
            st.info("👈 Selecione um provedor, um modelo e inicie o chat na barra lateral.")
            return

        for msg in st.session_state.chat_history_livre:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)
       
        if prompt := st.chat_input("Digite sua mensagem..."):
            st.session_state.chat_history_livre.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                self._process_user_query(prompt)

    def _process_user_query(self, query: str):
        """Envia a consulta para o modelo de LLM e exibe a resposta."""
        with st.spinner("Pensando..."):
            try:
                response = st.session_state.chat_model_livre.invoke(st.session_state.chat_history_livre)
                answer = response.content

                st.markdown(answer)
                st.session_state.chat_history_livre.append(AIMessage(content=answer))

            except Exception as e:
                st.error(f"❌ Erro ao processar sua pergunta: {e}")

    def run(self):
        """Executa a aplicação."""
        self._render_sidebar()
        self._render_chat_interface()

if __name__ == "__main__":
    st.set_page_config(page_title="Consulta Livre", page_icon="💬", layout="wide")
    app = ConsultaLivreApp()
    app.run()
