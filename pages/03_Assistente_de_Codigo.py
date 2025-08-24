# pages/03_Assistente_de_Codigo.py
"""
# ============================================================================
# ORÁCULO - PÁGINA DO ASSISTENTE DE CÓDIGO
# ============================================================================
"""

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import time

# Importações dos modelos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ============================================================================
# PROMPT DE SISTEMA (O "PROMPT PODEROSO")
# ============================================================================
SYSTEM_PROMPT = """Você é "Oráculo Coder", um assistente de programação de elite e especialista sênior em Python. Sua missão é fornecer assistência de programação da mais alta qualidade.

**Suas diretrizes são:**

1.  **Priorize a Qualidade e as Boas Práticas:** Sempre gere código limpo, eficiente, bem documentado e que siga as convenções da PEP 8.
2.  **Explique o Raciocínio:** Nunca forneça apenas o código. Sempre explique o "porquê" da sua solução, detalhando a lógica, as escolhas de arquitetura e as alternativas que considerou.
3.  **Seja um Depurador Meticuloso:** Ao analisar um código com erro, identifique a causa raiz exata do problema, explique por que o erro acontece e forneça o código corrigido com as alterações claramente destacadas.
4.  **Pense em Otimização:** Se um código funciona mas pode ser melhorado, sugira otimizações de performance, legibilidade ou segurança.
5.  **Formatação Impecável:** Sempre formate os blocos de código usando blocos de markdown com a linguagem especificada (ex: ```python ... ```) para garantir o syntax highlighting.
6.  **Interaja como um Mentor:** Adote um tom colaborativo e didático. Seu objetivo é não apenas resolver o problema, mas também ensinar o usuário a ser um programador melhor.
"""

# ============================================================================
# REGISTRO CENTRAL DE MODELOS (COM ANTHROPIC)
# ============================================================================
MODEL_REGISTRY = {
    "Anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "class": ChatAnthropic,
        "models": {
            "Claude 3.7 Soneto": {"id": "claude-3-7-sonnet-latest"},
            "Claude Haiku 3.5": {"id": "claude-3-5-haiku-latest"},
        }
    },
    "OpenAI": {
        "api_key_env": "OPENAI_API_KEY",
        "class": ChatOpenAI,
        "models": { "GPT-4o": {"id": "gpt-4o"}, "GPT-4o Mini": {"id": "gpt-4o-mini"} }
    },
    "Google Gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "class": ChatGoogleGenerativeAI,
        "models": { "Gemini 1.5 Pro": {"id": "gemini-1.5-pro"}, "Gemini 1.5 Flash": {"id": "gemini-1.5-flash"} }
    },
    "Hugging Face": {
        "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
        "class": HuggingFaceEndpoint,
        "models": { "Qwen3 Coder 35B": {"id": "Qwen/Qwen3-Coder-480B-A35B-Instruct"} }
    },
    "Groq": {
        "api_key_env": "GROQ_API_KEY",
        "class": ChatGroq,
        "models": { "Llama3 70B (Groq)": {"id": "llama3-70b-8192"} }
    },
    "Deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "class": ChatDeepSeek,
        "models": { "DeepSeek Coder": {"id": "deepseek-coder"}, "DeepSeek Reasoner": {"id": "deepseek-reasoner"} }
    },
}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================
def get_file_content(uploaded_file):
    """Lê o conteúdo de um arquivo enviado."""
    if uploaded_file is None:
        return ""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return ""

# ============================================================================
# CLASSE DA APLICAÇÃO
# ============================================================================
class AssistenteCodigoApp:
    def __init__(self):
        """Inicializa o estado da sessão."""
        if 'chat_history_coder' not in st.session_state:
            st.session_state.chat_history_coder = []
        if 'chat_model_coder' not in st.session_state:
            st.session_state.chat_model_coder = None
        if 'model_initialized_coder' not in st.session_state:
            st.session_state.model_initialized_coder = False

    def _initialize_chat_model(self, provider: str, model_name: str):
        """Inicializa o modelo de chat selecionado."""
        provider_info = MODEL_REGISTRY.get(provider)
        api_key = os.getenv(provider_info["api_key_env"])
        if not api_key:
            st.error(f"❌ Chave de API '{provider_info['api_key_env']}' não encontrada.")
            return

        try:
            ModelClass = provider_info["class"]
            model_id = provider_info["models"][model_name]["id"]
            
            if provider == "Hugging Face":
                llm = HuggingFaceEndpoint(huggingfacehub_api_token=api_key, repo_id=model_id)
                st.session_state.chat_model_coder = ChatHuggingFace(llm=llm)
            else:
                init_params = {
                    ChatOpenAI: {"openai_api_key": api_key, "model_name": model_id},
                    ChatGoogleGenerativeAI: {"google_api_key": api_key, "model": model_id},
                    ChatGroq: {"groq_api_key": api_key, "model_name": model_id},
                    ChatDeepSeek: {"api_key": api_key, "model_name": model_id},
                    ChatAnthropic: {"anthropic_api_key": api_key, "model_name": model_id}
                }
                st.session_state.chat_model_coder = ModelClass(**init_params.get(ModelClass, {}))

            st.session_state.model_initialized_coder = True
            st.success(f"Modelo '{model_name}' pronto para uso!")
        except Exception as e:
            st.error(f"❌ Falha ao inicializar o modelo: {e}")
            st.session_state.model_initialized_coder = False

    def _render_sidebar(self):
        """Renderiza a barra lateral."""
        with st.sidebar:
            st.markdown("# 🧠 Assistente de Código")
            provider = st.selectbox("Provedor de IA", list(MODEL_REGISTRY.keys()), key="provider_coder")
            model_name = st.selectbox("Modelo", list(MODEL_REGISTRY[provider]["models"].keys()), key="model_coder")
            
            if st.button("🚀 Iniciar Assistente", type="primary", use_container_width=True):
                self._initialize_chat_model(provider, model_name)
                st.session_state.chat_history_coder = []

            if st.session_state.model_initialized_coder and st.session_state.chat_history_coder:
                st.divider()
                st.markdown("### ⚙️ Ações")

                if st.button("🧹 Limpar Histórico", use_container_width=True):
                    st.session_state.chat_history_coder = []
                    st.rerun()

                st.markdown("### 📥 Download do Chat")
                
                txt_history = ""
                for msg in st.session_state.chat_history_coder:
                    if isinstance(msg, SystemMessage): continue
                    role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
                    txt_history += f"--- {role} ---\n{msg.content}\n\n"
                
                st.download_button(
                    label="Baixar como Texto (.txt)",
                    data=txt_history.encode('utf-8'),
                    file_name=f"chat_coder_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                md_history = ""
                for msg in st.session_state.chat_history_coder:
                    if isinstance(msg, SystemMessage): continue
                    role = "**Usuário**" if isinstance(msg, HumanMessage) else "**Assistente**"
                    md_history += f"### {role}\n{msg.content}\n\n---\n\n"

                st.download_button(
                    label="Baixar como Markdown (.md)",
                    data=md_history.encode('utf-8'),
                    file_name=f"chat_coder_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    def _render_main_interface(self):
        """Renderiza a interface principal."""
        st.title("👨‍💻 Oráculo Coder")
        st.caption("Seu especialista particular para analisar, depurar e otimizar códigos.")

        if not st.session_state.model_initialized_coder:
            st.info("👈 Selecione um modelo e inicie o assistente na barra lateral para começar.")
            return
        
        with st.expander("➕ Adicionar Contexto de Código (Opcional)"):
            uploaded_file = st.file_uploader(
                "Envie um arquivo (.py, .pdf, .md, .txt)",
                type=["py", "pdf", "md", "txt"]
            )
            st.markdown("<p style='text-align: center; color: grey;'>OU</p>", unsafe_allow_html=True)
            pasted_code = st.text_area("Cole seu código aqui", height=300)

        for msg in st.session_state.chat_history_coder:
            if not isinstance(msg, SystemMessage):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(msg.content)
        
        if prompt := st.chat_input("Faça sua pergunta sobre o código..."):
            context_code = get_file_content(uploaded_file) if uploaded_file else pasted_code
            self._process_user_query(prompt, context_code)

    def _process_user_query(self, query: str, context: str):
        """Processa a consulta do usuário."""
        if not st.session_state.chat_history_coder:
            st.session_state.chat_history_coder.append(SystemMessage(content=SYSTEM_PROMPT))
        
        full_prompt = query
        if context:
            full_prompt = f"Com base no código abaixo, responda à seguinte pergunta.\n\n**Pergunta:** {query}\n\n--- CÓDIGO ---\n```python\n{context}\n```"
        
        st.session_state.chat_history_coder.append(HumanMessage(content=query))
        
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    messages_for_api = st.session_state.chat_history_coder.copy()
                    messages_for_api[-1] = HumanMessage(content=full_prompt)

                    response = st.session_state.chat_model_coder.invoke(messages_for_api)
                    answer = response.content
                    st.markdown(answer)
                    st.session_state.chat_history_coder.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"❌ Erro ao processar sua pergunta: {e}")

    def run(self):
        """Executa a aplicação."""
        self._render_sidebar()
        self._render_main_interface()

if __name__ == "__main__":
    st.set_page_config(page_title="Assistente de Código", page_icon="👨‍💻", layout="wide")
    app = AssistenteCodigoApp()
    app.run()
