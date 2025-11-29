# DB/kb_app_kb.py
import os
import streamlit as st
from kb_mongo_config import get_kb_collection

st.set_page_config(
    page_title="Dataset de Áudio - Drones",
    layout="wide",
)

st.title("Dataset de Áudio - Drones")
st.write("Arquivos .wav organizados no MongoDB Atlas (metadados) + player local.")

collection = get_kb_collection()

# ===== RESUMO POR CLASSE =====
st.subheader("Resumo por classe")

pipeline = [
    {"$group": {"_id": "$classe", "quantidade": {"$sum": 1}}},
    {"$sort": {"_id": 1}},
]
stats = list(collection.aggregate(pipeline))

if stats:
    tabela = [{"classe": s["_id"], "quantidade": s["quantidade"]} for s in stats]
    st.table(tabela)
else:
    st.info("Nenhum documento encontrado na coleção ainda. Rode primeiro o script de migração.")
    st.stop()

st.markdown("---")

# ===== FILTROS =====
st.subheader("Filtrar e explorar áudios")

classes = sorted([c for c in collection.distinct("classe") if c])

col1, col2 = st.columns(2)

with col1:
    classe_sel = st.selectbox(
        "Classe",
        options=["Todas"] + classes,
        index=0,
    )

with col2:
    busca_nome = st.text_input(
        "Buscar por nome de arquivo (.wav):",
        placeholder="ex: drone_001, sample10..."
    )

query = {}

if classe_sel != "Todas":
    query["classe"] = classe_sel

if busca_nome:
    query["arquivo"] = {"$regex": busca_nome, "$options": "i"}

docs = list(collection.find(query).sort("arquivo", 1).limit(300))

st.write(f"Resultados encontrados: **{len(docs)}** (limitado a 300).")
st.markdown("---")

# ===== LISTAGEM COM PLAYER =====
for doc in docs:
    arquivo = doc.get("arquivo", "sem_nome.wav")
    classe = doc.get("classe", "N/A")
    dur = doc.get("duracao_seg", 0.0)
    sr = doc.get("sample_rate", None)
    ch = doc.get("n_channels", None)
    caminho_rel = doc.get("caminho_relativo", "—")
    caminho_abs = doc.get("caminho_absoluto", "")

    header = f"[{classe}] {arquivo}  —  {dur:.2f} s"
    with st.expander(header):
        st.markdown(f"**Classe:** `{classe}`")
        st.markdown(f"**Nome do arquivo:** `{arquivo}`")
        st.markdown(f"**Duração:** `{dur:.2f} s`")
        if sr is not None:
            st.markdown(f"**Sample rate:** `{sr} Hz`")
        if ch is not None:
            st.markdown(f"**Canais:** `{ch}`")
        st.markdown(f"**Caminho relativo:** `{caminho_rel}`")
        st.markdown(f"**Caminho absoluto:** `{caminho_abs}`")

        st.markdown("---")
        st.markdown("### Player de áudio")

        if caminho_abs and os.path.exists(caminho_abs):
            try:
                with open(caminho_abs, "rb") as f:
                    data = f.read()
                st.audio(data, format="audio/wav")
            except Exception as e:
                st.warning(f"Não foi possível carregar o áudio: {e}")
        else:
            st.warning("Arquivo .wav não encontrado nesse caminho local.")

st.markdown("---")
st.caption("MongoDB Atlas + Streamlit • Dataset de áudio para detecção de drones")
