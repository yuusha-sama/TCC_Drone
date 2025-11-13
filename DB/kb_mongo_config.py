# DB/kb_mongo_config.py
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# === CONFIGURAÇÃO DO MONGO ATLAS ===
# ⚠️ Cuidado para não commitar essa senha no GitHub.
MONGO_URI = (
    "mongodb+srv://Drone_db:drones@cluster0.zkrzaft.mongodb.net/"
    "?appName=Cluster0"
)

DB_NAME = "drone_kb"              # nome do banco da base de conhecimento
COLLECTION_NAME = "conhecimento"  # coleção onde vão ficar os textos


def get_mongo_client() -> MongoClient:
    """
    Retorna um cliente conectado ao MongoDB Atlas.
    """
    client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
    return client


def get_kb_collection():
    """
    Retorna a coleção da base de conhecimento.
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection


if __name__ == "__main__":
    # Teste rápido de conexão
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        print("✅ Conectado com sucesso ao MongoDB Atlas!")
    except Exception as e:
        print("❌ Erro ao conectar no MongoDB:", e)
