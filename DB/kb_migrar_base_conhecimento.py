# DB/kb_migrar_base_conhecimento.py
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import wave
import contextlib

from kb_mongo_config import get_kb_collection

# Caminho da sua pasta de base de conhecimento (ajusta se mudar de lugar)
BASE_DIR = Path(r"C:\Users\Gabriel\Desktop\Base de conhecimento")

# Só vamos importar arquivos .wav
ALLOWED_EXTENSIONS = {".wav"}


def listar_arquivos_base(base_dir: Path) -> List[Path]:
    """
    Retorna uma lista de arquivos .wav dentro da base de conhecimento.
    Percorre recursivamente todas as subpastas.
    """
    arquivos: List[Path] = []
    if not base_dir.exists():
        raise FileNotFoundError(f"Pasta base não encontrada: {base_dir}")

    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            arquivos.append(path)

    return arquivos


def extrair_info_wav(path: Path) -> Dict[str, Any]:
    """
    Lê metadados básicos de um arquivo .wav:
    canais, sample rate, frames, duração.
    """
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()  # bytes por amostra
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

    duracao = float(n_frames) / float(sample_rate) if sample_rate else 0.0

    return {
        "n_channels": n_channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "n_frames": n_frames,
        "duracao_seg": duracao,
    }


def arquivo_para_documento(path: Path, base_dir: Path) -> Dict[str, Any]:
    """
    Converte um arquivo .wav em um documento para o MongoDB.

    - classe: primeira pasta depois da base (ex: Drone, unknown)
    - arquivo: nome do arquivo
    - caminho_relativo: caminho relativo à pasta base
    - caminho_absoluto: caminho completo no disco
    - + metadados do .wav
    """
    rel_path = path.relative_to(base_dir)

    # classe = primeira pasta dentro da base (Drone, unknown, etc.)
    if len(rel_path.parts) > 1:
        classe = rel_path.parts[0]
    else:
        classe = "sem_classe"

    info_wav = extrair_info_wav(path)
    agora = datetime.utcnow()

    doc: Dict[str, Any] = {
        "arquivo": path.name,
        "classe": classe,
        "caminho_relativo": str(rel_path),
        "caminho_absoluto": str(path),
        "created_at": agora,
        "updated_at": agora,
        "origem": "dataset_audio_base_conhecimento",
    }

    # junta metadados do wav
    doc.update(info_wav)

    return doc


def migrar_base():
    """
    Lê todos os .wav da pasta Base de conhecimento e faz upsert na coleção.
    Upsert por 'caminho_absoluto' para não duplicar.
    """
    collection = get_kb_collection()

    print(f"📂 Lendo arquivos em: {BASE_DIR}")
    arquivos = listar_arquivos_base(BASE_DIR)
    print(f"Encontrados {len(arquivos)} arquivo(s) .wav.")

    if not arquivos:
        print("Nenhum arquivo com extensões:", ALLOWED_EXTENSIONS)
        return

    inseridos = 0
    atualizados = 0

    for path in arquivos:
        try:
            doc = arquivo_para_documento(path, BASE_DIR)
        except Exception as e:
            print(f"⚠️ Erro ao processar {path}: {e}")
            continue

        result = collection.update_one(
            {"caminho_absoluto": doc["caminho_absoluto"]},
            {"$set": doc},
            upsert=True,
        )

        if result.upserted_id is not None:
            inseridos += 1
        elif result.modified_count > 0:
            atualizados += 1

        print(f"OK -> {doc['caminho_relativo']}")

    print("======== RESUMO ========")
    print(f"Documentos inseridos:   {inseridos}")
    print(f"Documentos atualizados: {atualizados}")
    print("========================")


if __name__ == "__main__":
    migrar_base()
