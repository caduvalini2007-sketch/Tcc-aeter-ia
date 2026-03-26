

import os
import re
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from pytubefix import YouTube
from moviepy.editor import AudioFileClip
import whisper

# ==========================================
# CONFIGURAÇÕES
# ==========================================

# Pastas do sistema
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("outputs")

for directory in (UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Extensões permitidas
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".mts", ".wmv", ".m4v", ".3gp"}
ALL_ALLOWED = ALLOWED_AUDIO | ALLOWED_VIDEO

# Limites
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_VIDEO_DURATION = 7200  # 2 horas em segundos

# Modelo Whisper (opções: tiny, base, small, medium, large)
MODEL_NAME = "base"  # Bom balanço entre velocidade e precisão

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# INICIALIZAÇÃO DO FLASK
# ==========================================

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_muito_segura_mude_em_producao'
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ==========================================
# CARREGAMENTO DO MODELO WHISPER
# ==========================================

logger.info(f"Carregando modelo Whisper '{MODEL_NAME}'...")
try:
    model = whisper.load_model(MODEL_NAME)
    logger.info(f"✓ Modelo '{MODEL_NAME}' carregado com sucesso!")
except Exception as e:
    logger.error(f"✗ Erro ao carregar modelo Whisper: {e}")
    raise


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def clean_old_files(directory: Path, max_age_hours: int = 24):
    """Remove arquivos antigos de um diretório."""
    try:
        now = datetime.now()
        removed_count = 0
        
        for filepath in directory.iterdir():
            if filepath.is_file():
                file_age = datetime.fromtimestamp(filepath.stat().st_mtime)
                if now - file_age > timedelta(hours=max_age_hours):
                    filepath.unlink()
                    removed_count += 1
                    logger.info(f"Arquivo antigo removido: {filepath.name}")
        
        if removed_count > 0:
            logger.info(f"Limpeza concluída: {removed_count} arquivo(s) removido(s)")
            
    except Exception as e:
        logger.error(f"Erro ao limpar arquivos antigos: {e}")


def is_youtube_url(url: str) -> bool:
    """Verifica se a string é uma URL do YouTube."""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def get_safe_filename(filename: str, uid: str = None) -> str:
    """Gera um nome de arquivo seguro."""
    if uid is None:
        uid = uuid.uuid4().hex[:8]
    
    safe_name = secure_filename(filename)
    name, ext = os.path.splitext(safe_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{timestamp}_{uid}_{name}{ext}"


def download_youtube_audio(url: str, destination: Path) -> Path:
    """
    Baixa o stream de áudio de um vídeo do YouTube.
    Retorna o caminho do arquivo baixado.
    """
    logger.info(f"Iniciando download do YouTube: {url}")
    
    try:
        # Cria objeto YouTube
        yt = YouTube(url)
        
        # Verifica duração
        duration = yt.length
        if duration > MAX_VIDEO_DURATION:
            raise ValueError(
                f"Vídeo muito longo ({duration}s). Máximo permitido: {MAX_VIDEO_DURATION}s"
            )
        
        logger.info(f"Título: {yt.title}")
        logger.info(f"Duração: {duration}s ({duration//60}min)")
        
        # Obtém stream de áudio
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        
        if audio_stream is None:
            raise RuntimeError("Nenhum stream de áudio disponível")
        
        logger.info(f"Stream selecionado: {audio_stream.abr} - {audio_stream.mime_type}")
        
        # Download
        output_path = audio_stream.download(
            output_path=str(destination.parent),
            filename=destination.name
        )
        
        logger.info(f"✓ Download concluído: {output_path}")
        return Path(output_path)
        
    except Exception as e:
        logger.error(f"✗ Erro no download do YouTube: {e}")
        raise


def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> Path:
    """
    Extrai a trilha de áudio de um arquivo de vídeo.
    Converte para WAV (melhor para Whisper).
    """
    logger.info(f"Extraindo áudio de: {video_path.name}")
    
    try:
        clip = AudioFileClip(str(video_path))
        
        # Informações do áudio
        duration = clip.duration
        logger.info(f"Duração do áudio: {duration:.2f}s ({duration/60:.1f}min)")
        
        if duration > MAX_VIDEO_DURATION:
            clip.close()
            raise ValueError(
                f"Áudio muito longo ({duration:.0f}s). Máximo: {MAX_VIDEO_DURATION}s"
            )
        
        # Salva como WAV (PCM 16-bit)
        clip.write_audiofile(
            str(output_audio_path),
            codec="pcm_s16le",
            fps=16000,  # 16kHz é suficiente para fala
            verbose=False,
            logger=None
        )
        
        clip.close()
        logger.info(f"✓ Áudio extraído: {output_audio_path.name}")
        return output_audio_path
        
    except Exception as e:
        logger.error(f"✗ Erro ao extrair áudio: {e}")
        raise


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None
) -> dict:
    """
    Transcreve um arquivo de áudio usando Whisper.
    Retorna dicionário com texto, idioma detectado, etc.
    """
    logger.info(f"Iniciando transcrição: {audio_path.name}")
    logger.info(f"Idioma especificado: {language or 'auto-detect'}")
    
    try:
        # Parâmetros do Whisper
        transcribe_options = {
            "fp16": False,  # Evita problemas em CPU
            "verbose": False
        }
        
        if language and language != "auto":
            transcribe_options["language"] = language
        
        # Executa transcrição
        result = model.transcribe(str(audio_path), **transcribe_options)
        
        # Extrai informações
        text = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        segments = result.get("segments", [])
        
        # Estatísticas
        word_count = len(text.split())
        char_count = len(text)
        segment_count = len(segments)
        
        logger.info(f"✓ Transcrição concluída!")
        logger.info(f"  - Idioma detectado: {detected_language}")
        logger.info(f"  - Palavras: {word_count}")
        logger.info(f"  - Caracteres: {char_count}")
        logger.info(f"  - Segmentos: {segment_count}")
        
        return {
            "text": text,
            "language": detected_language,
            "word_count": word_count,
            "char_count": char_count,
            "segments": segments
        }
        
    except Exception as e:
        logger.error(f"✗ Erro na transcrição: {e}")
        raise


def save_transcription(text: str, output_path: Path) -> Path:
    """Salva a transcrição em arquivo de texto."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write("=" * 70 + "\n")
            f.write("TRANSCRIÇÃO AUTOMÁTICA - Aether IA\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Modelo: Whisper {MODEL_NAME}\n")
            f.write("=" * 70 + "\n\n")
            
            # Texto
            f.write(text)
            
            # Rodapé
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("Gerado automaticamente por Aether IA\n")
            f.write("Whisper AI by OpenAI\n")
        
        logger.info(f"✓ Transcrição salva: {output_path.name}")
        return output_path
        
    except Exception as e:
        logger.error(f"✗ Erro ao salvar transcrição: {e}")
        raise


# ==========================================
# ROTAS
# ==========================================

@app.route("/")
def index():
    """Página principal."""
    return render_template("transcricao_video.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    """
    Endpoint principal de transcrição.
    
    Aceita:
        - form['url']: URL do YouTube
        - file['file']: Upload de arquivo
        - form['language']: Código do idioma (opcional)
    
    Retorna JSON com:
        - transcription: texto transcrito
        - download: link para download do .txt
        - metadata: informações adicionais
    """
    
    # Limpa arquivos antigos a cada requisição
    clean_old_files(TEMP_DIR, max_age_hours=1)
    clean_old_files(UPLOAD_DIR, max_age_hours=24)
    clean_old_files(OUTPUT_DIR, max_age_hours=48)
    
    # Obtém parâmetros
    youtube_url = request.form.get("url", "").strip()
    uploaded_file = request.files.get("file")
    language = request.form.get("language", "pt")
    
    if language == "auto":
        language = None
    
    # Validações
    if not youtube_url and not uploaded_file:
        return jsonify({
            "error": "Envie uma URL do YouTube ou faça upload de um arquivo."
        }), 400
    
    # Gera ID único para esta transcrição
    session_id = uuid.uuid4().hex[:12]
    
    # Arquivos temporários
    temp_video = TEMP_DIR / f"video_{session_id}"
    temp_audio = TEMP_DIR / f"audio_{session_id}.wav"
    output_txt = OUTPUT_DIR / f"transcription_{session_id}.txt"
    
    temp_files = []  # Para limpeza posterior
    source_type = None
    source_name = None
    
    try:
        # ====================================
        # PROCESSAMENTO: YOUTUBE
        # ====================================
        if youtube_url:
            if not is_youtube_url(youtube_url):
                return jsonify({
                    "error": "A URL fornecida não parece ser do YouTube."
                }), 400
            
            source_type = "youtube"
            source_name = youtube_url
            
            # Download do áudio
            temp_download = temp_video.with_suffix(".m4a")
            downloaded_file = download_youtube_audio(youtube_url, temp_download)
            temp_files.append(downloaded_file)
            
            # Extrai/converte para WAV
            extract_audio_from_video(downloaded_file, temp_audio)
            temp_files.append(temp_audio)
            
            audio_to_transcribe = temp_audio
        
        # ====================================
        # PROCESSAMENTO: UPLOAD
        # ====================================
        elif uploaded_file:
            source_type = "upload"
            source_name = uploaded_file.filename
            
            # Validação de nome
            if not source_name:
                return jsonify({"error": "Nome do arquivo inválido."}), 400
            
            # Salva upload
            safe_name = get_safe_filename(source_name, session_id)
            upload_path = UPLOAD_DIR / safe_name
            uploaded_file.save(upload_path)
            temp_files.append(upload_path)
            
            logger.info(f"Arquivo recebido: {source_name} ({upload_path.stat().st_size / 1024:.1f} KB)")
            
            # Determina tipo
            file_ext = upload_path.suffix.lower()
            
            if file_ext in ALLOWED_AUDIO:
                # Áudio direto - ainda converte para WAV para padronizar
                extract_audio_from_video(upload_path, temp_audio)
                temp_files.append(temp_audio)
                audio_to_transcribe = temp_audio
                
            elif file_ext in ALLOWED_VIDEO:
                # Vídeo - extrai áudio
                extract_audio_from_video(upload_path, temp_audio)
                temp_files.append(temp_audio)
                audio_to_transcribe = temp_audio
                
            else:
                # Tentativa de extração genérica
                try:
                    extract_audio_from_video(upload_path, temp_audio)
                    temp_files.append(temp_audio)
                    audio_to_transcribe = temp_audio
                except Exception:
                    return jsonify({
                        "error": f"Formato não suportado: {file_ext}. "
                                f"Use: {', '.join(sorted(ALL_ALLOWED))}"
                    }), 400
        
        # ====================================
        # TRANSCRIÇÃO
        # ====================================
        logger.info("=" * 70)
        logger.info(f"NOVA TRANSCRIÇÃO - Session: {session_id}")
        logger.info(f"Fonte: {source_type} - {source_name}")
        logger.info("=" * 70)
        
        transcription_result = transcribe_audio(audio_to_transcribe, language)
        
        text = transcription_result["text"]
        
        if not text:
            return jsonify({
                "error": "Nenhum texto foi detectado no áudio. "
                        "Verifique se o arquivo contém fala inteligível."
            }), 400
        
        # Salva resultado
        save_transcription(text, output_txt)
        
        # Resposta
        return jsonify({
            "transcription": text,
            "download": f"/download/{output_txt.name}",
            "metadata": {
                "session_id": session_id,
                "source_type": source_type,
                "language": transcription_result["language"],
                "word_count": transcription_result["word_count"],
                "char_count": transcription_result["char_count"],
                "timestamp": datetime.now().isoformat()
            }
        })
    
    except ValueError as e:
        # Erros de validação (duração, tamanho, etc.)
        logger.warning(f"Validação falhou: {e}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        # Erros inesperados
        logger.error(f"Erro durante processamento: {e}", exc_info=True)
        return jsonify({
            "error": f"Erro no processamento: {str(e)}"
        }), 500
    
    finally:
        # Limpeza de arquivos temporários
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Arquivo temporário removido: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Erro ao remover temporário {temp_file}: {e}")


@app.route("/download/<path:filename>")
def download_file(filename):
    """Endpoint para download de transcrições."""
    
    
    safe_filename = os.path.basename(filename)
    file_path = OUTPUT_DIR / safe_filename
    
    if not file_path.exists():
        logger.warning(f"Arquivo não encontrado: {safe_filename}")
        abort(404)
    
    logger.info(f"Download solicitado: {safe_filename}")
    
    return send_from_directory(
        OUTPUT_DIR.resolve(),
        safe_filename,
        as_attachment=True,
        download_name=f"transcricao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handler para arquivos muito grandes."""
    return jsonify({
        "error": f"Arquivo muito grande. Tamanho máximo: {MAX_FILE_SIZE // (1024*1024)}MB"
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handler para erros internos."""
    logger.error(f"Erro 500: {error}")
    return jsonify({
        "error": "Erro interno do servidor. Tente novamente."
    }), 500


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("AETHER IA - SISTEMA DE TRANSCRIÇÃO")
    logger.info("=" * 70)
    logger.info(f"Modelo Whisper: {MODEL_NAME}")
    logger.info(f"Tamanho máximo de arquivo: {MAX_FILE_SIZE // (1024*1024)}MB")
    logger.info(f"Duração máxima: {MAX_VIDEO_DURATION // 60} minutos")
    logger.info(f"Formatos de áudio: {', '.join(sorted(ALLOWED_AUDIO))}")
    logger.info(f"Formatos de vídeo: {', '.join(sorted(ALLOWED_VIDEO))}")
    logger.info("=" * 70)
    
    
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True
    )