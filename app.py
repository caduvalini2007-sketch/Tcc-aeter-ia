#!/usr/bin/env python3
"""
Aether IA - Plataforma Integrada de Inteligência Artificial
Sistema completo com Chatbot, OCR e Transcrição de Vídeo
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify
from flask_cors import CORS

# ==========================================
# CONFIGURAÇÃO DE LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aether_ia.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("aether_ia")

# ==========================================
# INICIALIZAÇÃO DO FLASK
# ==========================================

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

CORS(app)

# Configurações globais
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.secret_key = os.getenv('SECRET_KEY', 'aether_ia_secret_key_change_in_production')

# Criar diretórios necessários
REQUIRED_DIRS = [
    'static/uploads',      # OCR
    'uploads',             # Transcrição de vídeo
    'temp',                # Temporários
    'outputs',             # Outputs de transcrição
    'chat_history',        # Histórico do chatbot
    'templates'            # Templates HTML
]

for directory in REQUIRED_DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretório verificado: {directory}")

# ==========================================
# IMPORTAÇÃO DOS MÓDULOS
# ==========================================

CHATBOT_AVAILABLE = False
CHATBOT_ERROR_MSG = ""

OCR_AVAILABLE = False
OCR_ERROR_MSG = ""

VIDEO_AVAILABLE = False
VIDEO_ERROR_MSG = ""

# Importa módulos do chatbot
try:
    logger.info("Tentando carregar módulo Chatbot...")
    from chatbot_backend import (
        chat_endpoint,
        search_endpoint,
        analyze_query_endpoint,
        find_similar_endpoint,
        get_history,
        list_histories,
        api_ocr as chatbot_ocr,
        status,
        list_known_terms,
        add_known_term
    )
    CHATBOT_AVAILABLE = True
    logger.info("✓ Módulo Chatbot carregado com sucesso")
    
except ImportError as e:
    CHATBOT_ERROR_MSG = f"Erro de importação: {e}"
    logger.warning(f"✗ Chatbot não disponível - {CHATBOT_ERROR_MSG}")
    logger.warning("   Verifique se chatbot_backend.py existe e se as dependências estão instaladas")
    logger.warning("   Execute: pip install ollama (ou pip install transformers torch)")
    
except Exception as e:
    CHATBOT_ERROR_MSG = f"Erro ao carregar: {e}"
    logger.error(f"✗ Chatbot não disponível - {CHATBOT_ERROR_MSG}")
    logger.error(f"   Erro completo: {e.__class__.__name__}: {str(e)}")

# Importa módulos do OCR
try:
    logger.info("Tentando carregar módulo OCR...")
    from ocr_backend import (
        index as ocr_index_route,
        request_entity_too_large,
        internal_server_error
    )
    OCR_AVAILABLE = True
    logger.info("✓ Módulo OCR carregado com sucesso")
    
except ImportError as e:
    OCR_ERROR_MSG = f"Erro de importação: {e}"
    logger.warning(f"✗ OCR não disponível - {OCR_ERROR_MSG}")
    logger.warning("   Verifique se ocr_backend.py existe")
    
except Exception as e:
    OCR_ERROR_MSG = f"Erro ao carregar: {e}"
    logger.warning(f"✗ OCR não disponível - {OCR_ERROR_MSG}")

# Importa módulos de transcrição de vídeo
try:
    logger.info("Tentando carregar módulo Transcrição de Vídeo...")
    from video_transcription_backend import (
        index as video_index_route,
        transcribe_endpoint,
        download_file
    )
    VIDEO_AVAILABLE = True
    logger.info("✓ Módulo Transcrição de Vídeo carregado com sucesso")
    
except ImportError as e:
    VIDEO_ERROR_MSG = f"Erro de importação: {e}"
    logger.warning(f"✗ Transcrição de Vídeo não disponível - {VIDEO_ERROR_MSG}")
    logger.warning("   Verifique se video_transcription_backend.py existe")
    
except Exception as e:
    VIDEO_ERROR_MSG = f"Erro ao carregar: {e}"
    logger.warning(f"✗ Transcrição de Vídeo não disponível - {VIDEO_ERROR_MSG}")

# ==========================================
# ROTAS PRINCIPAIS
# ==========================================

@app.route('/')
def home():
    """Página inicial da plataforma"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Erro ao renderizar index.html: {e}")
        return f"""
        <html>
        <head><title>Aether IA</title></head>
        <body>
            <h1>Aether IA</h1>
            <p>Erro ao carregar página inicial. Verifique se templates/index.html existe.</p>
            <p>Erro: {e}</p>
            <hr>
            <p><a href="/health">Health Check</a> | <a href="/api/status">API Status</a></p>
        </body>
        </html>
        """, 500

@app.route('/health')
def health_check():
    """Health check para monitoramento"""
    return jsonify({
        'status': 'healthy',
        'modules': {
            'chatbot': {
                'available': CHATBOT_AVAILABLE,
                'error': CHATBOT_ERROR_MSG if not CHATBOT_AVAILABLE else None
            },
            'ocr': {
                'available': OCR_AVAILABLE,
                'error': OCR_ERROR_MSG if not OCR_AVAILABLE else None
            },
            'video_transcription': {
                'available': VIDEO_AVAILABLE,
                'error': VIDEO_ERROR_MSG if not VIDEO_AVAILABLE else None
            }
        },
        'directories': {
            dir_name: os.path.exists(dir_name) for dir_name in REQUIRED_DIRS
        }
    }), 200

# ==========================================
# ROTAS DO CHATBOT
# ==========================================

if CHATBOT_AVAILABLE:
    @app.route('/chatbot')
    def chatbot_page():
        """Página do chatbot"""
        try:
            return render_template('index4.html')
        except Exception as e:
            logger.error(f"Erro ao renderizar index4.html: {e}")
            return render_template_safe('error.html', 
                message="Erro ao carregar Chatbot",
                details=f"Verifique se templates/index4.html existe. Erro: {e}"), 500
    
    # Registra rotas da API do chatbot
    app.add_url_rule('/api/chat', 'chat_endpoint', chat_endpoint, methods=['GET', 'POST'])
    app.add_url_rule('/api/search', 'search_endpoint', search_endpoint, methods=['GET', 'POST'])
    app.add_url_rule('/api/analyze', 'analyze_query_endpoint', analyze_query_endpoint, methods=['POST'])
    app.add_url_rule('/api/similar', 'find_similar_endpoint', find_similar_endpoint, methods=['POST'])
    app.add_url_rule('/api/history/<session_id>', 'get_history', get_history, methods=['GET'])
    app.add_url_rule('/api/histories', 'list_histories', list_histories, methods=['GET'])
    app.add_url_rule('/api/ocr', 'chatbot_ocr', chatbot_ocr, methods=['POST'])
    app.add_url_rule('/api/status', 'status', status, methods=['GET'])
    app.add_url_rule('/api/terms', 'list_known_terms', list_known_terms, methods=['GET'])
    app.add_url_rule('/api/terms/add', 'add_known_term_route', add_known_term, methods=['POST'])
    
    logger.info("✓ Rotas do Chatbot registradas")
    
else:
    @app.route('/chatbot')
    def chatbot_unavailable():
        """Página de erro quando chatbot não está disponível"""
        error_details = f"""
        <h3>Possíveis causas:</h3>
        <ul>
            <li>Ollama não está instalado ou não está rodando</li>
            <li>chatbot_backend.py não foi encontrado</li>
            <li>Dependências Python não instaladas</li>
        </ul>
        
        <h3>Como resolver:</h3>
        <ol>
            <li><strong>Instale o Ollama:</strong>
                <ul>
                    <li>Baixe em: <a href="https://ollama.ai" target="_blank">https://ollama.ai</a></li>
                    <li>Execute: <code>ollama pull llama3.2</code></li>
                </ul>
            </li>
            <li><strong>Verifique as dependências:</strong>
                <ul>
                    <li><code>pip install ollama</code></li>
                    <li>Ou: <code>pip install transformers torch</code></li>
                </ul>
            </li>
            <li><strong>Teste o Ollama:</strong>
                <ul>
                    <li><code>ollama --version</code></li>
                    <li><code>ollama list</code></li>
                    <li><code>ollama run llama3.2 "Olá"</code></li>
                </ul>
            </li>
            <li><strong>Verifique os logs:</strong>
                <ul>
                    <li>Arquivo: <code>aether_ia.log</code></li>
                    <li>Erro: <code>{CHATBOT_ERROR_MSG}</code></li>
                </ul>
            </li>
        </ol>
        
        <h3>Verificação rápida:</h3>
        <pre>python check_dependencies.py</pre>
        
        <p><a href="/">← Voltar para página inicial</a></p>
        """
        
        return render_template_safe('error.html', 
            message="Chatbot não disponível",
            details=error_details), 503
    
    @app.route('/api/status')
    def status_unavailable():
        """Endpoint de status quando chatbot não está disponível"""
        return jsonify({
            "status": "unavailable",
            "error": CHATBOT_ERROR_MSG,
            "suggestions": [
                "Instale Ollama: https://ollama.ai",
                "Execute: ollama pull llama3.2",
                "Ou instale: pip install transformers torch",
                "Verifique os logs em: aether_ia.log"
            ]
        }), 503

# ==========================================
# ROTAS DO OCR
# ==========================================

if OCR_AVAILABLE:
    @app.route('/ocr', methods=['GET', 'POST'])
    def ocr_page():
        """Página do OCR - redireciona para rota do módulo"""
        return ocr_index_route()
    
    # Handlers de erro do OCR
    app.register_error_handler(413, request_entity_too_large)
    app.register_error_handler(500, internal_server_error)
    
    logger.info("✓ Rotas do OCR registradas")
    
else:
    @app.route('/ocr')
    def ocr_unavailable():
        error_details = f"""
        <h3>Possíveis causas:</h3>
        <ul>
            <li>Tesseract OCR não está instalado</li>
            <li>ocr_backend.py não foi encontrado</li>
            <li>Dependências Python não instaladas</li>
        </ul>
        
        <h3>Como resolver:</h3>
        <ol>
            <li><strong>Instale o Tesseract:</strong>
                <ul>
                    <li>Windows: <a href="https://github.com/UB-Mannheim/tesseract/wiki" target="_blank">Baixar instalador</a></li>
                    <li>Linux: <code>sudo apt install tesseract-ocr tesseract-ocr-por</code></li>
                    <li>Mac: <code>brew install tesseract</code></li>
                </ul>
            </li>
            <li><strong>Instale as dependências Python:</strong>
                <ul>
                    <li><code>pip install easyocr opencv-python pillow pytesseract</code></li>
                </ul>
            </li>
            <li><strong>Verifique a instalação:</strong>
                <ul>
                    <li><code>tesseract --version</code></li>
                </ul>
            </li>
        </ol>
        
        <p>Erro: <code>{OCR_ERROR_MSG}</code></p>
        <p><a href="/">← Voltar para página inicial</a></p>
        """
        
        return render_template_safe('error.html',
            message="OCR não disponível",
            details=error_details), 503

# ==========================================
# ROTAS DA TRANSCRIÇÃO DE VÍDEO
# ==========================================

if VIDEO_AVAILABLE:
    @app.route('/transcribe-video')
    def video_page():
        """Página de transcrição de vídeo"""
        return video_index_route()
    
    app.add_url_rule('/transcribe', 'transcribe_endpoint', transcribe_endpoint, methods=['POST'])
    app.add_url_rule('/download/<path:filename>', 'download_file', download_file, methods=['GET'])
    
    logger.info("✓ Rotas de Transcrição de Vídeo registradas")
    
else:
    @app.route('/transcribe-video')
    def video_unavailable():
        error_details = f"""
        <h3>Possíveis causas:</h3>
        <ul>
            <li>Whisper AI não está instalado</li>
            <li>FFmpeg não está instalado</li>
            <li>video_transcription_backend.py não foi encontrado</li>
        </ul>
        
        <h3>Como resolver:</h3>
        <ol>
            <li><strong>Instale o FFmpeg:</strong>
                <ul>
                    <li>Windows: <a href="https://ffmpeg.org/download.html" target="_blank">Baixar e adicionar ao PATH</a></li>
                    <li>Linux: <code>sudo apt install ffmpeg</code></li>
                    <li>Mac: <code>brew install ffmpeg</code></li>
                </ul>
            </li>
            <li><strong>Instale as dependências Python:</strong>
                <ul>
                    <li><code>pip install openai-whisper pytubefix moviepy</code></li>
                </ul>
            </li>
            <li><strong>Verifique a instalação:</strong>
                <ul>
                    <li><code>ffmpeg -version</code></li>
                </ul>
            </li>
        </ol>
        
        <p>Erro: <code>{VIDEO_ERROR_MSG}</code></p>
        <p><a href="/">← Voltar para página inicial</a></p>
        """
        
        return render_template_safe('error.html',
            message="Transcrição de Vídeo não disponível",
            details=error_details), 503

# ==========================================
# ROTAS ESTÁTICAS E ASSETS
# ==========================================

@app.route('/favicon.ico')
def favicon():
    """Favicon da aplicação"""
    try:
        return send_from_directory(
            os.path.join(app.root_path, 'static'),
            'favicon.ico',
            mimetype='image/vnd.microsoft.icon'
        )
    except Exception:
        # Retorna um favicon vazio se não existir
        return '', 204

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve arquivos estáticos"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Erro ao servir arquivo estático {filename}: {e}")
        return f"Arquivo não encontrado: {filename}", 404

# ==========================================
# TRATAMENTO DE ERROS GLOBAL
# ==========================================

def render_template_safe(template_name, **context):
    """Renderiza template com fallback para HTML simples"""
    try:
        return render_template(template_name, **context)
    except Exception as e:
        logger.error(f"Erro ao renderizar {template_name}: {e}")
        # Fallback para HTML simples
        message = context.get('message', 'Erro')
        details = context.get('details', str(e))
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{message}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #e53e3e; }}
                code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
                a {{ color: #3182ce; }}
                pre {{ background: #f0f0f0; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>{message}</h1>
            <div>{details}</div>
        </body>
        </html>
        """

@app.errorhandler(404)
def not_found(error):
    """Página não encontrada"""
    return render_template_safe('error.html',
        message="Página não encontrada",
        details="A página que você procura não existe. <a href='/'>Voltar para o início</a>"), 404

@app.errorhandler(500)
def internal_error(error):
    """Erro interno do servidor"""
    logger.error(f"Erro 500: {error}")
    return render_template_safe('error.html',
        message="Erro interno do servidor",
        details="Algo deu errado. Tente novamente ou verifique os logs em aether_ia.log"), 500

@app.errorhandler(503)
def service_unavailable(error):
    """Serviço indisponível"""
    return render_template_safe('error.html',
        message="Serviço indisponível",
        details="O serviço está temporariamente indisponível."), 503

# ==========================================
# COMANDOS CLI
# ==========================================

@app.cli.command()
def init():
    """Inicializa o sistema"""
    print("🚀 Inicializando Aether IA...")
    
    # Verifica dependências
    print("\n📦 Verificando dependências...")
    
    dependencies_check = {
        'Flask': True,
        'Chatbot (Ollama/Transformers)': CHATBOT_AVAILABLE,
        'OCR (Tesseract)': OCR_AVAILABLE,
        'Transcrição (Whisper AI)': VIDEO_AVAILABLE,
    }
    
    for dep, available in dependencies_check.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    # Mostra erros se houver
    if not CHATBOT_AVAILABLE and CHATBOT_ERROR_MSG:
        print(f"     ⚠️  Chatbot: {CHATBOT_ERROR_MSG}")
    if not OCR_AVAILABLE and OCR_ERROR_MSG:
        print(f"     ⚠️  OCR: {OCR_ERROR_MSG}")
    if not VIDEO_AVAILABLE and VIDEO_ERROR_MSG:
        print(f"     ⚠️  Vídeo: {VIDEO_ERROR_MSG}")
    
    print("\n📁 Verificando diretórios...")
    for directory in REQUIRED_DIRS:
        exists = "✓" if os.path.exists(directory) else "✗"
        print(f"  {exists} {directory}")
    
    print("\n✅ Inicialização concluída!")
    print(f"🔧 Módulos disponíveis: {sum(dependencies_check.values())}/{len(dependencies_check)}")
    
    if sum(dependencies_check.values()) < len(dependencies_check):
        print("\n⚠️  Alguns módulos não estão disponíveis.")
        print("   Execute 'python check_dependencies.py' para mais informações.")

@app.cli.command()
def check():
    """Verifica status do sistema"""
    print("🔍 Status do Sistema Aether IA\n")
    print(f"{'Módulo':<40} {'Status':<15} {'Erro':<30}")
    print("-" * 85)
    
    modules = [
        ('Chatbot (Ollama/Transformers)', CHATBOT_AVAILABLE, CHATBOT_ERROR_MSG),
        ('OCR (Tesseract + OpenCV)', OCR_AVAILABLE, OCR_ERROR_MSG),
        ('Transcrição (Whisper AI)', VIDEO_AVAILABLE, VIDEO_ERROR_MSG),
    ]
    
    for name, available, error_msg in modules:
        status = "✓ Online" if available else "✗ Offline"
        error = error_msg[:27] + "..." if error_msg and len(error_msg) > 30 else error_msg or ""
        print(f"{name:<40} {status:<15} {error:<30}")
    
    print("-" * 85)
    
    total = len(modules)
    available = sum(1 for _, avail, _ in modules if avail)
    print(f"\n📊 {available}/{total} módulos disponíveis")
    
    if available < total:
        print("\n💡 Dicas:")
        print("   • Execute 'python check_dependencies.py' para diagnóstico completo")
        print("   • Acesse '/health' no navegador para ver detalhes")
        print("   • Verifique 'aether_ia.log' para logs detalhados")

# ==========================================
# INFORMAÇÕES DE INICIALIZAÇÃO
# ==========================================

def print_startup_info():
    """Imprime informações de inicialização"""
    print("\n" + "=" * 70)
    print(" " * 20 + "AETHER IA - PLATAFORMA DE IA")
    print("=" * 70)
    print(f"\n🌐 Servidor: http://0.0.0.0:5000")
    print(f"📁 Diretório: {os.getcwd()}")
    print(f"\n📦 Módulos Disponíveis:")
    print(f"  {'✓' if CHATBOT_AVAILABLE else '✗'} Chatbot Inteligente")
    if not CHATBOT_AVAILABLE:
        print(f"     ⚠️  {CHATBOT_ERROR_MSG}")
    print(f"  {'✓' if OCR_AVAILABLE else '✗'} OCR (Transcrição de Imagem)")
    if not OCR_AVAILABLE and OCR_ERROR_MSG:
        print(f"     ⚠️  {OCR_ERROR_MSG}")
    print(f"  {'✓' if VIDEO_AVAILABLE else '✗'} Transcrição de Vídeo/Áudio")
    if not VIDEO_AVAILABLE and VIDEO_ERROR_MSG:
        print(f"     ⚠️  {VIDEO_ERROR_MSG}")
    
    print(f"\n🔗 Rotas Principais:")
    print(f"  http://localhost:5000/              → Página Inicial")
    print(f"  http://localhost:5000/chatbot       → Chatbot {'✓' if CHATBOT_AVAILABLE else '✗'}")
    print(f"  http://localhost:5000/ocr           → OCR {'✓' if OCR_AVAILABLE else '✗'}")
    print(f"  http://localhost:5000/transcribe-video → Transcrição {'✓' if VIDEO_AVAILABLE else '✗'}")
    print(f"  http://localhost:5000/health        → Health Check")
    print(f"  http://localhost:5000/api/status    → API Status")
    
    total = sum([CHATBOT_AVAILABLE, OCR_AVAILABLE, VIDEO_AVAILABLE])
    if total < 3:
        print(f"\n⚠️  Alguns módulos não estão disponíveis ({total}/3)")
        print(f"   Execute: python check_dependencies.py")
    
    print("\n" + "=" * 70 + "\n")

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

if __name__ == '__main__':
    print_startup_info()
    
    # Configurações do servidor
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Iniciando servidor em {host}:{port}")
    logger.info(f"Modo debug: {debug}")
    
    # Aviso sobre produção
    if not debug:
        logger.warning("ATENÇÃO: Rodando em modo de produção")
        logger.warning("Use um servidor WSGI como Gunicorn para produção:")
        logger.warning("  gunicorn -w 4 -b 0.0.0.0:5000 --timeout 600 app:app")
    
    # Aviso se nenhum módulo disponível
    if not any([CHATBOT_AVAILABLE, OCR_AVAILABLE, VIDEO_AVAILABLE]):
        logger.error("=" * 70)
        logger.error("⚠️  NENHUM MÓDULO DISPONÍVEL!")
        logger.error("=" * 70)
        logger.error("Execute: python check_dependencies.py")
        logger.error("Ou acesse: http://localhost:5000/health")
        logger.error("=" * 70)
    
    # Inicia o servidor
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {e}")
        logger.error("Verifique se a porta 5000 está disponível")
        logger.error("Tente: python app.py --port 8000")
        sys.exit(1)