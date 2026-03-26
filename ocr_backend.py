import os
import cv2
import re
import logging
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import pytesseract
from flask import Flask, render_template, request, flash

# ==========================================
# CONFIGURAÇÕES
# ==========================================

# Configuracao do Tesseract 
pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Pichau/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# INICIALIZAÇÃO DO FLASK
# ==========================================

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui_troque_em_producao'  # Necessário para flash messages

# Configurações de upload
UPLOAD_FOLDER = "static/uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_old_uploads(folder, max_age_hours=24):
    """Remove arquivos antigos da pasta de uploads."""
    try:
        now = datetime.now()
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - file_modified > timedelta(hours=max_age_hours):
                    os.remove(filepath)
                    logger.info(f"Arquivo antigo removido: {filename}")
    except Exception as e:
        logger.error(f"Erro ao limpar uploads antigos: {e}")


def preprocess_variants(img):
    """
    Gera múltiplas variantes de pré-processamento da imagem.
    Retorna uma lista de imagens processadas.
    """
    variants = []
    
    try:
        # 1. Conversão para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants.append(("Original Gray", gray))
        
        # 2. Redimensionar se muito grande (melhora performance)
        height, width = gray.shape
        if width > 2000 or height > 2000:
            scale = min(2000/width, 2000/height)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 3. Denoising (redução de ruído)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        variants.append(("Denoised", denoised))
        
        # 4. Blur mediano (remove ruído mantendo bordas)
        blur = cv2.medianBlur(gray, 3)
        variants.append(("Median Blur", blur))
        
        # 5. Threshold adaptativo
        adapt_thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 6
        )
        variants.append(("Adaptive Threshold", adapt_thresh))
        
        # 6. Threshold OTSU
        _, otsu = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(("OTSU", otsu))
        
        # 7. Sharpening (acentuação de bordas)
        kernel_sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        sharp = cv2.addWeighted(gray, 1.5, kernel_sharpen, -0.5, 0)
        variants.append(("Sharpened", sharp))
        
        # 8. Dilatação + Erosão (remove pequenos ruídos)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        variants.append(("Morphology", morph))
        
        # 9. Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("CLAHE", clahe_img))
        
        logger.info(f"Geradas {len(variants)} variantes de pré-processamento")
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        # Retorna pelo menos a imagem original em caso de erro
        if len(variants) == 0:
            variants.append(("Fallback", img))
    
    return variants


def calculate_text_quality(text: str) -> dict:
    """
    Calcula métricas de qualidade do texto extraído.
    Retorna um dicionário com score e estatísticas.
    """
    # Remove espaços em branco excessivos
    text_clean = ' '.join(text.split())
    
    # Conta palavras legíveis (2+ caracteres alfabéticos)
    words = re.findall(r'[A-Za-zÀ-ÿ]{2,}', text)
    word_count = len(words)
    
    # Conta caracteres especiais/ruído
    special_chars = len(re.findall(r'[^A-Za-zÀ-ÿ0-9\s.,!?;:\-\'"()]', text))
    
    # Conta números
    numbers = len(re.findall(r'\d+', text))
    
    # Score baseado em: palavras - (ruído * penalidade)
    score = word_count - (special_chars * 0.5)
    
    # Penaliza se tiver muito pouco texto
    if len(text_clean) < 10:
        score *= 0.5
    
    return {
        'score': max(0, score),
        'word_count': word_count,
        'char_count': len(text_clean),
        'special_chars': special_chars,
        'numbers': numbers
    }


def run_ocr_with_variants(image_path, lang='por'):
    """
    Executa OCR em múltiplas variantes da imagem e retorna o melhor resultado.
    """
    try:
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        logger.info(f"Processando imagem: {image_path}")
        
        # Gera variantes de pré-processamento
        variants = preprocess_variants(img)
        
        best_text = ""
        best_score = -1
        best_method = "N/A"
        results = []
        
        # Configurações do Tesseract
        # --oem 1: LSTM neural net mode
        # --psm 3: Fully automatic page segmentation
        tesseract_config = "--oem 1 --psm 3"
        
        # Testa cada variante
        for name, variant in variants:
            try:
                # Executa OCR
                text = pytesseract.image_to_string(
                    variant, 
                    lang=lang,
                    config=tesseract_config
                )
                
                # Calcula qualidade
                quality = calculate_text_quality(text)
                score = quality['score']
                
                results.append({
                    'method': name,
                    'score': score,
                    'text_preview': text[:100] if text else '',
                    'stats': quality
                })
                
                logger.info(f"{name}: score={score:.2f}, words={quality['word_count']}")
                
                # Atualiza melhor resultado
                if score > best_score:
                    best_score = score
                    best_text = text
                    best_method = name
                    
            except Exception as e:
                logger.warning(f"Erro ao processar variante {name}: {e}")
                continue
        
        # Log do resultado final
        logger.info(f"Melhor método: {best_method} (score: {best_score:.2f})")
        
        # Se não encontrou nada, retorna mensagem apropriada
        if not best_text.strip():
            best_text = "[Nenhum texto foi detectado na imagem. Tente uma imagem com melhor qualidade ou contraste.]"
        
        return best_text.strip()
        
    except Exception as e:
        logger.error(f"Erro no OCR: {e}")
        return f"[Erro ao processar imagem: {str(e)}]"


# ==========================================
# ROTAS
# ==========================================

@app.route("/ocr", methods=["GET", "POST"])
def index():
    """Rota principal - upload e processamento OCR."""
    
    text_result = None
    image_path = None
    
    # Limpa arquivos antigos a cada requisição (em produção, use um job agendado)
    clean_old_uploads(app.config["UPLOAD_FOLDER"], max_age_hours=1)
    
    if request.method == "POST":
        # Verifica se o arquivo foi enviado
        if "file" not in request.files:
            flash("Nenhum arquivo foi enviado.", "error")
            return render_template("transcricao_java.html")
        
        file = request.files["file"]
        
        # Verifica se o arquivo tem nome
        if file.filename == "":
            flash("Arquivo inválido ou sem nome.", "error")
            return render_template("transcricao_java.html")
        
        # Verifica extensão
        if not allowed_file(file.filename):
            flash(f"Tipo de arquivo não permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}", "error")
            return render_template("transcricao_java.html")
        
        try:
            # Gera nome seguro com timestamp
            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # Salva o arquivo
            file.save(filepath)
            logger.info(f"Arquivo salvo: {filename}")
            
            # Verifica se é realmente uma imagem
            test_img = cv2.imread(filepath)
            if test_img is None:
                os.remove(filepath)
                flash("O arquivo enviado não é uma imagem válida.", "error")
                return render_template("transcricao_java.html")
            
            image_path = filepath
            
            # Obtém idioma (pode ser expandido para receber via form)
            lang = request.form.get('lang', 'por')
            
            # Executa OCR
            logger.info(f"Iniciando OCR (idioma: {lang})")
            text_result = run_ocr_with_variants(filepath, lang=lang)
            
            if text_result:
                flash(f"✅ Texto extraído com sucesso! ({len(text_result)} caracteres)", "success")
            else:
                flash("⚠️ Nenhum texto foi detectado na imagem.", "warning")
                
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            flash(f"Erro ao processar imagem: {str(e)}", "error")
            
            # Remove arquivo em caso de erro
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                image_path = None
    
    return render_template(
        "transcricao_java.html",
        image_path=image_path,
        text=text_result
    )


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handler para arquivos muito grandes."""
    flash("Arquivo muito grande! Tamanho máximo: 10MB", "error")
    return render_template("transcricao_java.html"), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handler para erros internos."""
    logger.error(f"Erro 500: {error}")
    flash("Erro interno do servidor. Tente novamente.", "error")
    return render_template("transcricao_java.html"), 500


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    logger.info("Iniciando servidor Flask OCR...")
    logger.info(f"Pasta de uploads: {UPLOAD_FOLDER}")
    logger.info(f"Extensões permitidas: {ALLOWED_EXTENSIONS}")
    
    # Em produção, use um servidor WSGI como Gunicorn
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )