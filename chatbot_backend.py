
import os
import time
import re
import json
import hashlib
import logging
import random
import tempfile
import threading
import io
from datetime import datetime
from functools import wraps
from typing import List, Dict, Optional, Any, Generator, Tuple
from dataclasses import dataclass
from collections import defaultdict
from difflib import SequenceMatcher

from flask import request, jsonify, Response, render_template
import requests
from bs4 import BeautifulSoup

# LangChain Imports (opcional)
try:
    from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain não instalado")

# Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama não instalado. Execute: pip install ollama")

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@dataclass
class Config:
    """Configuração centralizada"""
    
    # Ollama é o provider padrão
    LLM_PROVIDER: str = "ollama"
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:3b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    HTTP_TIMEOUT: int = 10
    MAX_PAGES_PER_QUERY: int = 5
    MAX_SEARCH_RESULTS: int = 8
    
    HISTORY_DIR: str = "chat_history"
    
    STREAM_CHAR_DELAY: float = 0.008
    HEARTBEAT_INTERVAL: float = 15.0
    
    MAX_GENERATION_CHUNK: int = 2048
    MAX_TOTAL_GENERATION: int = 65536
    MAX_CONTINUATION_CALLS: int = 16
    DEFAULT_MAX_WORDS: int = 1500
    MIN_RESPONSE_WORDS: int = 50
    
    CACHE_TTL_SECONDS: int = 3600
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600
    
    STOP_TOKEN: str = "<|endofresponse|>"
    
    SIMILARITY_THRESHOLD: float = 0.75
    DISAMBIGUATION_THRESHOLD: float = 0.85
    
    SEARCH_TRIGGERS: tuple = (
        "pesquise", "busque", "procure", "encontre", "o que é", "quem é",
        "quando", "onde", "como funciona", "me fale sobre", "notícias",
        "atual", "hoje", "recente", "último", "nova", "novo", "2024", "2025",
        "search", "find", "what is", "who is", "latest", "news", "current",
        "explique", "descreva", "compare", "diferença", "história de"
    )
    
    def __post_init__(self):
        os.makedirs(self.HISTORY_DIR, exist_ok=True)

config = Config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chatbot_backend")

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        with self._lock:
            now = time.time()
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if now - ts < self.window_seconds
            ]
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            self.requests[identifier].append(now)
            return True

rate_limiter = RateLimiter(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW)

# ============================================================================
# VALIDAÇÃO
# ============================================================================

class InputValidator:
    @staticmethod
    def validate_message(message: str, max_length: int = 5000) -> str:
        if not message or not isinstance(message, str):
            raise ValueError("Mensagem inválida")
        message = message.strip()
        if len(message) == 0:
            raise ValueError("Mensagem vazia")
        if len(message) > max_length:
            raise ValueError("Mensagem muito longa")
        message = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', message)
        return message
    
    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID inválido")
        if not re.match(r'^[a-zA-Z0-9\-]{8,64}$', session_id):
            raise ValueError("Formato inválido")
        return session_id

validator = InputValidator()

# ============================================================================
# QUERY ANALYZER
# ============================================================================

class QueryAnalyzer:
    SIMILARITY_THRESHOLD = 0.75
    KNOWN_SIMILAR_TERMS = {
        "dark souls": ["black souls", "dark soul", "demon souls"],
        "minecraft": ["mincraft", "mine craft", "mindcraft"],
    }

    @staticmethod
    def calculate_similarity(str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    @staticmethod
    def normalize_query(query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)
        return query

    @staticmethod
    def extract_main_terms(query: str) -> list:
        query_norm = QueryAnalyzer.normalize_query(query)
        words = query_norm.split()
        return [w for w in words if len(w) > 1]

    @staticmethod
    def find_similar_terms(term: str):
        term = QueryAnalyzer.normalize_query(term)
        similars = []
        for known_term, variants in QueryAnalyzer.KNOWN_SIMILAR_TERMS.items():
            sim_main = QueryAnalyzer.calculate_similarity(term, known_term)
            if sim_main >= QueryAnalyzer.SIMILARITY_THRESHOLD:
                similars.append({
                    "term": term,
                    "correct_term": known_term,
                    "similarity": sim_main,
                    "type": "main"
                })
            for alt in variants:
                sim_alt = QueryAnalyzer.calculate_similarity(term, alt)
                if sim_alt >= QueryAnalyzer.SIMILARITY_THRESHOLD:
                    similars.append({
                        "term": term,
                        "correct_term": known_term,
                        "similarity": sim_alt,
                        "type": "variant"
                    })
        return similars

    @staticmethod
    def analyze_query(query: str) -> dict:
        main_terms = QueryAnalyzer.extract_main_terms(query)
        potential_confusions = []
        
        for term in main_terms:
            sims = QueryAnalyzer.find_similar_terms(term)
            if sims:
                potential_confusions.extend(sims)
        
        disambiguation_needed = len(potential_confusions) > 0
        suggested_queries = []
        
        if disambiguation_needed:
            corrected = query.lower()
            for conf in potential_confusions:
                corrected = re.sub(
                    r'\b' + re.escape(conf["term"]) + r'\b',
                    conf["correct_term"],
                    corrected
                )
            suggested_queries.append({
                "query": corrected,
                "reason": f"Você quis dizer: {', '.join(set(conf['correct_term'] for conf in potential_confusions))}"
            })
            
        return {
            "original_query": query,
            "normalized_query": QueryAnalyzer.normalize_query(query),
            "main_terms": main_terms,
            "potential_confusions": potential_confusions,
            "disambiguation_needed": disambiguation_needed,
            "suggested_queries": suggested_queries,
            "contexts": []
        }

# ============================================================================
# SMART SEARCH MANAGER
# ============================================================================

class SmartSearchManager:
    def __init__(self):
        self._search_tool = None
        self._search_results_tool = None
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.analyzer = QueryAnalyzer()
    
    def _init_tools(self) -> bool:
        if not LANGCHAIN_AVAILABLE:
            return False
        
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                max_results=config.MAX_SEARCH_RESULTS,
                region="br-pt",
                safesearch="moderate"
            )
            self._search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
            self._search_results_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
            logger.info("Ferramentas de busca inicializadas")
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar busca: {e}")
            return False
    
    def needs_search(self, query: str) -> bool:
        query_lower = query.lower()
        for trigger in config.SEARCH_TRIGGERS:
            if trigger in query_lower:
                return True
        
        question_patterns = [
            r'\b(qual|quais|quanto|quantos|onde|quando|como|por ?que|quem)\b',
            r'\?$',
            r'\b(preço|cotação|valor|resultado|placar|tempo|clima)\b'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _get_cached(self, query: str) -> Optional[Dict]:
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if time.time() - cached['timestamp'] < config.CACHE_TTL_SECONDS:
                    return cached['data']
        return None
    
    def _set_cache(self, query: str, data: Dict):
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        
        with self._lock:
            self._cache[cache_key] = {'data': data, 'timestamp': time.time()}
            if len(self._cache) > 100:
                oldest = sorted(self._cache.items(), key=lambda x: x[1]['timestamp'])[:20]
                for key, _ in oldest:
                    del self._cache[key]
    
    def _execute_search(self, query: str) -> Tuple[str, List[Dict]]:
        if LANGCHAIN_AVAILABLE and self._search_tool is None:
            self._init_tools()
        
        if self._search_tool:
            try:
                search_text = self._search_tool.run(query)
                detailed_results = []
                
                try:
                    raw_results = self._search_results_tool.run(query)
                    if isinstance(raw_results, str):
                        detailed_results = self._parse_search_results(raw_results)
                    elif isinstance(raw_results, list):
                        detailed_results = raw_results
                except Exception:
                    pass
                
                return search_text, detailed_results
            except Exception as e:
                logger.error(f"Erro na busca: {e}")
        
        return self._fallback_search_raw(query)
    
    def _fallback_search_raw(self, query: str) -> Tuple[str, List[Dict]]:
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.select('.result')[:config.MAX_SEARCH_RESULTS]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                link_elem = result.select_one('.result__url')
                
                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'snippet': snippet_elem.get_text(strip=True),
                        'link': link_elem.get_text(strip=True) if link_elem else ''
                    })
            
            summary = "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])
            return summary or "Nenhum resultado encontrado.", results
            
        except Exception as e:
            logger.error(f"Erro na busca fallback: {e}")
            return "Não foi possível realizar a busca.", []
    
    def _parse_search_results(self, results_str: str) -> List[Dict]:
        results = []
        try:
            if results_str.startswith('['):
                return json.loads(results_str)
            
            items = re.split(r'\n(?=\d+\.|\[\d+\])', results_str)
            for item in items:
                if item.strip():
                    results.append({
                        'snippet': item.strip()[:500],
                        'title': item.split('\n')[0][:100] if '\n' in item else item[:100]
                    })
        except Exception:
            results.append({'snippet': results_str[:500]})
        
        return results[:config.MAX_SEARCH_RESULTS]
    
    def _validate_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        if not results:
            return []
        
        main_terms = query_analysis.get('main_terms', [])
        scored_results = []
        
        for result in results:
            score = 0.0
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            content = f"{title} {snippet}"
            
            for term in main_terms:
                if term.lower() in content:
                    score += 2.0
                    if term.lower() in title:
                        score += 1.0
            
            result['relevance_score'] = score
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_results
    
    def search(self, query: str, force_disambiguation: bool = False) -> Dict[str, Any]:
        cached = self._get_cached(query)
        if cached and not force_disambiguation:
            return cached
        
        query_analysis = self.analyzer.analyze_query(query)
        
        queries_to_search = [query]
        if query_analysis['disambiguation_needed']:
            for suggestion in query_analysis.get('suggested_queries', [])[:1]:
                suggested_q = suggestion.get('query', '')
                if suggested_q and suggested_q != query.lower():
                    queries_to_search.append(suggested_q)
        
        all_results = []
        all_summaries = []
        
        for q in queries_to_search[:2]:
            summary, results = self._execute_search(q)
            all_summaries.append(summary)
            all_results.extend(results)
        
        seen_titles = set()
        unique_results = []
        for r in all_results:
            title = r.get('title', '').lower()[:50]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(r)
        
        validated_results = self._validate_results(unique_results, query_analysis)
        filtered_results = [r for r in validated_results if r.get('relevance_score', 0) > 0.5][:config.MAX_SEARCH_RESULTS]
        
        if filtered_results:
            final_summary = "\n\n".join([
                f"**{r.get('title', 'Resultado')}**\n{r.get('snippet', '')}"
                for r in filtered_results[:5]
            ])
        else:
            final_summary = all_summaries[0] if all_summaries else "Nenhum resultado encontrado."
        
        result = {
            'success': True,
            'query': query,
            'query_analysis': query_analysis,
            'summary': final_summary,
            'results': filtered_results,
            'all_results_count': len(unique_results),
            'disambiguation_performed': query_analysis['disambiguation_needed'],
            'source': 'smart_search',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._set_cache(query, result)
        return result

search_manager = SmartSearchManager()

# ============================================================================
# PROMPT E FORMATAÇÃO
# ============================================================================

SYSTEM_RESPONSE_STYLE = """
Você é um assistente técnico especializado que fornece respostas COMPLETAS, claras e bem formatadas.

REGRAS IMPORTANTES:
1. SEMPRE complete suas respostas - nunca pare no meio
2. Use quebras de linha para separar conceitos
3. Para listas, use marcadores (- ou *)
4. Para código, use blocos com ```linguagem
5. Seja didático e organizado
6. NÃO inclua "Usuário:", "Assistente:" na resposta

ESTRUTURA:
- Introdução breve
- Desenvolvimento completo
- Exemplo quando relevante
- Conclusão

CRÍTICO: Sempre termine suas respostas de forma completa.
""".strip()

def build_prompt_with_search(user_message: str, history_text: str, search_results: Optional[Dict] = None) -> str:
    parts = [SYSTEM_RESPONSE_STYLE, ""]
    
    if history_text:
        parts.append(f"--- HISTÓRICO ---\n{history_text}\n")
    
    if search_results and search_results.get('success'):
        search_context = f"""
--- INFORMAÇÕES DA WEB ---
Busca: "{search_results.get('query', '')}"

{search_results.get('summary', '')}

Fonte: Pesquisa web ({search_results.get('all_results_count', 0)} resultados)
---
"""
        parts.append(search_context)
    
    parts.append(f"--- PERGUNTA ---\n{user_message}\n\n--- RESPOSTA COMPLETA ---\n")
    
    return "\n".join(parts)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'(?i)^\s*(Usuário|Usuario|User|Assistente|Assistant|Resposta)\s*[:]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)(não continue|nao continue|---+|—+|_{3,})', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def sanitize_agent_message(text: str, word_limit: int = config.DEFAULT_MAX_WORDS) -> str:
    if not text:
        return ""
    
    text = clean_text(text)
    text = text.replace(config.STOP_TOKEN, "")
    
    words = text.split()
    if len(words) < config.MIN_RESPONSE_WORDS:
        return text.strip()
    
    if len(words) > word_limit:
        cut = ' '.join(words[:word_limit])
        last_para = cut.rfind('\n\n')
        if last_para > len(cut) * 0.7:
            text = cut[:last_para].strip()
        else:
            last_punct = -1
            for punct in ['. ', '! ', '? ']:
                pos = cut.rfind(punct)
                if pos > last_punct:
                    last_punct = pos + 1
            
            if last_punct > len(cut) * 0.6:
                text = cut[:last_punct].strip()
            else:
                text = cut.strip() + '...'
    
    return text.strip()

# ============================================================================
# HISTÓRICO
# ============================================================================

class HistoryManager:
    @staticmethod
    def get_file_path(session_id: str) -> str:
        return os.path.join(config.HISTORY_DIR, f"{session_id}.json")
    
    @staticmethod
    def load(session_id: str) -> List[Dict]:
        path = HistoryManager.get_file_path(session_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")
                return []
        return []
    
    @staticmethod
    def save(session_id: str, history: List[Dict]):
        try:
            path = HistoryManager.get_file_path(session_id)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar: {e}")
    
    @staticmethod
    def format_for_prompt(history: List[Dict], max_turns: int = 8) -> str:
        if not history:
            return ""
        recent = history[-max_turns:]
        parts = []
        for turn in recent:
            role = turn.get("role", "user")
            label = "Usuário" if role == "user" else "Assistente"
            content = turn.get("content", "").strip()
            if content:
                parts.append(f"{label}: {content}")
        return "\n\n".join(parts)

# ============================================================================
# LLM MANAGER (OLLAMA)
# ============================================================================

class LLMManager:
    def __init__(self):
        self.ollama_client = None
        self._lock = threading.Lock()
    
    def _load_ollama(self):
        if self.ollama_client is not None:
            return self.ollama_client
        
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama não está instalado!")
            return None
        
        try:
            self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
            logger.info(f"Ollama carregado: {config.OLLAMA_MODEL_NAME}")
            return self.ollama_client
        except Exception as e:
            logger.error(f"Erro ao conectar Ollama: {e}")
            return None
    
    def _is_response_complete(self, text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        
        if text.endswith(('.', '!', '?', ':', '"', "'", ')', ']', '```')):
            return True
        
        lines = text.split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line.startswith(('-', '*', '•')) and len(last_line) > 30:
                return True
        
        return False
    
    def generate_stream(self, prompt: str, temperature: float = 0.7) -> Generator:
        total_generated = ""
        continuation_count = 0
        
        client = self._load_ollama()
        if not client:
            yield {"error": "Ollama não disponível. Certifique-se de que o Ollama está rodando."}
            return
        
        while continuation_count < config.MAX_CONTINUATION_CALLS:
            current_prompt = prompt if continuation_count == 0 else prompt + total_generated
            chunk_text = ""
            
            try:
                response_stream = client.generate(
                    model=config.OLLAMA_MODEL_NAME,
                    prompt=current_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': config.MAX_GENERATION_CHUNK,
                        'stop': [config.STOP_TOKEN, '<|im_end|>', '</s>']
                    },
                    stream=True
                )
                
                for chunk in response_stream:
                    if 'response' in chunk:
                        chunk_text += chunk['response']
                        yield chunk['response']
                    if chunk.get('done', False):
                        break
            
            except Exception as e:
                logger.exception(f"Erro Ollama: {e}")
                yield {"error": f"Erro ao gerar resposta: {str(e)}"}
                return
            
            total_generated += chunk_text
            
            if self._is_response_complete(total_generated):
                break
            
            if len(total_generated) >= config.MAX_TOTAL_GENERATION:
                break
            
            if len(chunk_text.strip()) < 10:
                break
            
            continuation_count += 1

llm_manager = LLMManager()

# ============================================================================
# OCR (OPCIONAL)
# ============================================================================

EASY_OCR_READER = None

def get_easyocr_reader(lang_list=None, gpu=False):
    global EASY_OCR_READER
    if EASY_OCR_READER is not None:
        return EASY_OCR_READER
    try:
        import easyocr
        EASY_OCR_READER = easyocr.Reader(lang_list or ['pt', 'en'], gpu=gpu)
        logger.info("EasyOCR carregado")
    except Exception as e:
        logger.exception(f"Erro EasyOCR: {e}")
        EASY_OCR_READER = None
    return EASY_OCR_READER

# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

def require_rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        identifier = request.remote_addr or "unknown"
        if not rate_limiter.is_allowed(identifier):
            return jsonify({"error": "Rate limit excedido"}), 429
        return f(*args, **kwargs)
    return decorated

@require_rate_limit
def chat_endpoint():
    try:
        if request.method == 'POST':
            body = request.get_json() or {}
            user_message = body.get("message")
            session_id = body.get("session_id")
            enable_search = body.get("enable_search", True)
        else:
            user_message = request.args.get("message")
            session_id = request.args.get("session_id")
            enable_search = request.args.get("enable_search", "true").lower() == "true"
        
        user_message = validator.validate_message(user_message)
        
        if not session_id:
            session_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
        else:
            session_id = validator.validate_session_id(session_id)
        
        history = HistoryManager.load(session_id)
        history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        search_results = None
        if enable_search and search_manager.needs_search(user_message):
            logger.info(f"Buscando: {user_message[:50]}...")
            search_results = search_manager.search(user_message)
        
        history_text = HistoryManager.format_for_prompt(history, max_turns=8)
        final_prompt = build_prompt_with_search(user_message, history_text, search_results)
        
        def generate():
            full_response = ""
            last_heartbeat = time.time()
            
            if search_results:
                search_info = {
                    'search': {
                        'performed': True,
                        'source': search_results.get('source', 'web'),
                        'disambiguation': search_results.get('disambiguation_performed', False),
                        'results_count': search_results.get('all_results_count', 0)
                    }
                }
                yield f"data: {json.dumps(search_info)}\n\n"
            
            try:
                for chunk in llm_manager.generate_stream(final_prompt, temperature=0.6):
                    if isinstance(chunk, dict) and 'error' in chunk:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        return
                    
                    if not chunk:
                        if time.time() - last_heartbeat >= config.HEARTBEAT_INTERVAL:
                            yield ": keepalive\n\n"
                            last_heartbeat = time.time()
                        continue
                    
                    chunk_str = str(chunk)
                    if config.STOP_TOKEN in chunk_str:
                        chunk_str = chunk_str.split(config.STOP_TOKEN)[0]
                    
                    if chunk_str:
                        yield f"data: {json.dumps({'token': chunk_str})}\n\n"
                        full_response += chunk_str
            
            except Exception as e:
                logger.exception(f"Erro: {e}")
                yield f"data: {json.dumps({'error': 'Erro interno'})}\n\n"
            finally:
                agent_message = sanitize_agent_message(full_response)
                if not agent_message:
                    agent_message = "Desculpe, não consegui gerar uma resposta adequada."
                
                history.append({
                    "role": "agent",
                    "content": agent_message,
                    "timestamp": datetime.utcnow().isoformat(),
                    "search_performed": search_results is not None
                })
                
                HistoryManager.save(session_id, history)
                yield f"data: {json.dumps({'end': True, 'session_id': session_id, 'history': history})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"Erro: {e}")
        return jsonify({"error": "Erro interno"}), 500

@require_rate_limit
def search_endpoint():
    try:
        if request.method == 'POST':
            body = request.get_json() or {}
            query = body.get("query")
            force_disambiguation = body.get("force_disambiguation", False)
        else:
            query = request.args.get("query")
            force_disambiguation = request.args.get("force_disambiguation", "false").lower() == "true"
        
        if not query:
            return jsonify({"error": "Query não fornecida"}), 400
        
        query = validator.validate_message(query, max_length=500)
        results = search_manager.search(query, force_disambiguation=force_disambiguation)
        return jsonify(results)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"Erro na busca: {e}")
        return jsonify({"error": "Erro interno"}), 500

@require_rate_limit
def analyze_query_endpoint():
    try:
        body = request.get_json() or {}
        query = body.get("query")
        
        if not query:
            return jsonify({"error": "Query não fornecida"}), 400
        
        query = validator.validate_message(query, max_length=500)
        analysis = QueryAnalyzer.analyze_query(query)
        return jsonify(analysis)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"Erro: {e}")
        return jsonify({"error": "Erro interno"}), 500

@require_rate_limit
def find_similar_endpoint():
    try:
        body = request.get_json() or {}
        term = body.get("term")
        
        if not term:
            return jsonify({"error": "Termo não fornecido"}), 400
        
        similar = QueryAnalyzer.find_similar_terms(term)
        return jsonify({
            "term": term,
            "similar_terms": similar,
            "has_potential_confusion": len(similar) > 0
        })
    
    except Exception as e:
        logger.exception(f"Erro: {e}")
        return jsonify({"error": "Erro interno"}), 500

def get_history(session_id):
    try:
        session_id = validator.validate_session_id(session_id)
        history = HistoryManager.load(session_id)
        return jsonify({"session_id": session_id, "history": history})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

def list_histories():
    try:
        items = []
        for fname in os.listdir(config.HISTORY_DIR):
            if not fname.endswith('.json'):
                continue
            
            session_id = fname[:-5]
            path = os.path.join(config.HISTORY_DIR, fname)
            mtime = os.path.getmtime(path)
            
            snippet = ""
            try:
                history = HistoryManager.load(session_id)
                for msg in history:
                    if msg.get('content'):
                        snippet = msg.get('content', '')[:160]
                        break
            except Exception:
                pass
            
            items.append({
                "session_id": session_id,
                "snippet": snippet,
                "mtime": mtime
            })
        
        items.sort(key=lambda x: x['mtime'], reverse=True)
        return jsonify(items)
    except Exception as e:
        logger.exception(f"Erro: {e}")
        return jsonify({"error": "Erro interno"}), 500

@require_rate_limit
def api_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "Campo 'image' não encontrado"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Arquivo inválido"}), 400
    
    try:
        img_bytes = file.read()
        reader = get_easyocr_reader(['pt', 'en'], False)
        
        if reader is None:
            return jsonify({"error": "EasyOCR não disponível"}), 500
        
        try:
            from PIL import Image
            import numpy as np
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(pil_img)
            results = reader.readtext(img_np, detail=0)
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            try:
                results = reader.readtext(tmp_path, detail=0)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        
        text = "\n".join([r.strip() for r in results if r and r.strip()])
        
        if not text:
            return jsonify({"text": "", "warning": "Nenhum texto detectado"})
        
        return jsonify({"text": text})
    
    except Exception as e:
        logger.exception(f"Erro OCR: {e}")
        return jsonify({"error": "Erro ao processar"}), 500

def status():
    return jsonify({
        "status": "online",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "ollama_available": OLLAMA_AVAILABLE,
        "llm_provider": "ollama",
        "model": config.OLLAMA_MODEL_NAME,
        "search_enabled": True,
        "timestamp": datetime.utcnow().isoformat()
    })

def list_known_terms():
    return jsonify({
        "terms": list(QueryAnalyzer.KNOWN_SIMILAR_TERMS.keys()),
        "total": len(QueryAnalyzer.KNOWN_SIMILAR_TERMS)
    })

@require_rate_limit
def add_known_term():
    try:
        body = request.get_json() or {}
        term = body.get("term", "").lower().strip()
        alternatives = body.get("alternatives", [])
        
        if not term:
            return jsonify({"error": "Termo não fornecido"}), 400
        
        if not alternatives or not isinstance(alternatives, list):
            return jsonify({"error": "Alternativas inválidas"}), 400
        
        QueryAnalyzer.KNOWN_SIMILAR_TERMS[term] = [a.lower().strip() for a in alternatives]
        
        return jsonify({
            "success": True,
            "term": term,
            "alternatives": QueryAnalyzer.KNOWN_SIMILAR_TERMS[term]
        })
    
    except Exception as e:
        logger.exception(f"Erro: {e}")
        return jsonify({"error": "Erro interno"}), 500

# ============================================================================
# EXPORTAR PARA APP.PY
# ============================================================================

# Estas funções serão importadas pelo app.py
__all__ = [
    'chat_endpoint',
    'search_endpoint',
    'analyze_query_endpoint',
    'find_similar_endpoint',
    'get_history',
    'list_histories',
    'api_ocr',
    'status',
    'list_known_terms',
    'add_known_term'
]

if __name__ == '__main__':
    logger.info("=== Chatbot Backend (Ollama) ===")
    logger.info(f"Modelo: {config.OLLAMA_MODEL_NAME}")
    logger.info(f"Ollama: {'Disponível' if OLLAMA_AVAILABLE else 'Não disponível'}")
    logger.info(f"LangChain: {'Disponível' if LANGCHAIN_AVAILABLE else 'Não disponível'}")
