# ✨ Aether IA — Assistente Inteligente Multimodal

> **Aether IA** é uma plataforma avançada de Inteligência Artificial que combina um chatbot conversacional, reconhecimento óptico de caracteres (OCR) e um poderoso sistema de transcrição de áudio e vídeo.

---

## 📋 Sobre o Projeto

Desenvolvido em **Python com Flask**, o Aether IA foi projetado para ser uma ferramenta centralizada de produtividade. Ele permite que usuários interajam com modelos de linguagem, extraiam textos de documentos físicos e convertam mídias audiovisuais em texto editável de forma simples e eficiente.

---

## 🚀 Funcionalidades Principais

### 💬 Chatbot com IA
*   **Conversação Inteligente**: Interface web moderna para diálogos fluidos.
*   **Integração com Ollama**: Suporte a modelos como Llama 3.2 para processamento local.
*   **Histórico Persistente**: Conversas são salvas automaticamente na pasta `chat_history/`.

### 🔍 Reconhecimento de Texto (OCR)
*   **Extração de Documentos**: Transforme imagens (JPG, PNG) em texto digital.
*   **Processamento em Lote**: Suporte para múltiplos uploads e organização de resultados.

### 🎥 Transcrição de Vídeo e Áudio (Nova!)
*   **Suporte Multimídia**: Transcreve arquivos de vídeo (`.mp4`, `.mkv`, `.mov`, `.avi`) e áudio (`.mp3`, `.wav`, `.m4a`).
*   **Integração com YouTube**: Basta colar a URL do vídeo para que o sistema baixe o áudio e realize a transcrição automaticamente.
*   **Tecnologia Whisper (OpenAI)**: Utiliza o modelo `base` do Whisper para garantir alta precisão e suporte a múltiplos idiomas com detecção automática.
*   **Exportação**: Gera arquivos de texto estruturados com cabeçalhos e metadados da transcrição.

---

## 🗂️ Estrutura do Projeto

```text
aether_ia/
├── app.py                        # Servidor principal e roteamento global
├── chatbot_backend.py            # Lógica do Chatbot e integração IA
├── ocr_backend.py                # Processamento de imagens (OCR)
├── video_transcription_backend.py # Lógica de transcrição de vídeo/áudio (Whisper)
├── requirements.txt              # Dependências do projeto
├── templates/                    # Interfaces HTML (index, chatbot, vídeo)
├── static/                       # CSS, JS e assets da interface
├── uploads/                      # Armazenamento temporário de mídias enviadas
├── outputs/                      # Resultados das transcrições e OCR
└── chat_history/                 # Histórico das conversas do chatbot
```

---

## 🛠️ Tecnologias Utilizadas

| Camada | Tecnologia |
| :--- | :--- |
| **Backend** | Python 3.8+ / Flask |
| **IA Conversacional** | Ollama (Llama 3.2) / Transformers |
| **Transcrição** | OpenAI Whisper / MoviePy / Pytubefix |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Logs** | Sistema de logging nativo do Python |

---

## ⚙️ Instalação e Execução

### Pré-requisitos
*   Python 3.8 ou superior.
*   [FFmpeg](https://ffmpeg.org/) instalado no sistema (necessário para processamento de vídeo/áudio).
*   [Ollama](https://ollama.ai/) (opcional, para o módulo de Chatbot local).

### Passo a Passo

1.  **Clone o repositório**:
    ```bash
    git clone <url-do-repositorio>
    cd aether_ia
    ```

2.  **Crie um ambiente virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Inicie a aplicação**:
    ```bash
    python app.py
    ```
    Acesse em: `http://localhost:5000`

---

## 📝 Logs e Monitoramento

O sistema mantém registros detalhados para facilitar a depuração:
*   `aether_ia.log`: Erros e eventos gerais da aplicação.
*   `transcription.log`: Histórico detalhado de cada transcrição de vídeo realizada, incluindo tempo de processamento e idioma detectado.

---
> ⚠️ **Nota**: Os diretórios de trabalho (`uploads/`, `outputs/`, `temp/`) são criados automaticamente na primeira execução.
