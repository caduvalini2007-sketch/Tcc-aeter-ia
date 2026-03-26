# ✨ Aether IA

> Assistente de Inteligência Artificial com interface web, chatbot conversacional e reconhecimento óptico de caracteres (OCR).

---

## 📋 Sobre o Projeto

**Aether IA** é uma aplicação web construída em Python com Flask que oferece uma interface conversacional com IA e funcionalidades de OCR. O projeto permite interações via chat com um modelo de linguagem e processamento de imagens/documentos para extração de texto.

---

## 🗂️ Estrutura do Projeto

```
aether_ia/
├── app.py                  # Aplicação principal Flask (rotas e servidor)
├── chatbot_backend.py      # Lógica do chatbot e integração com IA
├── ocr_backend.py          # Backend de OCR (reconhecimento de texto em imagens)
├── requirements.txt        # Dependências Python
├── templates/
│   ├── index.html          # Página principal da interface
│   ├── index4.html         # Variante da interface principal
│   └── error.html          # Página de erros
├── static/
│   └── uploads/            # Arquivos estáticos e uploads da interface
├── uploads/                # Diretório de arquivos enviados pelo usuário
├── outputs/                # Resultados processados
├── temp/                   # Arquivos temporários
├── chat_history/           # Histórico de conversas salvas
├── aether_ia.log           # Log da aplicação
└── transcription.log       # Log de transcrições OCR
```

---

## 🚀 Funcionalidades

- 💬 **Chatbot com IA** — Conversação inteligente via interface web
- 🔍 **OCR** — Extração de texto a partir de imagens e documentos enviados pelo usuário
- 📁 **Upload de arquivos** — Suporte ao envio de arquivos para processamento
- 💾 **Histórico de chat** — Armazenamento do histórico de conversas
- 🌐 **Interface web responsiva** — Frontend acessível pelo navegador

---

## 🛠️ Tecnologias Utilizadas

| Camada | Tecnologia |
|--------|-----------|
| Backend | Python + Flask |
| Frontend | HTML, CSS, JavaScript |
| IA / Chatbot | Integração via `chatbot_backend.py` |
| OCR | Integração via `ocr_backend.py` |
| Logs | Sistema de logging Python |

---

## ⚙️ Instalação e Execução

### Pré-requisitos

- Python 3.8+
- pip

### Passo a passo

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd aether_ia

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie a aplicação
python app.py
```

A aplicação estará disponível em `http://localhost:5000` (ou a porta configurada no `app.py`).

---

## 📦 Dependências

As dependências do projeto estão listadas em `requirements.txt`. Para instalá-las:

```bash
pip install -r requirements.txt
```

---

## 📁 Diretórios Gerados em Tempo de Execução

| Diretório | Descrição |
|-----------|-----------|
| `uploads/` | Arquivos enviados pelos usuários para processamento |
| `outputs/` | Resultados gerados (ex: textos extraídos via OCR) |
| `temp/` | Arquivos temporários durante o processamento |
| `chat_history/` | Histórico de conversas do chatbot |

> ⚠️ Esses diretórios são criados automaticamente pela aplicação. Não é necessário criá-los manualmente.

---

## 📝 Logs

- `aether_ia.log` — Log geral da aplicação (erros, requisições, eventos)
- `transcription.log` — Log específico das transcrições de OCR

---


