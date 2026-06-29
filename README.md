# 🤖 DevOps RAG Assistant

A production-ready AI chatbot that answers DevOps questions using Retrieval-Augmented Generation (RAG). Built with LangChain, Ollama, ChromaDB, and Streamlit — deployed on AWS EC2 with Nginx and HTTPS.

🌐 **Live Demo:** [https://rag.jibindevops.in](https://rag.jibindevops.in)

---

## 🏗️ Architecture
## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (Llama 3.2) — runs fully local, zero API cost |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Frontend | Streamlit |
| Web Server | Nginx (reverse proxy) |
| SSL | Certbot (Let's Encrypt) |
| Hosting | AWS EC2 t3.large |
| Process Manager | systemd |

## 📚 Knowledge Base

The assistant answers questions about:
- 🐳 Docker — containers, images, Dockerfile, Docker Compose
- ☸️ Kubernetes — pods, deployments, services, kubectl commands
- 🔧 Jenkins — CI/CD pipelines, Jenkinsfile, webhooks
- 🏗️ Terraform — IaC, providers, state management, AWS provisioning
- 🔄 ArgoCD — GitOps, sync, drift detection, rollbacks

## 🚀 How It Works

1. **Indexing phase** — DevOps documents are split into chunks, converted to vector embeddings using `nomic-embed-text`, and stored in ChromaDB
2. **Query phase** — User question is embedded, top-3 similar chunks are retrieved from ChromaDB, combined with the question into a prompt, and passed to Llama 3.2 for a grounded answer

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/Jibin-G-Jiji/devops-rag-assistant.git
cd devops-rag-assistant

# Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure
## 👨‍💻 Author

**Jibin G Jiji** — DevOps Engineer  
🌐 [jibindevops.in](https://jibindevops.in) | 💻 [GitHub](https://github.com/Jibin-G-Jiji)
