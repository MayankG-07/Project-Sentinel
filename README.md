# Project Sentinel 🛡️

> **Enterprise Intelligence. Absolute Privacy.**

Project Sentinel is a high-security, hybrid AI system that combines **Retrieval-Augmented Generation (RAG)** with **Natural Language to SQL** capabilities. Designed for air-gapped environments and privacy-conscious enterprises, it allows users to query both unstructured documents (PDFs) and structured databases (SQL) through a single, secure interface.

---

## 🚀 Key Features

### 🧠 Hybrid Intelligence
- **Dual-Engine Brain:** Seamlessly switches between a **ChromaDB Vector Store** (for documents) and **SQL Databases** (for structured data).
- **Intelligent Routing:** Uses an advanced LLM-based router to determine the best data source for every query.
- **Visual Page Rendering:** Don't just read the text—see the original document. Sentinel can render and display high-quality images of specific PDF pages directly in the chat.

### 🔒 Enterprise-Grade Security
- **100% Private:** Designed to run entirely offline using **Ollama**. Your data never leaves your infrastructure.
- **PII Redaction:** Integrated with **Microsoft Presidio** to automatically scrub sensitive information (emails, phone numbers) before AI processing.
- **RBAC (Role-Based Access Control):** Granular security enforcing user roles (`admin`, `finance`, `legal`) via API key authentication.
- **SQL Safety Loop:** Structural analysis of all generated SQL to block destructive commands and ensure read-only access.
- **Forensic Auditing:** Detailed `audit.log` records every query, user, and data source for compliance.

### 💻 Professional User Experience
- **Modern Split-Screen UI:** A sleek, dark-mode interface featuring a dedicated document viewer and interactive chat.
- **Secure Login:** Dedicated authentication page with API key visibility toggles.
- **Source Verification:** Every answer is backed by "Verified Sources," showing you exactly where the data came from.

---

## 🛠️ Tech Stack

- **Frontend:** React, Vite, Material-UI (MUI)
- **Backend:** FastAPI, Python 3.11
- **AI Orchestration:** LangChain
- **LLM Engine:** Ollama (Llama 3)
- **Vector Database:** ChromaDB
- **SQL Database:** PostgreSQL (Supabase) / SQLite
- **Deployment:** Docker, Docker Compose

---

## 🚦 Getting Started

### Prerequisites
- **Docker & Docker Compose**
- **Ollama:** Installed and running on the host machine.
  ```sh
  ollama pull llama3
  ```

### Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/MayankG-07/Project-Sentinel.git
   cd Project-Sentinel
   ```

2. **Configure Environment:**
   Create a `.env` file in the root directory based on the provided `.env.example`.
   ```sh
   # Example .env entry for Live SQL
   SQL_CONNECTION_STRING=postgresql://user:pass@IP_ADDRESS:5432/postgres
   ```

3. **Ingest Documents:**
   Place your PDFs in the `Data/` folder and vectorize them:
   ```sh
   docker-compose run --rm backend python ingest.py
   ```

4. **Launch:**
   ```sh
   docker-compose up --build
   ```

---

## 🔑 Access Control

| User | API Key | Access Level |
| :--- | :--- | :--- |
| **Administrator** | `admin_key` | Full Access (SQL + RAG) |
| **Finance** | `finance_key` | SQL Database Only |
| **Legal** | `legal_key` | PDF Documents Only |

---

## 🤝 Contributing
Project Sentinel is built with a focus on security and modularity. Contributions that enhance privacy or add support for new data sources are welcome.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
