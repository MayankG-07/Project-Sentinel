# Project Sentinel 🛡️

> **Enterprise Intelligence. Absolute Privacy.**

Project Sentinel is a high-security, hybrid AI system that combines **Retrieval-Augmented Generation (RAG)** with **Natural Language to SQL** capabilities. Designed for air-gapped environments and privacy-conscious enterprises, it allows users to query both unstructured documents (PDFs) and structured databases (SQL) through a single, secure interface.

---

## 🚀 Key Features

### 🧠 Hybrid Intelligence
- **Dual-Engine Brain:** Seamlessly switches between a **ChromaDB Vector Store** (for documents) and **SQL Databases** (for structured data).
- **Intelligent Routing:** Uses an advanced LLM-based router to determine the best data source for every query.
- **Visual Page Rendering:** Don't just read the text—see the original document. Sentinel can render and display high-quality images of specific PDF pages directly in the chat.
- **AI PDF Generation:** Transform AI insights into professional documents. Sentinel can generate formal PDF reports based on your queries and data.

### 🔒 Enterprise-Grade Security
- **100% Private:** Designed to run entirely offline using **Ollama**. Your data never leaves your infrastructure.
- **Zero-Persistence Architecture:** Sentinel is a stateless system. It does not store chat history or learn from user interactions, ensuring absolute data isolation between sessions.
- **PII Redaction:** Integrated with **Microsoft Presidio** to automatically scrub sensitive information (emails, phone numbers) before AI processing.
- **RBAC (Role-Based Access Control):** Granular security enforcing user roles via API key authentication.
- **SQL Safety Loop:** Structural analysis of all generated SQL to block destructive commands and ensure read-only access.
- **Aggressive Auto-Cleanup:** All rendered images and generated reports are automatically deleted from the server after 10 minutes to maintain data privacy.

### 💻 Professional User Experience
- **Modern Split-Screen UI:** A sleek, dark-mode interface featuring a dedicated document viewer and interactive chat.
- **Perfectly Margined Visuals:** Rendered document pages are displayed with professional styling, including shadows, borders, and rounded corners.
- **Secure Login:** Dedicated authentication page with API key visibility toggles.

---

## 🛠️ Tech Stack

- **Frontend:** React, Vite, Material-UI (MUI)
- **Backend:** FastAPI, Python 3.11
- **AI Orchestration:** LangChain
- **LLM Engine:** Ollama (Llama 3)
- **Vector Database:** ChromaDB
- **SQL Database:** PostgreSQL (Supabase) / SQLite
- **Document Processing:** PyMuPDF (Rendering), FPDF2 (Generation)
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

## 🔑 Access Control Matrix

The system uses Role-Based Access Control (RBAC) to ensure data privacy. Below are the default credentials configured in `auth.py`:

| User | API Key | Assigned Roles | Data Access Level |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin_key` | `admin`, `finance`, `legal` | **Full Access:** SQL Database + PDF Documents |
| **Alice** | `finance_key` | `finance` | **Restricted:** SQL Database Only |
| **Bob** | `legal_key` | `legal` | **Restricted:** PDF Documents Only |

---

## 🤝 Contributing
This project is currently proprietary. Contributions are welcome for review, but the owners retain all rights to the code.

---

## 📄 License
This project is proprietary and confidential. All rights reserved by Mayank Goyal and Yashi Kulshresth. See the [LICENSE](LICENSE) file for details.
