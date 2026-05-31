# Project Sentinel 🛡️

> **Enterprise Intelligence. Absolute Privacy.**

Project Sentinel is a high-security, hybrid AI system that combines **Retrieval-Augmented Generation (RAG)** with **Natural Language to SQL** capabilities. Designed for air-gapped environments and privacy-conscious enterprises, it allows users to query both unstructured documents (PDFs) and structured databases (SQL) through a single, secure interface.

---

## 🚀 Key Features

### 🧠 Hybrid Intelligence
- **Dual-Brain Architecture:** Sentinel utilizes a specialized two-stage reasoning process. A dedicated **Logic Engine** handles complex SQL generation and data retrieval, while a separate **Streaming Engine** synthesizes the final response for the user. This ensures 100% accuracy in data handling without sacrificing performance.
- **Real-Time Streaming:** Experience instant feedback with word-by-word text generation. Sentinel streams its "thoughts" directly to the UI, providing a premium, responsive user experience similar to industry-leading AI platforms.
- **Multi-Source Reasoning Engine:** Sentinel performs parallel analysis across both SQL databases and PDF documents for every query, synthesizing a unified, intelligent response that cross-references structured and unstructured data.
- **Visual Page Rendering:** Don't just read the text—see the original document. Sentinel can render and display high-quality images of specific PDF pages directly in the chat.
- **AI PDF Generation:** Transform AI insights into professional documents. Sentinel can generate formal PDF reports based on your queries and data.

### 🔒 Enterprise-Grade Security
- **100% Private:** Designed to run entirely offline using **Ollama**. Your data never leaves your infrastructure.
- **Zero-Persistence Architecture:** Sentinel is a stateless system. It does not store chat history or learn from user interactions, ensuring absolute data isolation between sessions.
- **Verbose Forensic Auditing:** Every interaction is recorded in a structured `forensic_audit.log`, capturing timestamps, user identities, AI-generated SQL, and data sources accessed for complete accountability.
- **PII Redaction:** Integrated with **Microsoft Presidio** to automatically scrub sensitive information (emails, phone numbers) before AI processing.
- **RBAC (Role-Based Access Control):** Granular security enforcing user roles via API key authentication.
- **SQL Safety Loop:** Structural analysis of all generated SQL to block destructive commands and ensure read-only access.
- **Aggressive Auto-Cleanup:** All rendered images and generated reports are automatically deleted from the server after 10 minutes to maintain data privacy.
- **Tenant Isolation (RAG):** Documents are tagged with `clearance_level` metadata during ingestion, and retrieval is filtered based on the authenticated user's clearance, ensuring users only access authorized information.

### 💻 Professional User Experience
- **Premium, Futuristic UI:** Completely redesigned frontend with a dark intelligence interface, glassmorphism, soft glows, and high-end animations, inspired by ChatGPT Enterprise, Palantir Foundry, and Microsoft Security Copilot.
- **Modern 3-Panel Layout:** A sophisticated workspace featuring a collapsible left sidebar for navigation, a central AI chat workspace, and a right context intelligence panel.
- **Structured AI Responses:** Chat messages are rendered in elegant analysis cards, intelligence containers, with support for markdown, code blocks, tabular data, and expandable source references.
- **Real-time Streaming UI:** Smooth streaming animations and AI thinking states provide a highly responsive and engaging user experience.
- **Secure Login:** Dedicated authentication page with API key visibility toggles and security trust badges.

---

## 🛠️ Tech Stack

- **Frontend:** React, Vite, Tailwind CSS, shadcn/ui, Framer Motion, Lucide React, next-themes, sonner
- **Backend:** FastAPI, Python 3.11, LangChain, SQLAlchemy, sqlglot, Presidio
- **AI Orchestration:** LangChain
- **LLM Engine:** Ollama (Llama 3)
- **Vector Database:** ChromaDB
- **SQL Database:** PostgreSQL (Supabase) / SQLite
- **Document Processing:** PyMuPDF (Rendering), FPDF2 (Generation)
- **Deployment:** Docker, Docker Compose

---

## 🚦 Getting Started (Local Development without Docker)

This guide will help you set up and run Project Sentinel locally without Docker, incorporating all the latest UI/UX and security enhancements.

### Prerequisites
- **Python 3.11+**: Installed on your system.
- **Node.js & npm**: Installed on your system.
- **Ollama**: Installed and running on your system. Download from [ollama.com/download](https://ollama.com/download).

### Installation Steps

1.  **Clone the Repository:**
    ```sh
    git clone https://github.com/MayankG-07/Project-Sentinel.git
    cd Project-Sentinel
    ```

2.  **Set up Ollama (AI Model Server):**
    *   Ensure Ollama is installed and running.
    *   Download the `llama3` model:
        ```sh
        ollama pull llama3
        ```

3.  **Set up Python Backend:**
    *   **Create & Activate Virtual Environment:**
        ```sh
        python -m venv .venv
        # On Windows:
        .venv\Scripts/activate
        # On macOS/Linux:
        source .venv/bin/activate
        ```
    *   **Install Python Dependencies:**
        ```sh
        pip install -r backend/requirements.txt
        ```
    *   **Configure Environment Variables:**
        Create a file named `.env` in the project root (`D:/Project-Sentinel/.env`) with the following content. **Ensure `SENTINEL_OLLAMA_HOST` is `http://localhost:11434`**.
        ```ini
        SENTINEL_DB_PATH=backend/data/chroma_db
        SENTINEL_MODEL_NAME=llama3
        SENTINEL_EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
        SENTINEL_OLLAMA_HOST=http://localhost:11434
        SENTINEL_DATA_PATH=backend/data/Data
        SENTINEL_CHUNK_SIZE=1000
        SENTINEL_CHUNK_OVERLAP=200
        # SENTINEL_SQL_CONNECTION_STRING=postgresql://user:password@host:port/database # Uncomment and configure for PostgreSQL/MySQL if needed
        DEFAULT_INGEST_CLEARANCE=confidential
        ```
    *   **Prepare Data Directory:**
        Ensure the directory `D:/Project-Sentinel/backend/data/Data` exists. Place any PDF documents you want the AI to process into this directory.
    *   **Ingest Data into ChromaDB:**
        This will process your PDFs and tag them with the `DEFAULT_INGEST_CLEARANCE` from your `.env`.
        ```sh
        python -m backend.engine.ingest
        ```
    *   **Run the Python Backend Server:**
        Keep this terminal open.
        ```sh
        python -m backend.api.server
        ```

4.  **Set up Frontend UI:**
    *   **Navigate to Client Directory:**
        ```sh
        cd client
        ```
    *   **Install Node.js Dependencies:**
        ```sh
        npm install
        ```
    *   **Initialize Tailwind CSS:**
        ```sh
        npm run tw-init
        ```
    *   **Manually Configure `shadcn/ui` (if CLI fails):**
        *   Create `client/jsconfig.json`:
            ```json
            {
              "compilerOptions": {
                "baseUrl": ".",
                "paths": {
                  "@/*": ["./src/*"]
                },
                "jsx": "react-jsx"
              },
              "include": ["src"]
            }
            ```
        *   Create `client/components.json`:
            ```json
            {
              "$schema": "https://ui.shadcn.com/schema.json",
              "style": "nova",
              "rsc": false,
              "tsx": false,
              "tailwind": {
                "config": "tailwind.config.js",
                "css": "src/index.css",
                "baseColor": "slate",
                "cssVariables": true
              },
              "aliases": {
                "components": "@/components",
                "utils": "@/lib/utils"
              }
            }
            ```
        *   **Manually Create `shadcn/ui` Components:**
            Create the directory `client/src/components/ui/` and then create the following files inside it with the content provided in the detailed change log above:
            *   `button.jsx`
            *   `input.jsx`
            *   `label.jsx`
            *   `checkbox.jsx`
            *   `accordion.jsx`
    *   **Run Frontend Development Server:**
        Keep this terminal open.
        ```sh
        npm run dev
        ```

5.  **Access the Application:**
    *   Open your web browser and navigate to the URL provided by `npm run dev` (usually `http://localhost:5173`).
    *   On the login page, use one of the API keys defined in `backend/core/auth.py`:
        *   `admin_key` (for user `admin`, clearance `top_secret`)
        *   `finance_key` (for user `alice`, clearance `confidential`)
        *   `legal_key` (for user `bob`, clearance `restricted`)

---

## 🔑 Access Control Matrix

The system uses Role-Based Access Control (RBAC) and Tenant Isolation to ensure data privacy. Below are the default credentials configured in `backend/core/auth.py`:

| User          | API Key       | Assigned Roles             | Data Access Level (Clearance) |
| :------------ | :------------ | :------------------------- | :---------------------------- |
| **Administrator** | `admin_key`   | `admin`, `finance`, `legal` | **Top Secret**                |
| **Alice**     | `finance_key` | `finance`                  | **Confidential**              |
| **Bob**       | `legal_key`   | `legal`                    | **Restricted**                |

---

## 🤝 Contributing
This project is currently proprietary. Contributions are welcome for review, but the owners retain all rights to the code.

---

## 📄 License
This project is proprietary and confidential. All rights reserved by Mayank Goyal and Yashi Kulshresth. See the [LICENSE](LICENSE) file for details.
