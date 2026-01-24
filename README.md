# ProjectSentinel

This is the README for ProjectSentinel, a secure, local-first RAG (Retrieval-Augmented Generation) application designed to answer questions about your documents and structured data.

## Running with Docker (Recommended)

This project is fully containerized, which is the recommended way to run it.

### Prerequisites
*   Docker and Docker Compose installed.
*   An Ollama instance running with the `llama3` model pulled (`ollama pull llama3`).

### Instructions

1.  **Configure:** Copy the `.env.example` file to `.env` and configure it for your deployment mode (On-Premise or Cloud). See the "Deployment Modes" section below.

2.  **Ingest Data:** Place your PDF documents into the `Data/` directory. Then, run the one-time ingestion process:
    ```bash
    docker-compose run --rm backend python ingest.py
    ```

3.  **Launch the Application:** Start the frontend and backend services:
    ```bash
    docker-compose up --build
    ```

4.  **Access:** Open your web browser and navigate to `http://localhost:8080`. Enter your API key and press "Connect".

## Changelog

*   **Personalized Authentication Flow (Latest):** Implemented a more intuitive, two-step authentication process. The user now enters their API key and presses a "Connect" button. Upon successful validation, they receive a personalized welcome message with their username, and the chat interface becomes active.
*   **Hybrid Database Capability:** The application is now hybrid-ready, able to connect to either a local SQLite file or a cloud database via a connection string.
*   **LangChain Modernization:** Refactored the project to use the new modular LangChain packages, resolving all deprecation warnings.
*   **UI Refinement: Copy to Clipboard:** Added a "copy" button to AI responses in the UI.
*   **Enhanced SQL Safety Loop:** Upgraded the SQL validation to use `sqlparse` for robust structural analysis.
*   **Ingestion Performance Improvement:** Parallelized the embedding process to speed up vectorization.
*   **Source Verification:** The UI now displays the specific sources the AI used to generate an answer.
*   **Forensic Logging:** Implemented a dedicated audit logger to record all queries in `audit.log`.
*   **Role-Based Access Control (RBAC):** Implemented API key authentication and role-based enforcement.
*   **Health Check Endpoint:** Added a `/health` endpoint for monitoring.
*   **Structured Logging:** Replaced all `print()` statements with Python's `logging` module.
*   **Containerization:** Packaged the entire application into Docker containers.
*   **Configuration Management:** Externalized all hardcoded settings into a `.env` file.
*   **Initial Project Setup:** Created the initial `README.md` and project analysis.

## Deployment Modes
... (The rest of the deployment modes section remains the same)

## Security & Operations
... (The rest of the security sections remain the same)

## Project Structure
... (The rest of the project structure remains the same)

## Project Sentinel: Technical Summary
... (The rest of the summary remains the same)
