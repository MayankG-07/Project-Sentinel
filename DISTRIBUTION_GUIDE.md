# Project Sentinel: Enterprise Distribution & Deployment Guide 🛡️

This document outlines the professional delivery models, security standards, and technical requirements for deploying **Project Sentinel** within an enterprise infrastructure.

---

## 1. Deployment Models

Project Sentinel is designed for maximum flexibility, supporting three primary deployment architectures:

### A. On-Premise (Air-Gapped / High Security)
*   **Best For:** Government, Military, and Financial Institutions.
*   **Delivery:** Encrypted Docker Image Bundle (.tar).
*   **Mechanism:** The software runs entirely within the client's internal network with zero external internet requirements.
*   **Data Privacy:** 100% of data (PDFs and SQL) remains behind the client's firewall.

### B. Managed Cloud (SaaS)
*   **Best For:** Startups and SMEs.
*   **Delivery:** Dedicated Instance URL (e.g., `https://client.sentinel-ai.com`).
*   **Mechanism:** Hosted on secure, SOC2-compliant infrastructure (AWS/Azure) managed by the Sentinel team.
*   **Data Privacy:** Data is isolated in encrypted, client-specific containers.

### C. Hybrid Bridge
*   **Best For:** Modern Enterprises.
*   **Delivery:** Docker Compose Orchestration.
*   **Mechanism:** AI Engine runs locally for document privacy, while the SQL bridge connects to existing cloud databases (e.g., Supabase, RDS).

---

## 2. Security & Compliance Standards

Project Sentinel is built on a **Security-First** philosophy:

*   **Zero-Persistence Architecture:** No session data or chat history is stored after a page refresh or session timeout.
*   **PII Redaction:** Integrated Microsoft Presidio layer automatically scrubs sensitive data (Emails, Phone Numbers, SSNs) before processing.
*   **Forensic Auditing:** Every query and data access event is recorded in a tamper-evident log for compliance reviews.
*   **Read-Only SQL Bridge:** The AI is restricted to `SELECT` operations, preventing any accidental or malicious data modification.

---

## 3. Licensing & Commercial Model

We offer a tiered commercial structure designed for scalability:

| Tier | Model | Features |
| :--- | :--- | :--- |
| **Sentinel Core** | One-time License | Local RAG, Basic SQL Bridge, Standard UI |
| **Sentinel Pro** | Per User / Per Year | Core + Visual Rendering + AI Report Generation |
| **Sentinel Enterprise** | Custom Add-on | Pro + **Forensic Audit Logging** + Custom PII Rules |

---

## 4. Technical Requirements

### Server Requirements (Minimum)
*   **OS:** Linux (Ubuntu 22.04+), Windows 10/11 (with WSL2), or macOS.
*   **Runtime:** Docker & Docker Compose.
*   **AI Engine:** Ollama (Local) or dedicated GPU instance.
*   **Hardware:** 16GB RAM, 4-Core CPU (GPU recommended for faster rendering).

### Client Requirements
*   **Browser:** Any modern web browser (Chrome, Edge, Firefox).
*   **Network:** Access to the internal server IP on port 8080.

---

## 5. Delivery Process

1.  **Consultation:** Define data sources (PDF directories and SQL schemas).
2.  **Configuration:** Customization of PII rules and RBAC roles.
3.  **Provisioning:** Generation of unique Client License Keys.
4.  **Deployment:** Remote or on-site installation of Docker containers.
5.  **Validation:** Forensic log verification and security audit.

---

**Proprietary & Confidential**
© 2024 Mayank Goyal and Yashi Kulshresth. All Rights Reserved.
