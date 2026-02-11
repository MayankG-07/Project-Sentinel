# Project Sentinel: Judge Presentation Strategy 🏆

Follow these steps to ensure a smooth, fast, and impressive demo on a slow PC.

---

## 1. Pre-Presentation Checklist (The "Warm-up")
*   **Pull the Model:** Before the demo, ensure you have the fast model:
    ```sh
    ollama pull phi3
    ```
*   **Prime the Memory:** Ask the AI 2-3 questions *before* the judges arrive. This loads the model into your RAM so the first "real" question isn't slow.
*   **Power Mode:** Ensure your laptop is plugged in and set to **"Best Performance"** in Windows Power Settings.
*   **Close Background Apps:** Close Chrome tabs, Spotify, or any other heavy apps to free up RAM for Ollama.

## 2. The "Speed" Strategy
*   **Use Phi-3:** We have switched from Llama 3 to Phi-3. It is 3x faster on slow hardware while remaining very intelligent.
*   **Short Questions:** Ask direct questions. Longer questions take more time for the AI to "read."
*   **The "Talk-Through":** While the AI is thinking, use that time to explain the **Security Layer** or the **Hybrid Architecture**. Never let there be "dead silence."

## 3. High-Impact Demo Questions
Ask these specific questions to show off the best features:
1.  **Visual Proof:** "Show me page 1 of the bot document." (Shows rendering).
2.  **Hybrid Logic:** "List all employees and tell me if Mayank Goyal is one of them." (Shows SQL + RAG).
3.  **Security:** "What is the highest salary?" (Shows SQL Bridge).
4.  **Privacy:** "What did I just ask you?" (Shows Zero-Persistence).

## 4. Post-Presentation (Production Readiness)
After the demo, to make the project "Production Ready":
1.  **Switch back to Llama 3:** Change `.env` back to `MODEL_NAME=llama3` for higher reasoning quality.
2.  **Enable Streaming:** Implement word-by-word text generation for a premium feel.
3.  **GPU Instance:** Move the backend to a server with an NVIDIA GPU.

---
**Confidential: For Founder Use Only**
