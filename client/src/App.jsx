import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const Source = ({ source }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const fileName = source.source.split(/[\\/]/).pop(); // Extract filename

  return (
    <div className="source">
      <div className="source-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span>📄 {fileName}</span>
        <span>{isExpanded ? '▲' : '▼'}</span>
      </div>
      {isExpanded && (
        <div className="source-content">
          <p>{source.content}</p>
        </div>
      )}
    </div>
  );
};

function App() {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const [apiKey, setApiKey] = useState('');
  const [user, setUser] = useState(null); // New state to hold user info
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: "/// SENTINEL OS v1.0 ONLINE ///\n\nSystem Status: SECURE CONNECTION REQUIRED\n\nPlease enter your API Key and press CONNECT.",
      sources: [],
    }
  ]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  const handleConnect = async () => {
    const trimmedApiKey = apiKey.trim();
    if (!trimmedApiKey) {
      const errorMsg = { role: 'ai', text: "⚠️ AUTHENTICATION FAILURE: API Key is required.", sources: [] };
      setMessages(prev => [...prev, errorMsg]);
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post(
        `${API_URL}/auth/verify`,
        {},
        { headers: { 'X-API-Key': trimmedApiKey } }
      );
      setUser(res.data);
      const welcomeMsg = { role: 'ai', text: `Welcome, ${res.data.username}. Secure enclave established. You may now enter your commands.`, sources: [] };
      setMessages(prev => [...prev, welcomeMsg]);
    } catch (error) {
      let text = "⚠️ AUTHENTICATION FAILURE: The provided API Key is invalid.";
      if (error.response && error.response.status !== 401) {
        text = `⚠️ ERROR ${error.response.status}: ${error.response.data.detail || 'An unknown error occurred.'}`;
      }
      const errorMsg = { role: 'ai', text, sources: [] };
      setMessages(prev => [...prev, errorMsg]);
    }
    setLoading(false);
  };

  const handleSearch = async () => {
    const trimmedApiKey = apiKey.trim();
    if (!query.trim() || !user) return;

    const userMsg = { role: 'user', text: query, sources: [] };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setLoading(true);

    try {
      const res = await axios.post(
        `${API_URL}/chat`,
        { text: userMsg.text },
        { headers: { 'X-API-Key': trimmedApiKey } }
      );
      const { response, sources } = res.data;
      const aiMsg = { role: 'ai', text: response, sources: sources || [] };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      let text = `⚠️ NETWORK ERROR: Verify server is running on ${API_URL}.`;
      if (error.response) {
        if (error.response.status === 403) {
          text = "🚫 AUTHORIZATION FAILURE: You do not have permission for this query.";
        } else {
          text = `⚠️ ERROR ${error.response.status}: ${error.response.data.detail || 'An unknown error occurred.'}`;
        }
      }
      const errorMsg = { role: 'ai', text, sources: [] };
      setMessages(prev => [...prev, errorMsg]);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (user) {
        handleSearch();
      }
    }
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <>
      {/* HEADER */}
      <header className="header">
        <div className="brand">
          <span>🛡️</span>
          PROJECT SENTINEL
        </div>
        <div className="api-key-wrapper">
          <input
            type="password"
            id="apiKey"
            name="apiKey"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter API Key..."
            disabled={!!user}
          />
          <button onClick={handleConnect} disabled={!!user || loading}>
            {loading && !user ? 'CONNECTING...' : 'CONNECT'}
          </button>
        </div>
      </header>

      {/* TERMINAL OUTPUT */}
      <div className="chat-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <span className="sender-name">
              {msg.role === 'user' ? `>> ${user?.username || 'OPERATOR_01'}` : '>> SENTINEL_CORE'}
            </span>
            <div className="msg-bubble">
              {msg.text}
              {msg.role === 'ai' && (
                <button className="copy-button" onClick={() => handleCopy(msg.text)}>
                  📋
                </button>
              )}
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources-container">
                <span className="sources-title">Verified Sources:</span>
                {msg.sources.map((source, idx) => (
                  <Source key={idx} source={source} />
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message ai">
            <span className="sender-name">>> SENTINEL_CORE</span>
            <div className="msg-bubble" style={{ color: '#94a3b8' }}>
              Processing secure query...
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* INPUT CONSOLE */}
      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            id="query"
            name="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={user ? "Enter command..." : "Awaiting secure connection..."}
            disabled={!user || loading}
          />
          <button onClick={handleSearch} disabled={!user || loading}>
            {loading ? 'BUSY' : 'EXECUTE'}
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
