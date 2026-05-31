import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, Brain, MessageSquare } from 'lucide-react';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import AIMessageCard from '../components/chat/AIMessageCard'; // Import the new AI message card
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid'; // For generating session IDs

// Define types for messages (simplified for JS)
// In a TS project, these would come from backend/api/schemas.ts
const USER_MESSAGE_TYPE = 'user';
const AI_MESSAGE_TYPE = 'ai';
const AI_THINKING_TYPE = 'thinking';

const ChatWorkspacePage = () => {
  const [inputPrompt, setInputPrompt] = useState('');
  const [messages, setMessages] = useState([]); // Stores { type: 'user' | 'ai', content: string | SentinelResponse, id: string }
  const [isSending, setIsSending] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate a new session ID when the component mounts
    setSessionId(uuidv4());
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputPrompt.trim() || isSending) return;

    const userMessage = { type: USER_MESSAGE_TYPE, content: inputPrompt, id: uuidv4() };
    setMessages((prev) => [...prev, userMessage]);
    setInputPrompt('');
    setIsSending(true);

    // Add an AI thinking message immediately
    const thinkingMessageId = uuidv4();
    setMessages((prev) => [...prev, { type: AI_THINKING_TYPE, content: 'Thinking...', id: thinkingMessageId }]);

    try {
      const apiKey = localStorage.getItem('api_key');
      if (!apiKey) {
        toast.error('API Key not found. Please log in again.');
        setIsSending(false);
        // Remove thinking message
        setMessages((prev) => prev.filter(msg => msg.id !== thinkingMessageId));
        return;
      }

      const response = await fetch('http://localhost:8000/v1/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
        body: JSON.stringify({ prompt: userMessage.content, session_id: sessionId }),
      });

      if (!response.ok) {
        const errorData = await response.headers.get('content-type')?.includes('application/json')
          ? await response.json()
          : { detail: `Server error: ${response.status} ${response.statusText}` };
        throw new Error(errorData.detail || 'Failed to get response from AI.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiResponseContent = {
        message: '',
        sql_query: null,
        table_data: null,
        sources: [],
      };
      let currentMessageId = uuidv4();

      // Remove the initial thinking message
      setMessages((prev) => prev.filter(msg => msg.id !== thinkingMessageId));

      // Add a new AI message card for streaming
      setMessages((prev) => [...prev, { type: AI_MESSAGE_TYPE, content: { ...aiResponseContent }, id: currentMessageId, isStreaming: true }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        // Split by event lines
        const eventLines = chunk.split('\n\n').filter(line => line.trim() !== '');

        for (const eventLine of eventLines) {
          const eventTypeMatch = eventLine.match(/^event: (\w+)/);
          const dataMatch = eventLine.match(/^data: (.*)/s); // 's' flag for single line mode to match across newlines

          if (eventTypeMatch && dataMatch) {
            const eventType = eventTypeMatch[1];
            const data = JSON.parse(dataMatch[1]);

            setMessages((prevMessages) => {
              const newMessages = [...prevMessages];
              const aiMessageIndex = newMessages.findIndex(msg => msg.id === currentMessageId);

              if (aiMessageIndex !== -1) {
                const currentAiMessage = newMessages[aiMessageIndex].content;

                switch (eventType) {
                  case 'metadata':
                    // Update metadata fields
                    newMessages[aiMessageIndex].content = {
                      ...currentAiMessage,
                      sql_query: data.sql_query || currentAiMessage.sql_query,
                      table_data: data.table_data || currentAiMessage.table_data,
                      sources: data.sources || currentAiMessage.sources,
                    };
                    break;
                  case 'message_chunk':
                    // Append LLM message chunks
                    newMessages[aiMessageIndex].content = {
                      ...currentAiMessage,
                      message: currentAiMessage.message + data.chunk,
                    };
                    break;
                  case 'message_end':
                    // Mark streaming as complete
                    newMessages[aiMessageIndex].isStreaming = false;
                    break;
                  // case 'complete': // Final event, can be used to finalize state
                  //   newMessages[aiMessageIndex].isStreaming = false;
                  //   break;
                  default:
                    break;
                }
              }
              return newMessages;
            });
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      toast.error(`Error: ${error.message}`);
      // Remove thinking message or mark AI message as error
      setMessages((prev) => prev.filter(msg => msg.id !== thinkingMessageId));
      setMessages((prev) => {
        const newMessages = [...prev];
        const lastAiMessageIndex = newMessages.findLastIndex(msg => msg.type === AI_MESSAGE_TYPE);
        if (lastAiMessageIndex !== -1) {
          newMessages[lastAiMessageIndex].content.message += `\n\n**Error:** ${error.message}`;
          newMessages[lastAiMessageIndex].isStreaming = false;
        } else {
          newMessages.push({ type: AI_MESSAGE_TYPE, content: { message: `**Error:** ${error.message}` }, id: uuidv4(), isStreaming: false });
        }
        return newMessages;
      });
    } finally {
      setIsSending(false);
      setMessages((prev) => prev.map(msg => msg.type === AI_MESSAGE_TYPE ? { ...msg, isStreaming: false } : msg));
    }
  };

  return (
    <motion.div
      className="flex h-full flex-col p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="mb-6 flex items-center text-3xl font-bold text-text-primary">
        <MessageSquare className="mr-3 h-8 w-8 text-primary-accent" /> AI Chat Workspace
      </h1>

      <div className="flex-1 overflow-y-auto rounded-xl bg-background-secondary p-4 shadow-inner-lg-glass border border-border">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-text-secondary/60">
            <p className="text-lg">Start a conversation with Project Sentinel...</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className="mb-4">
            {msg.type === USER_MESSAGE_TYPE ? (
              <div className="flex justify-end">
                <div className="glassmorphism max-w-xl rounded-lg p-3 shadow-md-glass bg-primary-accent/20 text-text-primary">
                  <p>{msg.content}</p>
                </div>
              </div>
            ) : msg.type === AI_MESSAGE_TYPE ? (
              <AIMessageCard
                message={msg.content.message}
                sqlQuery={msg.content.sql_query}
                tableData={msg.content.table_data}
                sources={msg.content.sources}
                isStreaming={msg.isStreaming}
              />
            ) : (
              <div className="flex items-center justify-start">
                <Brain className="mr-2 h-5 w-5 animate-pulse text-secondary-accent" />
                <p className="text-text-secondary">{msg.content}</p>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="mt-4 flex space-x-3">
        <Input
          type="text"
          placeholder="Ask Project Sentinel..."
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          className="flex-1 rounded-lg border border-border bg-surface p-3 text-text-primary focus:border-primary-accent focus:outline-none"
          disabled={isSending}
        />
        <Button type="submit" disabled={isSending} className="px-6 py-3">
          {isSending ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
        </Button>
      </form>
    </motion.div>
  );
};

export default ChatWorkspacePage;
