import { useState, useEffect, useCallback } from 'react';
import { chatApi } from '../api/chatApi.js';

const SESSIONS_KEY = 'chat_sessions_v2';

export function useChat() {
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem(SESSIONS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (e) {}
    return [{ id: 'default', title: 'New Conversation', timestamp: Date.now() }];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return sessions[0]?.id || 'default';
  });

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeModel, setActiveModel] = useState('llama3.1:8b');

  // Persist sessions array
  useEffect(() => {
    try {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    } catch (e) {}
  }, [sessions]);

  // Load history when active session changes
  const loadSessionHistory = useCallback(async (sessionId) => {
    try {
      const res = await chatApi.getHistory(sessionId);
      if (res.history && Array.isArray(res.history)) {
        const formatted = res.history.flatMap((turn, idx) => [
          {
            id: `user-${idx}`,
            sender: 'user',
            text: turn.question,
            timestamp: turn.timestamp,
          },
          {
            id: `ai-${idx}`,
            sender: 'assistant',
            text: turn.answer,
            operation: turn.operation,
            resultCount: turn.result_count,
            timestamp: turn.timestamp,
          },
        ]);
        setMessages(formatted);
      } else {
        setMessages([]);
      }
    } catch (err) {
      console.warn('Error loading chat history:', err);
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    if (activeSessionId) {
      loadSessionHistory(activeSessionId);
    }
  }, [activeSessionId, loadSessionHistory]);

  // Create new session
  const createNewSession = async () => {
    try {
      const res = await chatApi.newSession();
      const newId = res.session_id || `session-${Date.now()}`;
      const newSession = {
        id: newId,
        title: 'New Conversation',
        timestamp: Date.now(),
      };
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newId);
      setMessages([]);
      return newId;
    } catch (err) {
      console.error('Error creating new session:', err);
      const fallbackId = `session-${Date.now()}`;
      const fallbackSession = {
        id: fallbackId,
        title: 'New Conversation',
        timestamp: Date.now(),
      };
      setSessions((prev) => [fallbackSession, ...prev]);
      setActiveSessionId(fallbackId);
      setMessages([]);
      return fallbackId;
    }
  };

  // Switch session
  const selectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  // Rename session
  const renameSession = (sessionId, newTitle) => {
    if (!newTitle.trim()) return;
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, title: newTitle.trim() } : s))
    );
  };

  // Delete session
  const deleteSession = async (sessionId) => {
    try {
      await chatApi.clearChat(sessionId);
    } catch (e) {}

    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== sessionId);
      if (filtered.length === 0) {
        const defaultSess = { id: 'default', title: 'New Conversation', timestamp: Date.now() };
        setActiveSessionId('default');
        return [defaultSess];
      }
      if (activeSessionId === sessionId) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  // Send question
  const sendMessage = async (questionText) => {
    if (!questionText || !questionText.trim()) return;

    const userMsgId = `user-${Date.now()}`;
    const userMsg = {
      id: userMsgId,
      sender: 'user',
      text: questionText.trim(),
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    // Update session title if default
    const currentSession = sessions.find((s) => s.id === activeSessionId);
    if (currentSession && (currentSession.title === 'New Conversation' || !currentSession.title)) {
      const truncated = questionText.trim().substring(0, 30);
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, title: truncated } : s))
      );
    }

    try {
      const res = await chatApi.processQuery(questionText.trim(), activeSessionId);
      
      const aiMsg = {
        id: `ai-${Date.now()}`,
        sender: 'assistant',
        text: res.answer,
        operation: res.operation,
        results: res.results || [],
        aggregation: res.aggregation,
        resultCount: res.result_count,
        processingTime: res.processing_time,
        grounded: res.grounded !== false,
        isClarification: res.is_clarification || false,
        debugTrace: res.debug_trace,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg = {
        id: `ai-err-${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ **Error processing request**: ${err.message || 'Something went wrong while processing your question. Please try again.'}`,
        isError: true,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear current active session messages
  const clearCurrentChat = async () => {
    try {
      await chatApi.clearChat(activeSessionId);
    } catch (e) {}
    setMessages([]);
  };

  return {
    sessions,
    activeSessionId,
    messages,
    isLoading,
    activeModel,
    setActiveModel,
    sendMessage,
    createNewSession,
    selectSession,
    renameSession,
    deleteSession,
    clearCurrentChat,
  };
}

