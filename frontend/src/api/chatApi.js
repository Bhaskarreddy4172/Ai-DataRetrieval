import { request } from './client.js';

export const chatApi = {
  /**
   * Process a natural language question on the active dataset.
   */
  async processQuery(question, session_id = 'default') {
    return request('/chat', {
      method: 'POST',
      body: JSON.stringify({ question, session_id }),
    });
  },

  /**
   * Clear chat history for a session or all sessions.
   */
  async clearChat(session_id = 'default') {
    return request('/clear-chat', {
      method: 'POST',
      body: JSON.stringify({ session_id }),
    });
  },

  /**
   * Initialize a new conversation session.
   */
  async newSession() {
    return request('/new-session', {
      method: 'POST',
    });
  },

  /**
   * Get dialogue turns for a session.
   */
  async getHistory(session_id = 'default') {
    return request(`/history?session_id=${encodeURIComponent(session_id)}`, {
      method: 'GET',
    });
  },

  /**
   * Download conversation transcript as JSON or TXT.
   */
  getDownloadUrl(session_id = 'default', format = 'json') {
    return `/download-chat?session_id=${encodeURIComponent(session_id)}&format=${format}`;
  },

  /**
   * System health and Ollama status check.
   */
  async getHealth() {
    return request('/health', {
      method: 'GET',
    });
  },
};

