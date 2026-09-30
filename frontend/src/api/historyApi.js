import { chatApi } from './chatApi.js';

export const historyApi = {
  /**
   * Fetch full conversation history.
   */
  async loadHistory(sessionId) {
    try {
      const data = await chatApi.getHistory(sessionId);
      return data.history || [];
    } catch (err) {
      console.error('Failed to load history:', err);
      return [];
    }
  },

  /**
   * Clear session history.
   */
  async clearHistory(sessionId) {
    return chatApi.clearChat(sessionId);
  },
};

