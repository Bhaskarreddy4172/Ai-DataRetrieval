import React, { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { Send, Paperclip, Loader2, Sparkles } from 'lucide-react';

export function ChatInput() {
  const { sendMessage, isLoading, suggestions, openUploadModal } = useApp();
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  // Auto-grow textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [text]);

  const handleSend = () => {
    if (!text.trim() || isLoading) return;
    sendMessage(text.trim());
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSelectSuggestion = (prompt) => {
    sendMessage(prompt);
  };

  return (
    <div className="p-3 sm:p-4 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 shrink-0 select-none">
      <div className="max-w-3xl mx-auto space-y-2">
        {/* Dynamic Prompt Suggestion Chips */}
        {suggestions && suggestions.length > 0 && (
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar">
            <Sparkles className="w-3.5 h-3.5 text-blue-500 shrink-0" />
            {suggestions.slice(0, 4).map((sugg, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectSuggestion(sugg)}
                disabled={isLoading}
                className="px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs font-medium text-slate-600 dark:text-slate-300 whitespace-nowrap transition-colors border border-slate-200/60 dark:border-slate-700/60 disabled:opacity-50"
              >
                {sugg}
              </button>
            ))}
          </div>
        )}

        {/* Input Box Wrapper */}
        <div className="relative flex items-end gap-2 p-2 rounded-2xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 focus-within:border-blue-500 dark:focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all">
          <button
            type="button"
            onClick={openUploadModal}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-700 transition-colors shrink-0 mb-0.5"
            title="Upload dataset (CSV, XLSX, XLS, JSON)"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="Ask anything about your dataset..."
            className="flex-1 bg-transparent border-0 resize-none text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-0 max-h-40 py-1.5 px-0"
          />

          <button
            type="button"
            onClick={handleSend}
            disabled={!text.trim() || isLoading}
            className="p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:hover:bg-blue-600 transition-colors shrink-0 mb-0.5 shadow-sm"
            title="Send question (Enter)"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-400 px-1">
          <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for line breaks</span>
          <span>Zero Hallucination Guarantee</span>
        </div>
      </div>
    </div>
  );
}

