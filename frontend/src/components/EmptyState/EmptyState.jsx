import React from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { Sparkles, Upload, MessageSquare } from 'lucide-react';
import { Button } from '../Common/Button.jsx';

export function EmptyState() {
  const { activeDataset, rowCount, columnCount, suggestions, sendMessage, openUploadModal } = useApp();

  const defaultPrompts = [
    'Who has the highest salary?',
    'How many employees work in Hyderabad?',
    'Show employees from Engineering.',
    'What is the average salary?',
  ];

  const prompts = suggestions && suggestions.length > 0 ? suggestions : defaultPrompts;

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto select-none animate-fade-in">
      <div className="p-3.5 rounded-2xl bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 mb-4 ring-1 ring-blue-100 dark:ring-blue-900/40">
        <Sparkles className="w-8 h-8" />
      </div>

      <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
        AI Dataset Assistant
      </h2>

      <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 max-w-md">
        Ask any natural-language question about <span className="font-semibold text-slate-700 dark:text-slate-300">{activeDataset}</span> ({rowCount} rows • {columnCount} columns).
      </p>

      <div className="flex items-center gap-3 mb-8">
        <Button onClick={openUploadModal} variant="outline" size="sm" icon={Upload}>
          + Upload Dataset
        </Button>
      </div>

      {/* Suggested Prompts */}
      <div className="w-full text-left">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-3">
          <MessageSquare className="w-3.5 h-3.5" />
          <span>Suggested Questions</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {prompts.slice(0, 4).map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => sendMessage(prompt)}
              className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/60 text-left text-xs text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-blue-300 dark:hover:border-blue-700 transition-all group"
            >
              <div className="font-medium group-hover:text-blue-600 dark:group-hover:text-blue-400">
                "{prompt}"
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

