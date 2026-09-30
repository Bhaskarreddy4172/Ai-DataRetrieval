import React from 'react';
import { Sparkles } from 'lucide-react';

export function LoadingIndicator() {
  return (
    <div className="flex items-start gap-3 max-w-3xl mx-auto px-4 py-3 animate-fade-in">
      <div className="p-1.5 rounded-lg bg-blue-600 text-white shrink-0 mt-0.5">
        <Sparkles className="w-4 h-4" />
      </div>
      <div className="flex items-center gap-1.5 px-4 py-3 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 text-xs">
        <span className="font-medium text-slate-700 dark:text-slate-300">AI Dataset Assistant is thinking</span>
        <div className="flex items-center gap-1 ml-1">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce"></span>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce [animation-delay:0.2s]"></span>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce [animation-delay:0.4s]"></span>
        </div>
      </div>
    </div>
  );
}

