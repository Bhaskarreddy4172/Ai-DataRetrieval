import React, { useState } from 'react';
import { Sparkles, User, Copy, Check, ThumbsUp, ThumbsDown, RotateCw, ChevronDown, ChevronRight } from 'lucide-react';
import { MarkdownRenderer } from '../MarkdownRenderer/MarkdownRenderer.jsx';
import { DataTable } from '../DataTable/DataTable.jsx';
import { Badge } from '../Common/Badge.jsx';
import { useApp } from '../../context/AppContext.jsx';

export function MessageItem({ message }) {
  const { sendMessage } = useApp();
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // 'up' | 'down'
  const [showTrace, setShowTrace] = useState(false);

  const isUser = message.sender === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRegenerate = () => {
    if (message.text) {
      // Find prompt or re-ask
      sendMessage(message.text);
    }
  };

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-3 max-w-3xl mx-auto px-4 py-3 group">
        <div className="px-4 py-2.5 rounded-2xl bg-blue-600 text-white text-sm max-w-[85%] shadow-sm">
          {message.text}
        </div>
        <div className="p-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 shrink-0 mt-0.5">
          <User className="w-4 h-4" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 max-w-3xl mx-auto px-4 py-4 group animate-fade-in">
      <div className="p-1.5 rounded-lg bg-blue-600 text-white shrink-0 mt-0.5 shadow-sm">
        <Sparkles className="w-4 h-4" />
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        {/* Answer Content */}
        <MarkdownRenderer content={message.text} />

        {/* Supporting Records Table */}
        {message.results && message.results.length > 0 && (
          <DataTable data={message.results} />
        )}

        {/* Message Metadata & Action Toolbar */}
        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-2">
            {message.grounded !== false && (
              <Badge variant="success" className="text-[10px]">
                ✓ Grounded
              </Badge>
            )}
            {message.operation && (
              <Badge variant="default" className="text-[10px] font-mono">
                {message.operation}
              </Badge>
            )}
            {message.processingTime && (
              <span className="text-[11px] text-slate-400 font-mono">
                {(message.processingTime * 1000).toFixed(0)}ms
              </span>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-1 opacity-80 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Copy answer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
            </button>

            <button
              onClick={() => setFeedback('up')}
              className={`p-1.5 rounded-md transition-colors ${
                feedback === 'up'
                  ? 'text-emerald-600 bg-emerald-50 dark:bg-emerald-950'
                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
              title="Helpful"
            >
              <ThumbsUp className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={() => setFeedback('down')}
              className={`p-1.5 rounded-md transition-colors ${
                feedback === 'down'
                  ? 'text-red-600 bg-red-50 dark:bg-red-950'
                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
              title="Not helpful"
            >
              <ThumbsDown className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Collapsible Pipeline Execution Trace */}
        {message.debugTrace && (
          <div className="pt-1">
            <button
              onClick={() => setShowTrace((prev) => !prev)}
              className="flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            >
              {showTrace ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <span>Pipeline Execution Trace</span>
            </button>

            {showTrace && (
              <pre className="mt-2 p-3 rounded-lg bg-slate-900 text-slate-300 font-mono text-[11px] overflow-x-auto max-h-48 border border-slate-800">
                {JSON.stringify(message.debugTrace, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

