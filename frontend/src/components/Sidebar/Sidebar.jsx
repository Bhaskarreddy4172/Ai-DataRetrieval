import React, { useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { Plus, Database, MessageSquare, Edit2, Trash2, Check, X, FileText, Search } from 'lucide-react';
import { Button } from '../Common/Button.jsx';

export function Sidebar() {
  const {
    sessions,
    activeSessionId,
    createNewSession,
    selectSession,
    renameSession,
    deleteSession,
    activeDataset,
    rowCount,
    columnCount,
    samples,
    switchSample,
    isSidebarOpen,
    toggleSidebar,
  } = useApp();

  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const handleStartRename = (e, session) => {
    e.stopPropagation();
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const handleSaveRename = (e, id) => {
    e.stopPropagation();
    renameSession(id, editTitle);
    setEditingId(null);
  };

  const handleCancelRename = (e) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const handleDelete = (e, id) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      deleteSession(id);
    }
  };

  // Filter and group sessions by recency (Today, Yesterday, Older)
  const groupedSessions = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = query
      ? sessions.filter((s) => (s.title || 'New Conversation').toLowerCase().includes(query))
      : sessions;

    const today = [];
    const yesterday = [];
    const older = [];

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000;

    filtered.forEach((session) => {
      const ts = session.timestamp || Date.now();
      if (ts >= startOfToday) {
        today.push(session);
      } else if (ts >= startOfYesterday) {
        yesterday.push(session);
      } else {
        older.push(session);
      }
    });

    return [
      { label: 'Today', items: today },
      { label: 'Yesterday', items: yesterday },
      { label: 'Older Conversations', items: older },
    ].filter((group) => group.items.length > 0);
  }, [sessions, searchQuery]);

  if (!isSidebarOpen) return null;

  return (
    <aside
      aria-label="Chatbot sidebar navigation"
      className="w-72 h-full flex flex-col bg-slate-50 dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 shrink-0 select-none transition-all duration-200"
    >
      {/* Top Action: New Chat */}
      <div className="p-3 border-b border-slate-200/60 dark:border-slate-800">
        <Button
          onClick={createNewSession}
          variant="outline"
          aria-label="Start new conversation"
          className="w-full justify-start gap-2 bg-white dark:bg-slate-800 shadow-sm border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700"
        >
          <Plus className="w-4 h-4 text-slate-600 dark:text-slate-300" />
          <span className="font-medium text-slate-800 dark:text-slate-200">+ New Chat</span>
        </Button>
      </div>

      {/* Dataset Section */}
      <div className="p-3 border-b border-slate-200/60 dark:border-slate-800">
        <div className="flex items-center gap-1.5 px-2 mb-2 text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
          <Database className="w-3.5 h-3.5" />
          <span>Datasets</span>
        </div>

        {/* Active Dataset Pill */}
        <div
          role="status"
          aria-label={`Active dataset: ${activeDataset}, ${rowCount} rows, ${columnCount} columns`}
          className="p-2.5 rounded-lg bg-blue-50/70 dark:bg-blue-950/40 border border-blue-200/70 dark:border-blue-900/50 mb-2"
        >
          <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-900 dark:text-blue-300 mb-0.5">
            <Check className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400 shrink-0" />
            <span className="truncate">{activeDataset}</span>
          </div>
          <div className="text-[11px] text-blue-600/80 dark:text-blue-400/80 font-medium pl-5">
            {rowCount} rows • {columnCount} columns
          </div>
        </div>

        {/* Sample Datasets Switcher List */}
        {samples && samples.length > 0 && (
          <div className="space-y-1">
            <div className="px-2 py-1 text-[11px] font-medium text-slate-400 dark:text-slate-500">
              Sample Datasets
            </div>
            {samples.map((s) => {
              const isSelected = activeDataset.toLowerCase().includes(s.id.toLowerCase());
              return (
                <button
                  key={s.id}
                  onClick={() => switchSample(s.id)}
                  aria-label={`Switch dataset to ${s.name}`}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    isSelected
                      ? 'bg-slate-200/70 dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/50 dark:hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <span className="truncate">{s.name}</span>
                  </div>
                  {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-blue-600 shrink-0"></span>}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Chat History Section with Search & Date Grouping */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Search Conversations */}
        <div className="relative mb-2">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            aria-label="Search conversation history"
            className="w-full pl-8 pr-2.5 py-1.5 text-xs bg-slate-200/50 dark:bg-slate-800/60 border border-transparent focus:border-blue-400 dark:focus:border-blue-500 rounded-lg text-slate-800 dark:text-slate-200 placeholder-slate-400 focus:outline-none"
          />
        </div>

        {groupedSessions.length === 0 ? (
          <div className="text-center py-4 text-xs text-slate-400">
            No matching conversations
          </div>
        ) : (
          groupedSessions.map((group) => (
            <div key={group.label} className="space-y-1">
              <div className="px-2 text-[11px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                {group.label}
              </div>

              {group.items.map((s) => {
                const isActive = s.id === activeSessionId;
                const isEditing = editingId === s.id;

                return (
                  <div
                    key={s.id}
                    onClick={() => selectSession(s.id)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Select chat: ${s.title || 'New Conversation'}`}
                    aria-current={isActive ? 'page' : undefined}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') selectSession(s.id);
                    }}
                    className={`group relative flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium cursor-pointer transition-colors focus:outline-none focus:ring-1 focus:ring-blue-400 ${
                      isActive
                        ? 'bg-slate-200 dark:bg-slate-800 text-slate-900 dark:text-slate-100'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-2">
                      <MessageSquare className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                      {isEditing ? (
                        <input
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          className="w-full px-1 py-0.5 text-xs bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded focus:outline-none"
                          autoFocus
                        />
                      ) : (
                        <span className="truncate">{s.title || 'New Conversation'}</span>
                      )}
                    </div>

                    {/* Action Icons */}
                    <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                      {isEditing ? (
                        <>
                          <button
                            onClick={(e) => handleSaveRename(e, s.id)}
                            aria-label="Confirm rename"
                            className="p-1 text-emerald-600 hover:text-emerald-700 dark:text-emerald-400"
                          >
                            <Check className="w-3 h-3" />
                          </button>
                          <button
                            onClick={handleCancelRename}
                            aria-label="Cancel rename"
                            className="p-1 text-slate-400 hover:text-slate-500"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={(e) => handleStartRename(e, s)}
                            aria-label="Rename conversation"
                            className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                            title="Rename chat"
                          >
                            <Edit2 className="w-3 h-3" />
                          </button>
                          <button
                            onClick={(e) => handleDelete(e, s.id)}
                            aria-label="Delete conversation"
                            className="p-1 text-slate-400 hover:text-red-600 dark:hover:text-red-400"
                            title="Delete chat"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
