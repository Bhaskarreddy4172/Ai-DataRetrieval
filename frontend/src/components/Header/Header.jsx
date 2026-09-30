import React from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { Menu, Moon, Sun, Upload, Download, Sparkles } from 'lucide-react';
import { Button } from '../Common/Button.jsx';
import { chatApi } from '../../api/chatApi.js';

export function Header() {
  const {
    isDark,
    toggleTheme,
    activeDataset,
    rowCount,
    columnCount,
    childCount,
    openUploadModal,
    toggleSidebar,
    isSidebarOpen,
    activeSessionId,
  } = useApp();

  const handleDownload = (format) => {
    const url = chatApi.getDownloadUrl(activeSessionId, format);
    window.open(url, '_blank');
  };

  return (
    <header
      role="banner"
      className="h-14 px-3 sm:px-4 flex items-center justify-between bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 shrink-0 select-none"
    >
      {/* Left branding */}
      <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
        <button
          onClick={toggleSidebar}
          aria-label={isSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-expanded={isSidebarOpen}
          className="p-1.5 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/30"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 shrink-0">
          <div className="p-1 rounded-md bg-blue-600 text-white shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <span className="font-semibold text-sm text-slate-900 dark:text-slate-100 hidden xs:inline">
            AI Dataset Assistant
          </span>
        </div>

        {/* Active Dataset Pill */}
        <div
          role="status"
          aria-label={`Current active dataset is ${activeDataset}`}
          className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-[11px] sm:text-xs font-medium text-slate-600 dark:text-slate-300 truncate max-w-[180px] sm:max-w-none"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0"></span>
          <span className="font-semibold truncate">{activeDataset}</span>
          <span className="text-slate-400 dark:text-slate-500 hidden sm:inline">•</span>
          <span className="hidden sm:inline">{rowCount} rows</span>
          <span className="text-slate-400 dark:text-slate-500 hidden md:inline">•</span>
          <span className="hidden md:inline">{columnCount} cols</span>
          {childCount > 0 && (
            <>
              <span className="text-slate-400 dark:text-slate-500 hidden lg:inline">•</span>
              <span className="hidden lg:inline text-blue-600 dark:text-blue-400 font-semibold">{childCount} child datasets</span>
            </>
          )}
        </div>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={openUploadModal}
          icon={Upload}
          aria-label="Upload dataset"
          className="hidden sm:inline-flex text-xs"
        >
          Upload Dataset
        </Button>

        {/* Download Transcript */}
        <button
          onClick={() => handleDownload('json')}
          aria-label="Download conversation transcript as JSON"
          className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/30"
          title="Export chat transcript (JSON)"
        >
          <Download className="w-4 h-4" />
        </button>

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/30"
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
        </button>
      </div>
    </header>
  );
}
