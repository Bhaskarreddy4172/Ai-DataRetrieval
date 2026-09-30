import React, { createContext, useContext, useState } from 'react';
import { useTheme } from '../hooks/useTheme.js';
import { useDataset } from '../hooks/useDataset.js';
import { useChat } from '../hooks/useChat.js';

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const themeState = useTheme();
  const datasetState = useDataset();
  const chatState = useChat();

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);

  const value = {
    ...themeState,
    ...datasetState,
    ...chatState,
    isUploadModalOpen,
    setIsUploadModalOpen,
    openUploadModal: () => setIsUploadModalOpen(true),
    closeUploadModal: () => setIsUploadModalOpen(false),
    isSidebarOpen,
    setIsSidebarOpen,
    toggleSidebar,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

