import React from 'react';
import { Sidebar } from '../components/Sidebar/Sidebar.jsx';
import { Header } from '../components/Header/Header.jsx';
import { ChatArea } from '../components/Chat/ChatArea.jsx';
import { ChatInput } from '../components/ChatInput/ChatInput.jsx';
import { UploadModal } from '../components/DatasetUpload/UploadModal.jsx';

export function ChatPage() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 antialiased">
      <Sidebar />

      <div className="flex-1 flex flex-col h-full min-w-0">
        <Header />
        <main className="flex-1 flex flex-col min-h-0 relative">
          <ChatArea />
          <ChatInput />
        </main>
      </div>

      <UploadModal />
    </div>
  );
}

export default ChatPage;
