import React, { useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { EmptyState } from '../EmptyState/EmptyState.jsx';
import { MessageItem } from '../Message/MessageItem.jsx';
import { LoadingIndicator } from '../LoadingIndicator/LoadingIndicator.jsx';

export function ChatArea() {
  const { messages, isLoading } = useApp();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (!messages || messages.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="flex-1 overflow-y-auto py-4 space-y-2">
      {messages.map((msg) => (
        <MessageItem key={msg.id} message={msg} />
      ))}
      {isLoading && <LoadingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}

