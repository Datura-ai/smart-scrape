import React, { useEffect, useRef, useState, useMemo } from "react";
import { Message } from "../../types/Message";
import MessageItem from "./MessageItem";
import MessageContainerTemplate from "./MessageContainerTemplate";

interface MessagesContainerProps {
  messages: Message[];
  onSendMessage: (message: string, variations: number) => void;
}
const MessagesContainer: React.FC<MessagesContainerProps> = ({ messages, onSendMessage }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleQuestionSubmit = (e: string, v: number) => {
    onSendMessage(e.trim(), v)
  }

  const [showScrollButton, setShowScrollButton] = useState(false);
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollTop = messagesEndRef.current.scrollHeight;
    }
  };

  const handleScroll = (event: React.UIEvent) => {

    if (!messagesEndRef.current) return;

    const isAtBottom =
      messagesEndRef.current.scrollHeight - messagesEndRef.current.clientHeight <=
      messagesEndRef.current.scrollTop + 1;
    setShowScrollButton(!isAtBottom);
  };

  const messagesToShow = useMemo(() => messages.filter((message) => message.author === "assistant" || message.author === "user"), [messages]);

  useEffect(() => {
    if (!showScrollButton)
      scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const messageIdFromUrl = urlParams.get('message_id');

    if (messageIdFromUrl) {
      const messageElement = document.getElementById(messageIdFromUrl);
      if (messageElement && messagesEndRef.current) {
        messagesEndRef.current.scrollTop = messageElement.offsetTop - 20;
      }

      setTimeout(() => {
        messageElement?.classList.add('highlighted-message')
      }, 300);

      setTimeout(() => {
        messageElement?.classList.remove('highlighted-message')
      }, 2000);

    }
  }, [window.location.search]);

  return (
    <div
      ref={messagesEndRef}
      onScroll={(event) => handleScroll(event)}
      className="relative h-full w-full px-3 overflow-y-auto scrollbar-thumb-blue scrollbar-thumb-rounded scrollbar-track-blue-lighter scrollbar-w-2"
    >
      <div className="overflow-auto h-full w-full flex-grow px-2.5">
        {messagesToShow.map((message, index) => (
          <div id={message.uuid} key={index}>
            <MessageItem message={message} />
          </div>
        ))}
        <MessageContainerTemplate containerClassName={messages && messages.length > 0 ? 'hidden' : ''} onMessageSubmit={handleQuestionSubmit} />
      </div>
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="sticky bottom-0 left-1/2 transform -translate-x-1/2"
        >
          <svg className="transform rotate-180" width={16} height={16} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"><path d="M214.6 41.4c-12.5-12.5-32.8-12.5-45.3 0l-160 160c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L160 141.2V448c0 17.7 14.3 32 32 32s32-14.3 32-32V141.2L329.4 246.6c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3l-160-160z" /></svg>        </button>
      )}
    </div>
  );
};

export default MessagesContainer;
