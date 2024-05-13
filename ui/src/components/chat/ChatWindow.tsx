import React, { useEffect, useRef, useState } from "react";
import MessagesContainer from "./MessagesContainer";
import InputBar from "./InputBar";
import { fetchAnalyseTweetsSummaryMessage } from "../../services/api";
import { EventSourceMessage } from "@microsoft/fetch-event-source";
import { Message } from "../../types/Message";

const ChatWindow: React.FC = () => {
  const [uids, setUids] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const texts = useRef<{ role: string; text: string }[]>([]);

  const [inputEnabled, setInputEnabled] = useState<boolean>(true);

  const controller = useRef<AbortController>(new AbortController());

  // useEffect(() => {
  //   const fetchData = async () => {
  //     try {
  //     } catch (error) {
  //       console.error('Error fetching session data:', error);
  //     }
  //   };

  //   fetchData();
  // }, []);

  const handleSendMessage = (newMessage: string, variations: number) => {
    texts.current = [];
    setMessages((prevState) => [
      ...prevState,
      { author: "user", text: newMessage, type: "text" },
    ]);
    setMessages((prevState) => [
      ...prevState,
      {
        author: "assistant",
        text: "",
        type: "text-loading",
      },
    ]);
  };

  useEffect(() => {
    const onopen = (res: Response) => {
      if (res.ok && res.status === 200) {
        setMessages((prevState) => [
          ...prevState.slice(0, -1),
          {
            author: "assistant",
            text: "",
            type: "text",
          },
        ]);
        setInputEnabled(false);
      } else {
        setMessages((prevState) => [
          ...prevState.slice(0, -1),
          {
            author: "assistant",
            text: "Can't generate text. Please try again",
            type: "error",
          },
        ]);
      }
    };

    const onmessage = (event: EventSourceMessage) => {
      let data: { type: string; content: string; role?: string };

      try {
        data = JSON.parse(event.data);
      } catch (error) {
        data = { type: "text", content: event.data, role: "" };
      }

      if (data.type === "text") {
        const role = data.role as string;

        const textItem = texts.current.find((text) => text.role === role);

        if (textItem) {
          textItem.text += data.content as string;
        } else {
          texts.current.push({ role, text: data.content as string });
        }

        const text = texts.current.reduce((acc, text) => acc + text.text, "");

        setMessages((prevState) => [
          ...prevState.slice(0, -1),
          {
            ...prevState[prevState.length - 1],
            text,
          },
        ]);
      } else if (data.type === "tweets") {
        console.log("Tweets: ", data);
      } else if (data.type === "search") {
        console.log("Search results: ", data);
      }
    };

    const onerror = (err: any) => {
      if (messages[messages.length - 1]?.type === "text-loading") {
        setMessages((prevState) => [
          ...prevState.slice(0, -1),
          {
            author: "assistant",
            text: "Error during text generation. Please try again",
            type: "error",
          },
        ]);
      } else {
        setMessages((prevState) => [
          ...prevState,
          {
            author: "assistant",
            text: "Error during text generation. Please try again",
            type: "error",
          },
        ]);
      }
      throw new Error();
    };

    const onclose = () => {
      setInputEnabled(true);
      // const pathSegments = window.location.pathname.split('/');

      // setState((prevState) => ({
      //   ...prevState,
      //   session: session_id,
      //   isClosed: true
      // }));
    };

    if (
      messages[messages.length - 1]?.author === "assistant" &&
      messages[messages.length - 1]?.type === "text-loading"
    ) {
      let messagesToSend = [];
      messagesToSend.push({
        role: "user",
        content: messages[messages.length - 2].text,
        // tools: ['Twitter Search', "Google Search"],
        tools: ['Twitter Search'],
      });

      fetchAnalyseTweetsSummaryMessage(
        messagesToSend,
        onopen,
        onmessage, //selectedType === 'twitter' ? onmessage : onmessage,
        onerror,
        onclose,
        controller.current.signal,
        uids
      ).catch(() => setInputEnabled(true));
    }
  }, [messages]);

  return (
    <div className="flex-1 p:2 mr-2 sm:p-6 justify-between flex flex-col h-full w-full relative">
      <MessagesContainer
        messages={messages}
        onSendMessage={handleSendMessage}
      />
      <InputBar
        onSendMessage={handleSendMessage}
        enabled={inputEnabled}
        setEnabled={setInputEnabled}
        controller={controller}
        uids={uids}
        setUids={setUids}
      />
    </div>
  );
};

export default ChatWindow;
