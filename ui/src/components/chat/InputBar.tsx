import { FC, useState, useEffect, useRef } from "react";
import Closer from "../Closer";

interface InputBarProps {
  onSendMessage: (message: string, variations: number) => void;
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  controller: React.MutableRefObject<AbortController>;
}

const InputBar: FC<InputBarProps> = ({
  onSendMessage,
  enabled,
  setEnabled,
  controller,
}) => {
  const [message, setMessage] = useState("");
  const [variations, setVariations] = useState(4);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim()) {
      onSendMessage(message.trim(), variations);
      setMessage("");
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (inputRef.current && enabled) {
      inputRef.current.focus();
    }
  }, [enabled]);

  return (
    <div className="bottom-0 left-0 right-0 flex m-4 mx-6 bg-slate-100">
      <textarea
        rows={message.split("\n").length}
        className="w-full focus:outline-none focus:placeholder-gray-400 placeholder-gray-600 pl-4 bg-slate-100 text-lg py-3 resize-none"
        value={message}
        ref={inputRef}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a message..."
        disabled={!enabled}
      />
      {enabled ? (
        <button onClick={handleSend} className="text-inherit">
          <img src="/arrow-up-solid.svg" alt="send" className="w-6 h-6 rounded-full mr-2" />
        </button>
      ) : (
        <button
          onClick={() => {
            controller.current.abort();
            controller.current = new AbortController();
            setEnabled(true);
          }}
          className="text-inherit mr-2"
        >
          <Closer />
        </button>
      )}
    </div>
  );
};

export default InputBar;
