import React from "react";
import { Message } from "../../types/Message";
import ImageLoader from "../loader/ImageLoader";
import MarkdownRenderer from "../markdown/MarkdownRenderer";

interface MessageItemProps {
  message: Message;
}

const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  let contentElem;
  switch (message.type) {
    case "text":
      if (message.text === null) {
        contentElem = <p className="break-words text-red">Can't Generate Text</p>
        break;
      }
      contentElem = <MarkdownRenderer markdownContent={message.text} author={message.author} />
      break;
    case "image":
      if (message?.images && message.images.length > 0) {
        contentElem = (
          <div className="flex flex-wrap justify-center gap-4">
            {message.images.map((img, index) => (
              <img key={index} src={img.url} alt={`Preview ${index + 1}`} className="mt-4 shadow-2xl rounded max-w-full h-auto align-middle border-none lg:max-w-[calc(50%-1rem)]" />
            ))}
          </div>
        );
      } else {
        contentElem = <p className="break-words text-red">Can't Generate Image</p>;
      }
      break;


    case "image-loading":

      contentElem = <div className="flex flex-1 justify-center py-20"><ImageLoader /></div>

      break;
    case "text-loading":
      contentElem = <div className="w-1/2 h-40vh bg-gray-200 animate-pulse" />
      break;
    case "error":
      contentElem = (
        <p className="break-words text-red">{message.text}</p>
      );
  }

  return (
    <div>
      <div className="flex flex-col items-start w-full justify-stretch p-2 pb-4 mb-2 text-primary">
        <div className="flex items-center">
          <div>
            <img src={message.author === 'assistant' ? "/black_t.png" : localStorage.getItem("user_info") ? `${JSON.parse(localStorage.getItem("user_info")!).photoURL!}` : "/user-solid.svg"} alt="My profile"
              className="w-4 h-4 rounded-full" />
          </div>
          <span className="ml-3 font-bold text-md">
            {message.author === 'assistant' ? 'Assistant' : 'User'}
          </span>
        </div>
        <div className="flex w-full text-xs">
          {contentElem}
        </div>
      </div>
    </div>
  );
};

export default MessageItem;