import { FC, useState } from "react";
import ChatWindow from "./ChatWindow";
import Closer from "../Closer";
import Navbar from "../navbar/Navbar";

const ChatPage: FC = () => {
  const [mobileSettingsOpen, setMobileSettingsOpen] = useState<boolean>(false);
  
  return (
    <div className="w-full">
      <Navbar />
      <div className=" bg-regular h-max-available flex">
        <button
          className={`md:hidden absolute rounded-lg p-3 flex z-10 ${!mobileSettingsOpen ? 'bg-black text-white top-5 right-5' : 'bg-white text-black top-1 right-1'}`}
          onClick={() => {
            setMobileSettingsOpen((prev) => !prev);
          }}
        >
          <Closer />
        </button>
        <ChatWindow />
      </div>
    </div>
  );
};

export default ChatPage;