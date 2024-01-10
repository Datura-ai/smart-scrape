import { FC } from "react";

interface TemplateProps {
  onMessageSubmit: (message: string, variations: number) => void;
  containerClassName?: string;
}

const MessageContainerTemplate: FC<TemplateProps> = ({ onMessageSubmit, containerClassName }) => {

  const submitQuestion = (index: number) => {
    onMessageSubmit(defaultQuestions[index].trim(), 4)
  }

  const defaultQuestions = [
    "#RecipeTrends: What are the hottest culinary creations taking over Twitter this year?", 
    "How are Bittensor and Crypto spicing up 2024's recipes?", 
    "What's the latest buzz in recipes this year?", 
    "#Bitcoin: How are latest Bitcoin trends shaping investment and tech strategies?"
  ]

  return (
    <div className={`flex flex-col h-full justify-end items-center ${containerClassName || ""
      }`}>
      <div className="w-15 h-15 bg-white rounded-full flex items-center justify-center">
        <img src="/images/logo.png" alt="taotensor" />
      </div>
      <div className="w-full flex justify-center text-sm text-primary my-2">How can I help you today ?</div>
      <div className="flex w-full mt-5 flex-wrap justify-center">
        {defaultQuestions.map((question, index) => (
          <div key={index} onClick={() => submitQuestion(index)} className="cursor-pointer flex bg-logo-circle shadow-md w-74 mx-4 my-1 p-3 rounded-xl hover:bg-slate-600 transition duration-300 ease-in-out">
            <div>
              <p className="m-0 p-0 text-sm text-primary">{question}</p>
            </div>
            <div className="hidden">Might place button on hover here later</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MessageContainerTemplate;
