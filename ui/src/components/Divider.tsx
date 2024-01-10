import React from "react"

export interface DividerProps {
    containerClassName: string;
    text: string;
  }

const Divider : React.FC<DividerProps> = (props)=> {

    const {text,containerClassName}= props 

    return(
        <div className={`${containerClassName}`}>
            <div className="flex items-center">
                <div className="flex-1 border-t border-c-gray"></div>
                <div className="mx-4 text-darker-gray">{text}</div>
                <div className="flex-1 border-t border-c-gray"></div>
            </div>
        </div>
    )
}

export default Divider