import React, {useState} from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { a11yDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { CopyIcon } from '../icons';
import { TickIcon } from '../icons';

const CodeBlock: React.FC<any> = (props) => {
  const [copy, setCopy] = useState(false);
  const handleCopy = (content:string)=> {
    setCopy(true);
    navigator.clipboard.writeText(content);
    const timeoutId = setTimeout(() => {
      setCopy(false)
      clearTimeout(timeoutId);
    }, 1000); 
  }
  const codeContent = props.children || '';
  const codeLines = codeContent.split('\n');
  const isSingleLine = codeLines.length === 1;

  if (isSingleLine) {
    return <code>{props.children}</code>;
  }

  return (
    <div className='p-0 relative'>
      {!copy ? ( <button className='absolute  top-2 right-2'>
        <CopyIcon fillColor="white" handleClick={()=>handleCopy(props.children)} />
      </button>):(<div className='absolute  top-2 right-2'><TickIcon fillColor="white"/></div>)}
      <SyntaxHighlighter language={props && props.className && props.className.slice(9)} style={a11yDark} wrapLongLines>
        {props.children}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock;