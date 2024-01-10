import React, { FC, ReactElement, AnchorHTMLAttributes, useState } from 'react';
import ReactMarkdown, { Components } from 'react-markdown';
import CodeBlock from './CodeBlock';
import { CopyIcon, TickIcon } from '../icons';

interface MarkdownRendererProps {
  markdownContent: string;
  author: string;
}

interface AnchorProps extends AnchorHTMLAttributes<HTMLAnchorElement> {
  node?: any; // Now 'node' is optional
  code?: any;
}

const MarkdownRenderer: FC<MarkdownRendererProps> = ({ markdownContent, author }) => {
  const [copy, setCopy] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(markdownContent);
    setCopy(true);
    const timeoutId = setTimeout(() => {
      setCopy(false);
      clearTimeout(timeoutId);
    }, 1000);
  };

  const renderers: Components = {
    code: CodeBlock,
    a: ({ node, ...props }: AnchorProps) => (
      <a {...props} style={{ color: 'blue' }} target="_blank" rel="noopener noreferrer" />
    ),
  };

  return (
    <div className="px-4">
      <ReactMarkdown components={renderers}>
        {markdownContent}
      </ReactMarkdown>
      {author && author === 'assistant' ? (
        !copy ? (
          <button className='mt-3' onClick={handleCopy}>
            <CopyIcon fillColor="var(--svg-primary)" />
          </button>
        ) : (
          <div className='mt-3'>
            <TickIcon fillColor="var(--svg-primary)" />
          </div>
        )
      ) : null}
    </div>
  );
};

export default MarkdownRenderer;