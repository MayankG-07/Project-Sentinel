import React from 'react';
import { motion } from 'framer-motion';
import Markdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'; // Dark theme for code blocks
import { FileText, Code, Table, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '../ui/accordion'; // shadcn/ui Accordion
import { Button } from '../ui/button'; // shadcn/ui Button
import { toast } from 'sonner'; // Assuming sonner for toasts

const AIMessageCard = ({ message, sqlQuery, tableData, sources, isStreaming }) => {
  const [showSql, setShowSql] = React.useState(false);
  const [showTable, setShowTable] = React.useState(false);
  const [copied, setCopied] = React.useState(false);

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success('Copied to clipboard!');
    setTimeout(() => setCopied(false), 2000);
  };

  const renderers = {
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="glassmorphism mb-6 rounded-xl p-6 shadow-lg-glass"
    >
      {/* AI Message Content */}
      <div className="prose prose-invert max-w-none text-text-primary">
        <Markdown components={renderers}>{message}</Markdown>
        {isStreaming && (
          <span className="ml-2 inline-block h-3 w-3 animate-pulse rounded-full bg-primary-accent"></span>
        )}
      </div>

      {/* SQL Query Display */}
      {sqlQuery && (
        <Accordion type="single" collapsible className="w-full mt-4 border-t border-border pt-4">
          <AccordionItem value="sql-query">
            <AccordionTrigger className="flex items-center justify-between text-text-secondary hover:text-primary-accent">
              <div className="flex items-center">
                <Code className="mr-2 h-5 w-5 text-secondary-accent" />
                <span className="font-semibold">SQL Query Executed</span>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="relative rounded-md bg-surface p-3 text-sm font-mono text-text-primary">
                <SyntaxHighlighter style={oneDark} language="sql" PreTag="div">
                  {sqlQuery}
                </SyntaxHighlighter>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-2 text-text-secondary hover:text-primary-accent"
                  onClick={() => handleCopy(sqlQuery)}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Table Data Display */}
      {tableData && tableData.length > 0 && (
        <Accordion type="single" collapsible className="w-full mt-4 border-t border-border pt-4">
          <AccordionItem value="table-data">
            <AccordionTrigger className="flex items-center justify-between text-text-secondary hover:text-primary-accent">
              <div className="flex items-center">
                <Table className="mr-2 h-5 w-5 text-success" />
                <span className="font-semibold">Tabular Data Results ({tableData.length} rows)</span>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="max-h-60 overflow-auto rounded-md border border-border">
                <table className="w-full table-auto text-left text-sm text-text-secondary">
                  <thead className="sticky top-0 bg-surface">
                    <tr>
                      {Object.keys(tableData[0]).map((key) => (
                        <th key={key} className="px-4 py-2 font-medium text-text-primary">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-t border-border hover:bg-background-secondary">
                        {Object.values(row).map((value, colIndex) => (
                          <td key={colIndex} className="px-4 py-2">{String(value)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Source Citations */}
      {sources && sources.length > 0 && (
        <Accordion type="single" collapsible className="w-full mt-4 border-t border-border pt-4">
          <AccordionItem value="sources">
            <AccordionTrigger className="flex items-center justify-between text-text-secondary hover:text-primary-accent">
              <div className="flex items-center">
                <FileText className="mr-2 h-5 w-5 text-primary-accent" />
                <span className="font-semibold">Verified Sources ({sources.length})</span>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <ul className="list-disc space-y-1 pl-5 text-sm text-text-secondary">
                {sources.map((source, index) => (
                  <li key={index}>
                    <span className="text-primary-accent">{source.document_name}</span> - Page {source.page_number}
                    {/* Add action to view source preview if implemented */}
                  </li>
                ))}
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </motion.div>
  );
};

export default AIMessageCard;
