'use client';

import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { useMessages } from '@/lib/i18n/useMessages';

export interface NewsItem {
    date: string;
    content: string;
}

interface NewsProps {
    items: NewsItem[];
    title?: string;
    initialCount?: number;
}

const DEFAULT_INITIAL_COUNT = 10;

export default function News({ items, title, initialCount = DEFAULT_INITIAL_COUNT }: NewsProps) {
    const messages = useMessages();
    const resolvedTitle = title || messages.home.news;
    const [expanded, setExpanded] = useState(false);

    const isCollapsible = items.length > initialCount;
    const visibleItems = isCollapsible && !expanded ? items.slice(0, initialCount) : items;

    return (
        <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
        >
            <h2 className="text-2xl font-serif font-bold text-primary mb-4">{resolvedTitle}</h2>
            <div className="space-y-3">
                <AnimatePresence initial={false}>
                    {visibleItems.map((item, index) => (
                        <motion.div
                            key={`${item.date}-${index}`}
                            initial={index < initialCount ? false : { opacity: 0, y: -6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -6 }}
                            transition={{ duration: 0.2 }}
                            className="flex items-start space-x-3"
                        >
                            <span className="text-xs text-neutral-500 mt-1 w-16 flex-shrink-0">{item.date}</span>
                            <div className="text-sm text-neutral-700 [&>p]:inline">
                                <ReactMarkdown
                                    components={{
                                        p: ({ children }) => <p className="inline">{children}</p>,
                                        strong: ({ children }) => <strong className="font-semibold text-primary">{children}</strong>,
                                        em: ({ children }) => <em className="italic text-neutral-600 dark:text-neutral-500">{children}</em>,
                                        a: ({ ...props }) => (
                                            <a
                                                {...props}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-accent font-medium transition-all duration-200 rounded hover:bg-accent/10 hover:shadow-sm"
                                            />
                                        ),
                                    }}
                                >
                                    {item.content}
                                </ReactMarkdown>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
            </div>
            {isCollapsible && (
                <button
                    type="button"
                    onClick={() => setExpanded((prev) => !prev)}
                    aria-expanded={expanded}
                    className="mt-3 ml-[4.75rem] inline-flex items-center gap-1 text-accent hover:text-accent-dark text-sm font-medium transition-all duration-200 rounded px-1.5 py-0.5 hover:bg-accent/10 hover:shadow-sm"
                >
                    {expanded ? messages.home.showLess : messages.home.showMore}
                    <span aria-hidden="true" className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}>
                        ↓
                    </span>
                </button>
            )}
        </motion.section>
    );
}
