"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface TermProps {
    children: React.ReactNode;
    explanation: string;
}

/**
 * A tooltip wrapper for technical jargon terms.
 * Displays a dotted underline and shows explanation on hover.
 */
export function Term({ children, explanation }: TermProps) {
    const [isHovered, setIsHovered] = useState(false);

    return (
        <span
            className="relative inline-block"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <span className="border-b border-dotted border-primary/50 cursor-help">
                {children}
            </span>
            <AnimatePresence>
                {isHovered && (
                    <motion.div
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 5 }}
                        transition={{ duration: 0.15 }}
                        className="absolute z-50 left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-popover border border-border rounded-md shadow-lg text-sm max-w-[250px] text-center whitespace-normal"
                    >
                        <div className="text-popover-foreground">{explanation}</div>
                        <div className="absolute left-1/2 -translate-x-1/2 top-full border-4 border-transparent border-t-popover" />
                    </motion.div>
                )}
            </AnimatePresence>
        </span>
    );
}

// Common glossary terms for reuse
export const GLOSSARY = {
    NLU: "Natural Language Understanding - Reads tone and meaning from text",
    state: "What the AI knows about the current conversation",
    sentiment: "How positive or negative the debtor feels (-1 to +1)",
    cooperation: "How willing the debtor is to work together (0 to 1)",
    intent: "What the debtor is trying to do (e.g., explain, avoid, commit)",
    DDQ: "Deep Dyna-Q - An AI that imagines 'what if' scenarios to learn faster",
    worldModel: "AI's prediction of what happens next when it takes an action",
    selfPlay: "Training by having the AI argue with itself",
    action: "A strategy the collector agent can use",
    episode: "One complete conversation from start to finish",
    reward: "Points the AI gets for good outcomes (payment commitment)",
    adversary: "A simulated difficult debtor that resists payment",
} as const;
