"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Message {
    role: "agent" | "debtor" | "system";
    content: string;
    action?: string;
}

interface ChatPanelProps {
    messages: Message[];
    title?: string;
    className?: string;
    height?: string;
}

export function ChatPanel({
    messages,
    title = "Conversation",
    className,
    height = "h-[400px]",
}: ChatPanelProps) {
    return (
        <Card className={cn("", className)}>
            <CardHeader className="pb-3">
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <ScrollArea className={cn(height, "pr-4")}>
                    {messages.length === 0 ? (
                        <p className="text-center text-muted-foreground py-8">
                            Start a conversation to see the dialogue here
                        </p>
                    ) : (
                        <div className="space-y-4">
                            {messages.map((message, index) => (
                                <div
                                    key={index}
                                    className={cn(
                                        "flex flex-col gap-1 rounded-lg p-3",
                                        message.role === "agent" && "bg-primary/10 ml-4",
                                        message.role === "debtor" && "bg-secondary mr-4",
                                        message.role === "system" && "bg-muted text-center text-sm"
                                    )}
                                >
                                    {message.role !== "system" && (
                                        <span className="text-xs font-semibold text-muted-foreground uppercase">
                                            {message.role === "agent" ? "Agent" : "Debtor"}
                                            {message.action && ` (${message.action})`}
                                        </span>
                                    )}
                                    <p className="text-sm">{message.content}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
