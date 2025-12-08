"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { StatsCard } from "@/components/stats-card";

interface Checkpoint {
    name: string;
    algorithm: string;
}

interface ConversationTurn {
    agent_utterance: string;
    debtor_response: string;
}

interface EvalResult {
    success_rate: number;
    avg_reward: number;
    avg_length: number;
    num_episodes: number;
    sample_conversations: ConversationTurn[][];
}

export default function EvaluatePage() {
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
    const [numEpisodes, setNumEpisodes] = useState("10");
    const [useLlm, setUseLlm] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState("Loading checkpoints...");
    const [result, setResult] = useState<EvalResult | null>(null);
    const [convIndex, setConvIndex] = useState(0);

    useEffect(() => {
        loadCheckpoints();
    }, []);

    const loadCheckpoints = async () => {
        try {
            const response = await fetch("/api/evaluate/checkpoints");
            const data = await response.json();
            setCheckpoints(data.checkpoints || []);
            if (data.checkpoints?.length > 0) {
                setSelectedCheckpoint(data.checkpoints[0].name);
                setStatus(`Found ${data.checkpoints.length} checkpoint(s)`);
            } else {
                setStatus("No checkpoints found - train a model first");
            }
        } catch (error) {
            console.error("Failed to load checkpoints:", error);
            setStatus("Failed to load checkpoints - is FastAPI running?");
        }
    };

    const runEvaluation = useCallback(async () => {
        if (!selectedCheckpoint) {
            setStatus("Please select a checkpoint");
            return;
        }

        const algorithm = selectedCheckpoint.toLowerCase().includes("ddq")
            ? "ddq"
            : "dqn";

        setIsLoading(true);
        setStatus("Running evaluation...");
        setResult(null);

        try {
            const response = await fetch("/api/evaluate/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    checkpoint: selectedCheckpoint,
                    algorithm,
                    num_episodes: parseInt(numEpisodes),
                    use_llm: useLlm,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Evaluation failed");
            }

            const evalResult = await response.json();
            setResult(evalResult);
            setConvIndex(0);
            setStatus("Evaluation complete!");
        } catch (error) {
            setStatus(`Error: ${error}`);
        } finally {
            setIsLoading(false);
        }
    }, [selectedCheckpoint, numEpisodes, useLlm]);

    const conversations = result?.sample_conversations || [];
    const currentConv = conversations[convIndex] || [];

    const getRating = (successRate: number) => {
        if (successRate >= 0.6) return { text: "Good", color: "text-green-500" };
        if (successRate >= 0.4) return { text: "Moderate", color: "text-yellow-500" };
        return { text: "Needs improvement", color: "text-red-500" };
    };

    return (
        <div className="space-y-6">
            {/* Control Panel */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Evaluation Configuration</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Checkpoint:</span>
                            <Select
                                value={selectedCheckpoint}
                                onValueChange={setSelectedCheckpoint}
                                disabled={isLoading}
                            >
                                <SelectTrigger className="w-[200px]">
                                    <SelectValue placeholder="Select checkpoint" />
                                </SelectTrigger>
                                <SelectContent>
                                    {checkpoints.map((cp) => (
                                        <SelectItem key={cp.name} value={cp.name}>
                                            {cp.name} ({cp.algorithm.toUpperCase()})
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Episodes:</span>
                            <Select
                                value={numEpisodes}
                                onValueChange={setNumEpisodes}
                                disabled={isLoading}
                            >
                                <SelectTrigger className="w-[80px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="5">5</SelectItem>
                                    <SelectItem value="10">10</SelectItem>
                                    <SelectItem value="20">20</SelectItem>
                                    <SelectItem value="50">50</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                id="use-llm"
                                checked={useLlm}
                                onChange={(e) => setUseLlm(e.target.checked)}
                                disabled={isLoading}
                                className="rounded"
                            />
                            <label htmlFor="use-llm" className="text-sm">
                                Use LLM
                            </label>
                        </div>

                        <Button
                            onClick={runEvaluation}
                            disabled={isLoading || !selectedCheckpoint}
                            className="ml-auto"
                        >
                            {isLoading ? "Evaluating..." : "Run Evaluation"}
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Status Bar */}
            <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <Badge variant={result ? "default" : "secondary"}>
                    {result ? "Complete" : "Ready"}
                </Badge>
                <span className="text-sm">{status}</span>
                {isLoading && (
                    <span className="ml-auto text-sm text-muted-foreground">
                        Loading...
                    </span>
                )}
            </div>

            {/* Results */}
            {result && (
                <>
                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <StatsCard
                            title="Success Rate"
                            value={`${(result.success_rate * 100).toFixed(1)}%`}
                            trend={result.success_rate >= 0.5 ? "up" : "down"}
                        />
                        <StatsCard
                            title="Avg Reward"
                            value={result.avg_reward.toFixed(2)}
                            trend={result.avg_reward >= 0 ? "up" : "down"}
                        />
                        <StatsCard
                            title="Avg Length"
                            value={`${result.avg_length.toFixed(1)} turns`}
                        />
                        <StatsCard title="Episodes" value={result.num_episodes} />
                    </div>

                    {/* Summary */}
                    <Card>
                        <CardHeader className="pb-2">
                            <CardTitle className="text-base">Evaluation Summary</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm">
                                The model achieved a{" "}
                                <strong>{(result.success_rate * 100).toFixed(1)}%</strong>{" "}
                                success rate over <strong>{result.num_episodes}</strong>{" "}
                                episodes. Performance rating:{" "}
                                <span className={getRating(result.success_rate).color}>
                                    <strong>{getRating(result.success_rate).text}</strong>
                                </span>
                                .
                            </p>
                        </CardContent>
                    </Card>

                    {/* Sample Conversations */}
                    {conversations.length > 0 && (
                        <Card>
                            <CardHeader className="pb-2">
                                <div className="flex items-center justify-between">
                                    <CardTitle className="text-base">
                                        Sample Conversations
                                    </CardTitle>
                                    <div className="flex items-center gap-2">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setConvIndex((i) => Math.max(0, i - 1))}
                                            disabled={convIndex === 0}
                                        >
                                            ←
                                        </Button>
                                        <span className="text-sm text-muted-foreground">
                                            {convIndex + 1} / {conversations.length}
                                        </span>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() =>
                                                setConvIndex((i) =>
                                                    Math.min(conversations.length - 1, i + 1)
                                                )
                                            }
                                            disabled={convIndex === conversations.length - 1}
                                        >
                                            →
                                        </Button>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <ScrollArea className="h-[300px] pr-4">
                                    <div className="space-y-4">
                                        {currentConv.map((turn, idx) => (
                                            <div key={idx} className="border-b pb-3 last:border-0">
                                                <div className="text-xs text-muted-foreground mb-2">
                                                    Turn {idx + 1}
                                                </div>
                                                <div className="bg-primary/10 rounded p-2 mb-2 text-sm">
                                                    <span className="font-semibold">🤖 Agent:</span>{" "}
                                                    {turn.agent_utterance || "[Action taken]"}
                                                </div>
                                                <div className="bg-secondary rounded p-2 text-sm">
                                                    <span className="font-semibold">👤 Debtor:</span>{" "}
                                                    {turn.debtor_response || "[Response]"}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </ScrollArea>
                            </CardContent>
                        </Card>
                    )}
                </>
            )}

            {/* Checkpoints Table */}
            <Card>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base">Available Checkpoints</CardTitle>
                </CardHeader>
                <CardContent>
                    {checkpoints.length === 0 ? (
                        <p className="text-center text-muted-foreground py-8 text-sm">
                            No checkpoints found
                        </p>
                    ) : (
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Checkpoint</TableHead>
                                    <TableHead>Algorithm</TableHead>
                                    <TableHead className="w-[100px]">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {checkpoints.map((cp) => (
                                    <TableRow key={cp.name}>
                                        <TableCell>{cp.name}</TableCell>
                                        <TableCell>
                                            <Badge
                                                variant={
                                                    cp.algorithm === "ddq" ? "default" : "secondary"
                                                }
                                            >
                                                {cp.algorithm.toUpperCase()}
                                            </Badge>
                                        </TableCell>
                                        <TableCell>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => setSelectedCheckpoint(cp.name)}
                                            >
                                                Select
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
