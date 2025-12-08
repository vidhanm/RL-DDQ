"use client";

import { useState, useEffect, useCallback, useRef } from "react";
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
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatsCard } from "@/components/stats-card";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
} from "recharts";

interface TrainingMessage {
    type: string;
    message?: string;
    episode?: number;
    total_episodes?: number;
    reward?: number;
    success_rate?: number;
    success?: boolean;
    agent_utterance?: string;
    debtor_response?: string;
    is_training?: boolean;
    current_episode?: number;
}

interface DialogueTurn {
    episode: number;
    agent: string;
    debtor: string;
}

interface RewardPoint {
    episode: number;
    reward: number;
}

export default function TrainPage() {
    const [algorithm, setAlgorithm] = useState("ddq");
    const [episodes, setEpisodes] = useState("100");
    const [difficulty, setDifficulty] = useState("random");
    const [useLlm, setUseLlm] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [status, setStatus] = useState("Connecting...");
    const [currentEpisode, setCurrentEpisode] = useState(0);
    const [totalEpisodes, setTotalEpisodes] = useState(0);
    const [successRate, setSuccessRate] = useState(0);
    const [lastReward, setLastReward] = useState(0);
    const [rewardHistory, setRewardHistory] = useState<RewardPoint[]>([]);
    const [dialogues, setDialogues] = useState<DialogueTurn[]>([]);

    const ws = useRef<WebSocket | null>(null);
    const dialogueEndRef = useRef<HTMLDivElement>(null);

    // Handle incoming messages - defined first so it can be used in connectWebSocket
    const handleMessage = useCallback((data: TrainingMessage) => {
        switch (data.type) {
            case "started":
                setIsTraining(true);
                setStatus(data.message || "Training started");
                break;

            case "episode":
                setCurrentEpisode(data.episode || 0);
                setTotalEpisodes(data.total_episodes || 0);
                setLastReward(data.reward || 0);
                setSuccessRate(data.success_rate || 0);
                setRewardHistory((prev) => {
                    const newHistory = [...prev, { episode: data.episode!, reward: data.reward! }];
                    return newHistory.slice(-100);
                });
                break;

            case "dialogue":
                setDialogues((prev) => {
                    const newDialogues = [
                        ...prev,
                        {
                            episode: data.episode || 0,
                            agent: data.agent_utterance || "[Action taken]",
                            debtor: data.debtor_response || "[Response]",
                        },
                    ];
                    return newDialogues.slice(-20);
                });
                break;

            case "complete":
                setIsTraining(false);
                setStatus(data.message || "Training complete");
                break;

            case "stopped":
                setIsTraining(false);
                setStatus(data.message || "Training stopped");
                break;

            case "stopping":
                setStatus(data.message || "Stopping...");
                break;

            case "error":
                setIsTraining(false);
                setStatus(`Error: ${data.message}`);
                break;

            case "status":
                if (data.is_training) {
                    setIsTraining(true);
                    setCurrentEpisode(data.current_episode || 0);
                    setTotalEpisodes(data.total_episodes || 0);
                    setSuccessRate(data.success_rate || 0);
                }
                break;
        }
    }, []);

    const connectWebSocket = useCallback(() => {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/api/training/ws`;

        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            setIsConnected(true);
            setStatus("Connected to training server");
        };

        ws.current.onmessage = (event) => {
            const data: TrainingMessage = JSON.parse(event.data);
            handleMessage(data);
        };

        ws.current.onclose = () => {
            setIsConnected(false);
            setStatus("Disconnected from server");
            setIsTraining(false);
        };

        ws.current.onerror = () => {
            setIsConnected(false);
            setStatus("Connection error - is FastAPI running?");
        };
    }, [handleMessage]);

    // Send start command - defined before startTraining
    const sendStartCommand = useCallback(() => {
        setRewardHistory([]);
        setDialogues([]);
        setCurrentEpisode(0);

        ws.current?.send(
            JSON.stringify({
                action: "start",
                algorithm,
                episodes: parseInt(episodes),
                difficulty,
                use_llm: useLlm,
            })
        );
    }, [algorithm, episodes, difficulty, useLlm]);

    const startTraining = useCallback(() => {
        if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            setTimeout(() => {
                if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                    sendStartCommand();
                }
            }, 1000);
        } else {
            sendStartCommand();
        }
    }, [connectWebSocket, sendStartCommand]);

    const stopTraining = useCallback(() => {
        ws.current?.send(JSON.stringify({ action: "stop" }));
    }, []);

    useEffect(() => {
        connectWebSocket();
        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [connectWebSocket]);

    useEffect(() => {
        dialogueEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [dialogues]);

    const progress = totalEpisodes > 0 ? (currentEpisode / totalEpisodes) * 100 : 0;

    return (
        <div className="space-y-6">
            {/* Control Panel */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Training Configuration</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Algorithm:</span>
                            <Select value={algorithm} onValueChange={setAlgorithm} disabled={isTraining}>
                                <SelectTrigger className="w-[100px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="ddq">DDQ</SelectItem>
                                    <SelectItem value="dqn">DQN</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Episodes:</span>
                            <Select value={episodes} onValueChange={setEpisodes} disabled={isTraining}>
                                <SelectTrigger className="w-[100px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="10">10</SelectItem>
                                    <SelectItem value="50">50</SelectItem>
                                    <SelectItem value="100">100</SelectItem>
                                    <SelectItem value="500">500</SelectItem>
                                    <SelectItem value="1000">1000</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Difficulty:</span>
                            <Select value={difficulty} onValueChange={setDifficulty} disabled={isTraining}>
                                <SelectTrigger className="w-[100px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="random">Random</SelectItem>
                                    <SelectItem value="easy">Easy</SelectItem>
                                    <SelectItem value="medium">Medium</SelectItem>
                                    <SelectItem value="hard">Hard</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                id="use-llm"
                                checked={useLlm}
                                onChange={(e) => setUseLlm(e.target.checked)}
                                disabled={isTraining}
                                className="rounded"
                            />
                            <label htmlFor="use-llm" className="text-sm">Use LLM</label>
                        </div>

                        <div className="flex gap-2 ml-auto">
                            <Button
                                onClick={startTraining}
                                disabled={isTraining || !isConnected}
                            >
                                Start Training
                            </Button>
                            <Button
                                variant="destructive"
                                onClick={stopTraining}
                                disabled={!isTraining}
                            >
                                Stop
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Status Bar */}
            <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
                <Badge variant={isConnected ? "default" : "destructive"}>
                    {isConnected ? "Connected" : "Disconnected"}
                </Badge>
                <Badge variant={isTraining ? "default" : "secondary"}>
                    {isTraining ? "Training..." : "Idle"}
                </Badge>
                <span className="text-sm">{status}</span>
            </div>

            {/* Progress */}
            {isTraining && (
                <Card>
                    <CardContent className="pt-4">
                        <div className="flex items-center justify-between text-sm mb-2">
                            <span>Episode {currentEpisode} / {totalEpisodes}</span>
                            <span>{progress.toFixed(1)}%</span>
                        </div>
                        <Progress value={progress} className="h-2" />
                    </CardContent>
                </Card>
            )}

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatsCard title="Current Episode" value={currentEpisode} />
                <StatsCard title="Total Episodes" value={totalEpisodes} />
                <StatsCard
                    title="Success Rate"
                    value={`${(successRate * 100).toFixed(1)}%`}
                    trend={successRate > 0.5 ? "up" : "down"}
                />
                <StatsCard
                    title="Last Reward"
                    value={lastReward.toFixed(2)}
                    trend={lastReward > 0 ? "up" : "down"}
                />
            </div>

            {/* Main Content */}
            <div className="grid lg:grid-cols-2 gap-6">
                {/* Reward Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">Training Progress</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {rewardHistory.length > 0 ? (
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={rewardHistory}>
                                    <XAxis
                                        dataKey="episode"
                                        fontSize={12}
                                        tickLine={false}
                                    />
                                    <YAxis fontSize={12} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "var(--radius)",
                                        }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="reward"
                                        stroke="hsl(var(--primary))"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <p className="text-center text-muted-foreground py-16 text-sm">
                                Start training to see reward chart
                            </p>
                        )}
                    </CardContent>
                </Card>

                {/* Dialogue Log */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">Training Dialogues</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-[250px] pr-4">
                            {dialogues.length === 0 ? (
                                <p className="text-center text-muted-foreground py-16 text-sm">
                                    Waiting for dialogues...
                                </p>
                            ) : (
                                <div className="space-y-4">
                                    {dialogues.map((d, idx) => (
                                        <div key={idx} className="border-b pb-3 last:border-0">
                                            <div className="text-xs text-muted-foreground mb-2">
                                                Episode {d.episode}
                                            </div>
                                            <div className="bg-primary/10 rounded p-2 mb-2 text-sm">
                                                <span className="font-semibold">🤖 Agent:</span> {d.agent}
                                            </div>
                                            <div className="bg-secondary rounded p-2 text-sm">
                                                <span className="font-semibold">👤 Debtor:</span> {d.debtor}
                                            </div>
                                        </div>
                                    ))}
                                    <div ref={dialogueEndRef} />
                                </div>
                            )}
                        </ScrollArea>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
