"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatsCard } from "@/components/stats-card";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts";

interface GenerationData {
    generation: number;
    collector_win_rate: number;
    adversary_win_rate: number;
    avg_collector_reward: number;
    avg_adversary_reward: number;
}

interface BattleTurn {
    collector_strategy: string;
    collector_utterance: string;
    adversary_strategy: string;
    adversary_response: string;
}

interface SelfPlayMessage {
    type: string;
    message?: string;
    generation?: number;
    episode?: number;
    collector_win_rate?: number;
    adversary_win_rate?: number;
    avg_collector_reward?: number;
    avg_adversary_reward?: number;
    collector_strategy?: string;
    collector_utterance?: string;
    adversary_strategy?: string;
    adversary_response?: string;
}

export default function AdversarialPage() {
    const [generations, setGenerations] = useState("5");
    const [episodesPerGen, setEpisodesPerGen] = useState("10");
    const [useLlm, setUseLlm] = useState(true);
    const [zeroSum, setZeroSum] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [status, setStatus] = useState("Connecting...");
    const [currentGen, setCurrentGen] = useState(0);
    const [currentEpisode, setCurrentEpisode] = useState(0);
    const [totalBattles, setTotalBattles] = useState(0);
    const [collectorWinRate, setCollectorWinRate] = useState(0);
    const [adversaryWinRate, setAdversaryWinRate] = useState(0);
    const [generationHistory, setGenerationHistory] = useState<GenerationData[]>([]);
    const [battles, setBattles] = useState<BattleTurn[]>([]);

    const ws = useRef<WebSocket | null>(null);
    const battleEndRef = useRef<HTMLDivElement>(null);

    // Handle incoming messages - defined first
    const handleMessage = useCallback((data: SelfPlayMessage) => {
        switch (data.type) {
            case "started":
                setIsRunning(true);
                setStatus(data.message || "Self-play started");
                break;

            case "generation":
                setCurrentGen(data.generation || 0);
                setCollectorWinRate(data.collector_win_rate || 0);
                setAdversaryWinRate(data.adversary_win_rate || 0);
                setGenerationHistory((prev) => [
                    ...prev,
                    {
                        generation: data.generation || 0,
                        collector_win_rate: data.collector_win_rate || 0,
                        adversary_win_rate: data.adversary_win_rate || 0,
                        avg_collector_reward: data.avg_collector_reward || 0,
                        avg_adversary_reward: data.avg_adversary_reward || 0,
                    },
                ]);
                break;

            case "episode":
                setCurrentEpisode(data.episode || 0);
                setTotalBattles((prev) => prev + 1);
                break;

            case "battle":
                setBattles((prev) => {
                    const newBattles = [
                        ...prev,
                        {
                            collector_strategy: data.collector_strategy || "unknown",
                            collector_utterance: data.collector_utterance || "[action]",
                            adversary_strategy: data.adversary_strategy || "unknown",
                            adversary_response: data.adversary_response || "[response]",
                        },
                    ];
                    return newBattles.slice(-15);
                });
                break;

            case "complete":
                setIsRunning(false);
                setStatus(data.message || "Training complete");
                break;

            case "stopped":
                setIsRunning(false);
                setStatus(data.message || "Training stopped");
                break;

            case "error":
                setIsRunning(false);
                setStatus(`Error: ${data.message}`);
                break;
        }
    }, []);

    const connectWebSocket = useCallback(() => {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/api/selfplay/ws`;

        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            setIsConnected(true);
            setStatus("Connected to self-play server");
        };

        ws.current.onmessage = (event) => {
            const data: SelfPlayMessage = JSON.parse(event.data);
            handleMessage(data);
        };

        ws.current.onclose = () => {
            setIsConnected(false);
            setStatus("Disconnected from server");
            setIsRunning(false);
        };

        ws.current.onerror = () => {
            setIsConnected(false);
            setStatus("Connection error - is FastAPI running?");
        };
    }, [handleMessage]);

    // Send start command - defined before startSelfPlay
    const sendStartCommand = useCallback(() => {
        setGenerationHistory([]);
        setBattles([]);
        setTotalBattles(0);
        setCurrentGen(0);
        setCurrentEpisode(0);

        ws.current?.send(
            JSON.stringify({
                action: "start",
                generations: parseInt(generations),
                episodes_per_gen: parseInt(episodesPerGen),
                use_llm: useLlm,
                zero_sum: zeroSum,
            })
        );
    }, [generations, episodesPerGen, useLlm, zeroSum]);

    const startSelfPlay = useCallback(() => {
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

    const stopSelfPlay = useCallback(() => {
        ws.current?.send(JSON.stringify({ action: "stop" }));
        setStatus("Stopping...");
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
        battleEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [battles]);

    return (
        <div className="space-y-6">
            {/* Control Panel */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-lg">⚔️ Adversarial Arena</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Generations:</span>
                            <input
                                type="number"
                                min="1"
                                max="100"
                                value={generations}
                                onChange={(e) => setGenerations(e.target.value)}
                                disabled={isRunning}
                                className="w-[70px] h-9 px-3 rounded-md border border-input bg-background text-sm"
                            />
                        </div>

                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">
                                Episodes/Gen:
                            </span>
                            <input
                                type="number"
                                min="1"
                                max="500"
                                value={episodesPerGen}
                                onChange={(e) => setEpisodesPerGen(e.target.value)}
                                disabled={isRunning}
                                className="w-[70px] h-9 px-3 rounded-md border border-input bg-background text-sm"
                            />
                        </div>

                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    id="use-llm"
                                    checked={useLlm}
                                    onChange={(e) => setUseLlm(e.target.checked)}
                                    disabled={isRunning}
                                    className="rounded"
                                />
                                <label htmlFor="use-llm" className="text-sm">
                                    LLM
                                </label>
                            </div>
                            <div className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    id="zero-sum"
                                    checked={zeroSum}
                                    onChange={(e) => setZeroSum(e.target.checked)}
                                    disabled={isRunning}
                                    className="rounded"
                                />
                                <label htmlFor="zero-sum" className="text-sm">
                                    Zero-Sum
                                </label>
                            </div>
                        </div>

                        <div className="flex gap-2 ml-auto">
                            <Button
                                onClick={startSelfPlay}
                                disabled={isRunning || !isConnected}
                            >
                                Start Battle
                            </Button>
                            <Button
                                variant="destructive"
                                onClick={stopSelfPlay}
                                disabled={!isRunning}
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
                <Badge variant={isRunning ? "default" : "secondary"}>
                    {isRunning ? "Running..." : "Idle"}
                </Badge>
                <span className="text-sm">{status}</span>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <StatsCard title="Generation" value={currentGen} />
                <StatsCard title="Episode" value={currentEpisode} />
                <StatsCard title="Total Battles" value={totalBattles} />
                <StatsCard
                    title="Collector Win Rate"
                    value={`${(collectorWinRate * 100).toFixed(1)}%`}
                    className="border-l-4 border-l-green-500"
                />
                <StatsCard
                    title="Adversary Win Rate"
                    value={`${(adversaryWinRate * 100).toFixed(1)}%`}
                    className="border-l-4 border-l-red-500"
                />
            </div>

            {/* Main Content */}
            <div className="grid lg:grid-cols-2 gap-6">
                {/* Win Rate Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">Win Rate Over Generations</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {generationHistory.length > 0 ? (
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={generationHistory}>
                                    <XAxis
                                        dataKey="generation"
                                        fontSize={12}
                                        tickLine={false}
                                        label={{ value: "Generation", position: "bottom", fontSize: 11 }}
                                    />
                                    <YAxis
                                        domain={[0, 1]}
                                        fontSize={12}
                                        tickLine={false}
                                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                                    />
                                    <Tooltip
                                        formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "var(--radius)",
                                        }}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="collector_win_rate"
                                        name="Collector"
                                        stroke="#22c55e"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="adversary_win_rate"
                                        name="Adversary"
                                        stroke="#ef4444"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <p className="text-center text-muted-foreground py-16 text-sm">
                                Start self-play to see win rates
                            </p>
                        )}
                    </CardContent>
                </Card>

                {/* Reward Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">
                            Average Reward Over Generations
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {generationHistory.length > 0 ? (
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={generationHistory}>
                                    <XAxis
                                        dataKey="generation"
                                        fontSize={12}
                                        tickLine={false}
                                        label={{ value: "Generation", position: "bottom", fontSize: 11 }}
                                    />
                                    <YAxis fontSize={12} tickLine={false} />
                                    <Tooltip
                                        formatter={(value: number) => value.toFixed(2)}
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "var(--radius)",
                                        }}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="avg_collector_reward"
                                        name="Collector Reward"
                                        stroke="#22c55e"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="avg_adversary_reward"
                                        name="Adversary Reward"
                                        stroke="#ef4444"
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <p className="text-center text-muted-foreground py-16 text-sm">
                                Start self-play to see rewards
                            </p>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Battle Dialogue */}
            <Card>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base">Battle Dialogue</CardTitle>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[300px] pr-4">
                        {battles.length === 0 ? (
                            <p className="text-center text-muted-foreground py-16 text-sm">
                                Waiting for battles...
                            </p>
                        ) : (
                            <div className="space-y-4">
                                {battles.map((battle, idx) => (
                                    <div key={idx} className="border-b pb-3 last:border-0">
                                        <div className="bg-green-500/10 rounded p-2 mb-2 text-sm">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="font-semibold">🎯 Collector</span>
                                                <Badge variant="outline" className="text-xs">
                                                    {battle.collector_strategy}
                                                </Badge>
                                            </div>
                                            <p>{battle.collector_utterance}</p>
                                        </div>
                                        <div className="bg-red-500/10 rounded p-2 text-sm">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="font-semibold">🛡️ Adversary</span>
                                                <Badge variant="outline" className="text-xs">
                                                    {battle.adversary_strategy}
                                                </Badge>
                                            </div>
                                            <p>{battle.adversary_response}</p>
                                        </div>
                                    </div>
                                ))}
                                <div ref={battleEndRef} />
                            </div>
                        )}
                    </ScrollArea>
                </CardContent>
            </Card>
        </div>
    );
}
