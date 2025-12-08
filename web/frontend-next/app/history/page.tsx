"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatsCard } from "@/components/stats-card";
import {
    api,
    SessionSummary,
    SessionDetailResponse,
    GenerationDetailResponse,
    EpisodeDetailResponse,
    HistoryStatsResponse,
} from "@/lib/api";

type ViewState =
    | { type: "sessions" }
    | { type: "session"; id: number }
    | { type: "generation"; id: number; sessionId: number }
    | { type: "episode"; id: number; generationId: number; sessionId: number };

export default function HistoryPage() {
    const [viewState, setViewState] = useState<ViewState>({ type: "sessions" });
    const [sessions, setSessions] = useState<SessionSummary[]>([]);
    const [sessionDetail, setSessionDetail] = useState<SessionDetailResponse | null>(null);
    const [generationDetail, setGenerationDetail] = useState<GenerationDetailResponse | null>(null);
    const [episodeDetail, setEpisodeDetail] = useState<EpisodeDetailResponse | null>(null);
    const [stats, setStats] = useState<HistoryStatsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Load sessions list
    const loadSessions = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [historyResponse, statsResponse] = await Promise.all([
                api.getBattleHistory(50, 0),
                api.getHistoryStats(),
            ]);
            setSessions(historyResponse.sessions);
            setStats(statsResponse);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load history");
        } finally {
            setLoading(false);
        }
    }, []);

    // Load session detail
    const loadSession = useCallback(async (id: number) => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.getSessionDetail(id);
            setSessionDetail(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load session");
        } finally {
            setLoading(false);
        }
    }, []);

    // Load generation detail
    const loadGeneration = useCallback(async (id: number) => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.getGenerationDetail(id);
            setGenerationDetail(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load generation");
        } finally {
            setLoading(false);
        }
    }, []);

    // Load episode detail
    const loadEpisode = useCallback(async (id: number) => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.getEpisodeDetail(id);
            setEpisodeDetail(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load episode");
        } finally {
            setLoading(false);
        }
    }, []);

    // Delete session
    const deleteSession = async (id: number) => {
        if (!confirm("Delete this session and all its data?")) return;
        try {
            await api.deleteSession(id);
            await loadSessions();
            setViewState({ type: "sessions" });
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to delete");
        }
    };

    // Effect to load data based on view state
    useEffect(() => {
        if (viewState.type === "sessions") {
            loadSessions();
        } else if (viewState.type === "session") {
            loadSession(viewState.id);
        } else if (viewState.type === "generation") {
            loadGeneration(viewState.id);
        } else if (viewState.type === "episode") {
            loadEpisode(viewState.id);
        }
    }, [viewState, loadSessions, loadSession, loadGeneration, loadEpisode]);

    // Format date
    const formatDate = (dateStr: string | null) => {
        if (!dateStr) return "-";
        return new Date(dateStr).toLocaleString();
    };

    // Get status badge variant
    const getStatusVariant = (status: string) => {
        switch (status) {
            case "completed": return "default";
            case "running": return "secondary";
            case "error": return "destructive";
            default: return "outline";
        }
    };

    // Get outcome badge variant
    const getOutcomeVariant = (outcome: string) => {
        if (outcome === "collector_win") return "default";
        if (outcome === "adversary_win") return "destructive";
        return "secondary";
    };

    // Render breadcrumb navigation
    const renderBreadcrumb = () => {
        const parts: { label: string; onClick?: () => void }[] = [
            { label: "📜 History", onClick: () => setViewState({ type: "sessions" }) },
        ];

        if (viewState.type === "session" || viewState.type === "generation" || viewState.type === "episode") {
            parts.push({
                label: `Session #${viewState.type === "session" ? viewState.id : viewState.sessionId}`,
                onClick: () => setViewState({
                    type: "session",
                    id: viewState.type === "session" ? viewState.id : viewState.sessionId
                }),
            });
        }

        if (viewState.type === "generation" || viewState.type === "episode") {
            parts.push({
                label: `Generation`,
                onClick: viewState.type === "episode"
                    ? () => setViewState({ type: "generation", id: viewState.generationId, sessionId: viewState.sessionId })
                    : undefined,
            });
        }

        if (viewState.type === "episode") {
            parts.push({ label: `Episode #${episodeDetail?.episode_num || viewState.id}` });
        }

        return (
            <div className="flex items-center gap-2 text-sm mb-4">
                {parts.map((part, idx) => (
                    <span key={idx} className="flex items-center gap-2">
                        {idx > 0 && <span className="text-muted-foreground">/</span>}
                        {part.onClick ? (
                            <button
                                onClick={part.onClick}
                                className="text-primary hover:underline"
                            >
                                {part.label}
                            </button>
                        ) : (
                            <span className="text-muted-foreground">{part.label}</span>
                        )}
                    </span>
                ))}
            </div>
        );
    };

    // Render sessions list
    const renderSessionsList = () => (
        <>
            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    <StatsCard title="Total Sessions" value={stats.total_sessions} />
                    <StatsCard title="Completed" value={stats.completed_sessions} />
                    <StatsCard title="Total Generations" value={stats.total_generations} />
                    <StatsCard title="Total Episodes" value={stats.total_episodes} />
                    <StatsCard
                        title="Avg Collector Win"
                        value={`${(stats.avg_collector_win_rate * 100).toFixed(1)}%`}
                        className="border-l-4 border-l-green-500"
                    />
                </div>
            )}

            {/* Sessions Table */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Training Sessions</CardTitle>
                </CardHeader>
                <CardContent>
                    {sessions.length === 0 ? (
                        <p className="text-center text-muted-foreground py-8">
                            No training sessions yet. Run some battles in the Adversarial Arena!
                        </p>
                    ) : (
                        <div className="space-y-3">
                            {sessions.map((session) => (
                                <div
                                    key={session.id}
                                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors cursor-pointer"
                                    onClick={() => setViewState({ type: "session", id: session.id })}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="text-2xl font-bold text-muted-foreground">
                                            #{session.id}
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <Badge variant={getStatusVariant(session.status)}>
                                                    {session.status}
                                                </Badge>
                                                {session.use_llm && <Badge variant="outline">LLM</Badge>}
                                                {session.zero_sum && <Badge variant="outline">Zero-Sum</Badge>}
                                            </div>
                                            <div className="text-sm text-muted-foreground mt-1">
                                                {formatDate(session.started_at)}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6 text-sm">
                                        <div className="text-center">
                                            <div className="font-semibold">{session.generation_count}</div>
                                            <div className="text-muted-foreground">Gens</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="font-semibold">{session.episodes_per_gen}</div>
                                            <div className="text-muted-foreground">Eps/Gen</div>
                                        </div>
                                        {session.final_collector_win_rate !== null && (
                                            <div className="text-center">
                                                <div className="font-semibold text-green-500">
                                                    {(session.final_collector_win_rate * 100).toFixed(0)}%
                                                </div>
                                                <div className="text-muted-foreground">Win Rate</div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>
        </>
    );

    // Render session detail
    const renderSessionDetail = () => {
        if (!sessionDetail) return null;

        return (
            <>
                <Card className="mb-6">
                    <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-lg">
                                Session #{sessionDetail.id}
                            </CardTitle>
                            <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => deleteSession(sessionDetail.id)}
                            >
                                Delete
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                                <div className="text-muted-foreground">Status</div>
                                <Badge variant={getStatusVariant(sessionDetail.status)}>
                                    {sessionDetail.status}
                                </Badge>
                            </div>
                            <div>
                                <div className="text-muted-foreground">Started</div>
                                <div>{formatDate(sessionDetail.started_at)}</div>
                            </div>
                            <div>
                                <div className="text-muted-foreground">Ended</div>
                                <div>{formatDate(sessionDetail.ended_at)}</div>
                            </div>
                            <div>
                                <div className="text-muted-foreground">Config</div>
                                <div className="flex gap-1">
                                    {sessionDetail.use_llm && <Badge variant="outline" className="text-xs">LLM</Badge>}
                                    {sessionDetail.zero_sum && <Badge variant="outline" className="text-xs">Zero-Sum</Badge>}
                                </div>
                            </div>
                        </div>
                        {sessionDetail.final_collector_win_rate !== null && (
                            <div className="mt-4 p-3 bg-muted rounded-lg flex gap-6">
                                <div>
                                    <span className="text-muted-foreground">Collector Win: </span>
                                    <span className="font-semibold text-green-500">
                                        {(sessionDetail.final_collector_win_rate * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Adversary Win: </span>
                                    <span className="font-semibold text-red-500">
                                        {((sessionDetail.final_adversary_win_rate || 0) * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Generations */}
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Generations ({sessionDetail.generations.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2">
                            {sessionDetail.generations.map((gen) => (
                                <div
                                    key={gen.id}
                                    className="flex items-center justify-between p-3 border rounded hover:bg-muted/50 cursor-pointer"
                                    onClick={() => setViewState({ type: "generation", id: gen.id, sessionId: sessionDetail.id })}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="font-semibold">Gen {gen.generation_num}</div>
                                        <div className="text-sm text-muted-foreground">
                                            {gen.episode_count} episodes
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4 text-sm">
                                        <span className="text-green-500">
                                            C: {(gen.collector_win_rate * 100).toFixed(0)}%
                                        </span>
                                        <span className="text-red-500">
                                            A: {(gen.adversary_win_rate * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </>
        );
    };

    // Render generation detail
    const renderGenerationDetail = () => {
        if (!generationDetail) return null;

        return (
            <>
                <Card className="mb-6">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                            Generation {generationDetail.generation_num}
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <StatsCard
                                title="Collector Win Rate"
                                value={`${(generationDetail.collector_win_rate * 100).toFixed(1)}%`}
                                className="border-l-4 border-l-green-500"
                            />
                            <StatsCard
                                title="Adversary Win Rate"
                                value={`${(generationDetail.adversary_win_rate * 100).toFixed(1)}%`}
                                className="border-l-4 border-l-red-500"
                            />
                            <StatsCard
                                title="Avg Collector Reward"
                                value={generationDetail.avg_collector_reward.toFixed(2)}
                            />
                            <StatsCard
                                title="Avg Adversary Reward"
                                value={generationDetail.avg_adversary_reward.toFixed(2)}
                            />
                        </div>
                    </CardContent>
                </Card>

                {/* Episodes */}
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Episodes ({generationDetail.episodes.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-[400px]">
                            <div className="space-y-2">
                                {generationDetail.episodes.map((ep) => (
                                    <div
                                        key={ep.id}
                                        className="flex items-center justify-between p-3 border rounded hover:bg-muted/50 cursor-pointer"
                                        onClick={() => setViewState({
                                            type: "episode",
                                            id: ep.id,
                                            generationId: generationDetail.id,
                                            sessionId: viewState.type === "generation" ? viewState.sessionId : 0
                                        })}
                                    >
                                        <div className="flex items-center gap-4">
                                            <div className="font-semibold">Ep {ep.episode_num}</div>
                                            <Badge variant={getOutcomeVariant(ep.outcome)}>
                                                {ep.outcome.replace("_", " ")}
                                            </Badge>
                                        </div>
                                        <div className="flex items-center gap-4 text-sm">
                                            <span>{ep.num_turns} turns</span>
                                            <span className="text-green-500">C: {ep.collector_total_reward.toFixed(1)}</span>
                                            <span className="text-red-500">A: {ep.adversary_total_reward.toFixed(1)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    </CardContent>
                </Card>
            </>
        );
    };

    // Render episode detail with battle dialogue
    const renderEpisodeDetail = () => {
        if (!episodeDetail) return null;

        return (
            <>
                <Card className="mb-6">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-3">
                            Episode {episodeDetail.episode_num}
                            <Badge variant={getOutcomeVariant(episodeDetail.outcome)}>
                                {episodeDetail.outcome.replace("_", " ")}
                            </Badge>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                                <div className="text-muted-foreground">Turns</div>
                                <div className="font-semibold">{episodeDetail.num_turns}</div>
                            </div>
                            <div>
                                <div className="text-muted-foreground">Collector Reward</div>
                                <div className="font-semibold text-green-500">
                                    {episodeDetail.collector_total_reward.toFixed(2)}
                                </div>
                            </div>
                            <div>
                                <div className="text-muted-foreground">Adversary Reward</div>
                                <div className="font-semibold text-red-500">
                                    {episodeDetail.adversary_total_reward.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Battle Dialogue */}
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Battle Dialogue</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-[500px] pr-4">
                            <div className="space-y-4">
                                {episodeDetail.turns.map((turn) => (
                                    <div key={turn.id} className="border-b pb-4 last:border-0">
                                        <div className="text-xs text-muted-foreground mb-2">
                                            Turn {turn.turn_num}
                                        </div>
                                        <div className="bg-green-500/10 rounded p-3 mb-2">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="font-semibold">🎯 Collector</span>
                                                <Badge variant="outline" className="text-xs">
                                                    {turn.collector_strategy || "unknown"}
                                                </Badge>
                                                <span className="text-xs text-muted-foreground ml-auto">
                                                    reward: {turn.collector_reward.toFixed(2)}
                                                </span>
                                            </div>
                                            <p className="text-sm">{turn.collector_utterance || "[no utterance]"}</p>
                                        </div>
                                        <div className="bg-red-500/10 rounded p-3">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="font-semibold">🛡️ Adversary</span>
                                                <Badge variant="outline" className="text-xs">
                                                    {turn.adversary_strategy || "unknown"}
                                                </Badge>
                                                <span className="text-xs text-muted-foreground ml-auto">
                                                    reward: {turn.adversary_reward.toFixed(2)}
                                                </span>
                                            </div>
                                            <p className="text-sm">{turn.adversary_response || "[no response]"}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    </CardContent>
                </Card>
            </>
        );
    };

    return (
        <div className="space-y-6">
            {renderBreadcrumb()}

            {error && (
                <div className="p-4 bg-destructive/10 border border-destructive rounded-lg text-destructive">
                    {error}
                </div>
            )}

            {loading ? (
                <div className="flex items-center justify-center py-16">
                    <div className="text-muted-foreground">Loading...</div>
                </div>
            ) : (
                <>
                    {viewState.type === "sessions" && renderSessionsList()}
                    {viewState.type === "session" && renderSessionDetail()}
                    {viewState.type === "generation" && renderGenerationDetail()}
                    {viewState.type === "episode" && renderEpisodeDetail()}
                </>
            )}
        </div>
    );
}
