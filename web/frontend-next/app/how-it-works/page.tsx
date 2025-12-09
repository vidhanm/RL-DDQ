"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Term, GLOSSARY } from "@/components/term";

// ============== Data Constants ==============

const COLLECTOR_ACTIONS = [
    { id: 0, name: "empathetic_listening", icon: "💚", category: "empathy", description: "Show understanding and let debtor express feelings", when: "Start of call, angry/upset debtor" },
    { id: 1, name: "ask_about_situation", icon: "❓", category: "empathy", description: "Understand debtor's circumstances", when: "Need more information" },
    { id: 2, name: "firm_reminder", icon: "⚠️", category: "pressure", description: "State consequences clearly", when: "Avoidant or stalling debtor" },
    { id: 3, name: "offer_payment_plan", icon: "📋", category: "solution", description: "Propose installment options", when: "Debtor shows willingness" },
    { id: 4, name: "propose_settlement", icon: "🤝", category: "solution", description: "Offer reduced amount to settle", when: "Financial hardship confirmed" },
    { id: 5, name: "hard_close", icon: "🔒", category: "pressure", description: "Push firmly for commitment", when: "Ready to commit, just needs push" },
    { id: 6, name: "acknowledge_and_redirect", icon: "🔄", category: "empathy", description: "Handle venting, bring back to topic", when: "Debtor goes off-topic" },
    { id: 7, name: "validate_then_offer", icon: "💡", category: "solution", description: "Acknowledge emotion, then solution", when: "Emotional but open debtor" },
    { id: 8, name: "gentle_urgency", icon: "⏰", category: "urgency", description: "Create soft time pressure", when: "Needs motivation to act" },
];

const ADVERSARY_ACTIONS = [
    { id: 0, name: "aggressive", icon: "😠", description: "Hostile, threatening to hang up" },
    { id: 1, name: "evasive", icon: "🙈", description: "Deflect, change subject" },
    { id: 2, name: "emotional", icon: "😢", description: "Express distress, play victim" },
    { id: 3, name: "negotiate_hard", icon: "💪", description: "Demand unrealistic terms" },
    { id: 4, name: "partial_cooperate", icon: "🤷", description: "Give minimal ground" },
    { id: 5, name: "stall", icon: "⏳", description: "Ask for delays, need to think" },
    { id: 6, name: "dispute", icon: "⚖️", description: "Challenge validity of debt" },
];

const STATE_FEATURES = [
    { name: "Sentiment", type: "gauge", range: [-1, 1], emoji: (v: number) => v < -0.5 ? "😠" : v < 0 ? "😕" : v < 0.5 ? "😐" : "🙂" },
    { name: "Cooperation", type: "gauge", range: [0, 1], emoji: (v: number) => v < 0.3 ? "🚫" : v < 0.6 ? "🤷" : "✅" },
    { name: "Intent", type: "pills", options: ["committing", "willing", "explaining", "questioning", "neutral", "avoidant", "refusing", "hostile"] },
    { name: "Shared Situation", type: "flag" },
    { name: "Feels Understood", type: "flag" },
    { name: "Commitment Signal", type: "flag" },
    { name: "Quit Signal", type: "flag" },
    { name: "Turn", type: "progress", max: 15 },
];

const PIPELINE_STEPS = [
    { id: "debtor", label: "📞 Debtor", description: "Debtor speaks or responds" },
    { id: "nlu", label: "🧠 NLU", description: "Extract meaning, tone, intent" },
    { id: "state", label: "📊 State", description: "19 features the AI can see" },
    { id: "agent", label: "🤖 Agent", description: "DDQ chooses best action" },
    { id: "action", label: "⚡ Action", description: "One of 9 strategies" },
    { id: "response", label: "💬 Response", description: "LLM generates dialogue" },
];

// ============== Sub-Components ==============

function HumanTermsBox({ children }: { children: React.ReactNode }) {
    return (
        <div className="bg-muted/50 border-l-4 border-primary/30 rounded-r-md p-4 my-4">
            <div className="flex items-start gap-2">
                <span className="text-lg">🧑</span>
                <div>
                    <div className="font-medium text-sm text-muted-foreground mb-1">In human terms:</div>
                    <div className="text-sm">{children}</div>
                </div>
            </div>
        </div>
    );
}

function PipelineFlow() {
    return (
        <div className="flex flex-wrap items-center justify-center gap-2 py-8">
            {PIPELINE_STEPS.map((step, idx) => (
                <motion.div
                    key={step.id}
                    className="flex items-center"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                >
                    <div className="group relative">
                        <div className="bg-card border border-border rounded-lg px-4 py-3 text-center hover:border-primary/50 transition-colors cursor-default">
                            <div className="text-2xl mb-1">{step.label.split(" ")[0]}</div>
                            <div className="text-xs font-medium">{step.label.split(" ").slice(1).join(" ")}</div>
                        </div>
                        <div className="absolute opacity-0 group-hover:opacity-100 transition-opacity bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-popover border rounded-md text-xs whitespace-nowrap z-10">
                            {step.description}
                        </div>
                    </div>
                    {idx < PIPELINE_STEPS.length - 1 && (
                        <motion.div
                            className="mx-2 text-muted-foreground"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: idx * 0.1 + 0.05 }}
                        >
                            →
                        </motion.div>
                    )}
                </motion.div>
            ))}
        </div>
    );
}

function ActionCard({ action, index }: { action: typeof COLLECTOR_ACTIONS[0], index: number }) {
    const categoryColors: Record<string, string> = {
        empathy: "border-l-green-500",
        pressure: "border-l-red-500",
        solution: "border-l-blue-500",
        urgency: "border-l-yellow-500",
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`group bg-card border rounded-lg p-4 hover:shadow-lg transition-all cursor-default border-l-4 ${categoryColors[action.category]}`}
        >
            <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{action.icon}</span>
                <span className="font-mono text-sm">{action.name}</span>
            </div>
            <p className="text-sm text-muted-foreground">{action.description}</p>
            <div className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Badge variant="outline" className="text-xs">Best when: {action.when}</Badge>
            </div>
        </motion.div>
    );
}

function StateVisualizer() {
    // Demo state values
    const demoState = { sentiment: -0.3, cooperation: 0.6, intent: "explaining", turn: 5 };

    // Helper functions for emoji display
    const getSentimentEmoji = (v: number) => v < -0.5 ? "😠" : v < 0 ? "😕" : v < 0.5 ? "😐" : "🙂";
    const getCooperationEmoji = (v: number) => v < 0.3 ? "🚫" : v < 0.6 ? "🤷" : "✅";

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Sentiment */}
            <div className="bg-card border rounded-lg p-4">
                <div className="text-sm text-muted-foreground mb-1">Sentiment</div>
                <div className="flex items-center gap-2">
                    <span className="text-2xl">{getSentimentEmoji(demoState.sentiment)}</span>
                    <span className="text-lg font-semibold">{demoState.sentiment.toFixed(1)}</span>
                </div>
                <div className="text-xs text-muted-foreground">Slightly Upset</div>
            </div>

            {/* Cooperation */}
            <div className="bg-card border rounded-lg p-4">
                <div className="text-sm text-muted-foreground mb-1">Cooperation</div>
                <div className="flex items-center gap-2">
                    <span className="text-2xl">{getCooperationEmoji(demoState.cooperation)}</span>
                    <span className="text-lg font-semibold">{(demoState.cooperation * 100).toFixed(0)}%</span>
                </div>
                <div className="text-xs text-muted-foreground">Somewhat Willing</div>
            </div>

            {/* Intent */}
            <div className="bg-card border rounded-lg p-4">
                <div className="text-sm text-muted-foreground mb-1">Current Intent</div>
                <Badge variant="default" className="capitalize">{demoState.intent}</Badge>
                <div className="text-xs text-muted-foreground mt-1">Sharing their story</div>
            </div>

            {/* Turn */}
            <div className="bg-card border rounded-lg p-4">
                <div className="text-sm text-muted-foreground mb-1">Conversation Progress</div>
                <div className="flex items-center gap-2">
                    <span className="text-lg font-semibold">Turn {demoState.turn}/15</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 mt-2">
                    <div className="bg-primary rounded-full h-2" style={{ width: `${(demoState.turn / 15) * 100}%` }} />
                </div>
            </div>
        </div>
    );
}

function DDQExplainer() {
    return (
        <div className="space-y-4">
            <div className="flex flex-col md:flex-row items-center gap-8">
                {/* Real Experience */}
                <motion.div
                    className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-center flex-1"
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                >
                    <div className="text-sm font-medium mb-2">Real Experience</div>
                    <div className="text-xs text-muted-foreground">
                        "I tried empathy → debtor opened up"
                    </div>
                </motion.div>

                <motion.div
                    className="text-2xl"
                    initial={{ opacity: 0, scale: 0.5 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 }}
                >
                    →
                </motion.div>

                {/* World Model */}
                <motion.div
                    className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 text-center flex-1"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="text-sm font-medium mb-2">🧠 World Model</div>
                    <div className="text-xs text-muted-foreground">
                        "What would happen if I tried something else?"
                    </div>
                </motion.div>

                <motion.div
                    className="text-2xl"
                    initial={{ opacity: 0, scale: 0.5 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.4 }}
                >
                    →
                </motion.div>

                {/* Imagined */}
                <motion.div
                    className="flex-1 space-y-1"
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.5 }}
                >
                    {[1, 2, 3, 4, 5].map((i) => (
                        <div key={i} className="bg-purple-500/10 border border-purple-500/30 rounded px-2 py-1 text-xs text-center">
                            Imagined #{i}
                        </div>
                    ))}
                </motion.div>
            </div>
        </div>
    );
}

// ============== Main Page ==============

export default function HowItWorksPage() {
    return (
        <div className="space-y-12 pb-12">
            {/* Hero Section */}
            <section className="text-center space-y-4">
                <motion.h1
                    className="text-4xl font-bold"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    How It Works
                </motion.h1>
                <motion.p
                    className="text-lg text-muted-foreground max-w-2xl mx-auto"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.1 }}
                >
                    An AI that learns to have difficult debt collection conversations.
                    <br />
                    <span className="text-sm">Hover over <Term explanation={GLOSSARY.NLU}>highlighted terms</Term> for explanations.</span>
                </motion.p>
            </section>

            {/* Pipeline Overview */}
            <section>
                <Card>
                    <CardHeader>
                        <CardTitle>The Conversation Loop</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <PipelineFlow />
                        <HumanTermsBox>
                            When the debtor speaks, the AI reads between the lines (<Term explanation={GLOSSARY.NLU}>NLU</Term>),
                            figures out what it knows (<Term explanation={GLOSSARY.state}>state</Term>),
                            picks a strategy (<Term explanation={GLOSSARY.action}>action</Term>),
                            and generates a natural response. Then the debtor responds, and the loop continues.
                        </HumanTermsBox>
                    </CardContent>
                </Card>
            </section>

            {/* State Section */}
            <section>
                <Card>
                    <CardHeader>
                        <CardTitle>What the AI "Sees"</CardTitle>
                        <p className="text-sm text-muted-foreground">
                            The <Term explanation={GLOSSARY.state}>state</Term> is 19 numbers representing the conversation
                        </p>
                    </CardHeader>
                    <CardContent>
                        <StateVisualizer />
                        <HumanTermsBox>
                            The AI doesn't read the actual words. Instead, it sees numbers: "debtor seems -0.3 unhappy,
                            60% cooperative, explaining their situation, on turn 5 of 15." That's enough to make smart decisions!
                        </HumanTermsBox>
                    </CardContent>
                </Card>
            </section>

            {/* Actions Section */}
            <section>
                <Card>
                    <CardHeader>
                        <CardTitle>9 Strategies to Choose From</CardTitle>
                        <p className="text-sm text-muted-foreground">
                            Each turn, the agent picks one <Term explanation={GLOSSARY.action}>action</Term>
                        </p>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {COLLECTOR_ACTIONS.map((action, idx) => (
                                <ActionCard key={action.id} action={action} index={idx} />
                            ))}
                        </div>
                        <div className="flex gap-4 mt-6 text-sm">
                            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-green-500" /> Empathy</div>
                            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-blue-500" /> Solution</div>
                            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-yellow-500" /> Urgency</div>
                            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-red-500" /> Pressure</div>
                        </div>
                    </CardContent>
                </Card>
            </section>

            {/* DDQ Section */}
            <section>
                <Card>
                    <CardHeader>
                        <CardTitle>Learning Through Imagination</CardTitle>
                        <p className="text-sm text-muted-foreground">
                            <Term explanation={GLOSSARY.DDQ}>DDQ</Term> uses a <Term explanation={GLOSSARY.worldModel}>World Model</Term> to learn faster
                        </p>
                    </CardHeader>
                    <CardContent>
                        <DDQExplainer />
                        <HumanTermsBox>
                            For every real conversation, the AI imagines 5 "what if" alternatives.
                            "What if I'd been firmer? What if I'd offered a payment plan earlier?"
                            This means it learns 6x faster than just practicing on real calls.
                        </HumanTermsBox>
                    </CardContent>
                </Card>
            </section>

            {/* Self-Play Section */}
            <section>
                <Card>
                    <CardHeader>
                        <CardTitle>Training Against Itself</CardTitle>
                        <p className="text-sm text-muted-foreground">
                            <Term explanation={GLOSSARY.selfPlay}>Self-play</Term> makes the AI robust
                        </p>
                    </CardHeader>
                    <CardContent>
                        <div className="grid md:grid-cols-2 gap-8">
                            {/* Collector */}
                            <div className="space-y-3">
                                <h3 className="font-semibold flex items-center gap-2">
                                    <span className="text-green-500">🎯</span> Collector Agent
                                </h3>
                                <p className="text-sm text-muted-foreground">
                                    Goal: Get payment commitment using 9 strategies
                                </p>
                            </div>

                            {/* Adversary */}
                            <div className="space-y-3">
                                <h3 className="font-semibold flex items-center gap-2">
                                    <span className="text-red-500">🛡️</span> <Term explanation={GLOSSARY.adversary}>Adversary</Term> Agent
                                </h3>
                                <p className="text-sm text-muted-foreground">
                                    Goal: Resist payment using 7 tactics
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    {ADVERSARY_ACTIONS.map((action) => (
                                        <Badge key={action.id} variant="outline" className="text-xs">
                                            {action.icon} {action.name}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        </div>
                        <HumanTermsBox>
                            The collector practices against a simulated "difficult debtor" that tries every trick:
                            being aggressive, evasive, emotional, stalling... This way, when it faces real tough
                            situations, it's already prepared.
                        </HumanTermsBox>
                    </CardContent>
                </Card>
            </section>
        </div>
    );
}
