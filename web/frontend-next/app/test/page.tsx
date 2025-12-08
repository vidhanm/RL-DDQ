"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const plannedFeatures = [
    "🎙️ Speech-to-Text (Whisper)",
    "🔊 Text-to-Speech (ElevenLabs)",
    "📡 Real-time WebRTC via LiveKit",
    "🧠 NLU State Extraction",
    "🎯 RL Action Selection",
];

export default function TestPage() {
    return (
        <div className="flex justify-center items-center min-h-[60vh]">
            <Card className="max-w-lg text-center">
                <CardHeader>
                    <div className="flex justify-center mb-2">
                        <Badge variant="secondary" className="text-sm">
                            Coming Soon
                        </Badge>
                    </div>
                    <CardTitle className="text-2xl">LiveKit Voice Integration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                    <p className="text-muted-foreground">
                        Soon you&apos;ll be able to test the DDQ agent with real voice
                        conversations. Speak as a debtor and watch the RL agent adapt in
                        real-time!
                    </p>

                    <div className="text-left">
                        <h4 className="font-semibold mb-3">Planned Features:</h4>
                        <ul className="space-y-2">
                            {plannedFeatures.map((feature, idx) => (
                                <li
                                    key={idx}
                                    className="py-2 border-b last:border-0 text-sm text-muted-foreground"
                                >
                                    {feature}
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="pt-4">
                        <Link href="/">
                            <Button variant="outline">← Back to Demo</Button>
                        </Link>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
