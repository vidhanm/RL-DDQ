"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";
import { cn } from "@/lib/utils";

const ACTION_NAMES = [
    "Empathetic Listening",
    "Ask About Situation",
    "Firm Reminder",
    "Offer Payment Plan",
    "Propose Settlement",
    "Hard Close",
    "Schedule Callback",
    "Escalate",
    "End Call",
];

interface QValueChartProps {
    qValues: number[];
    selectedAction?: number;
    className?: string;
}

export function QValueChart({
    qValues,
    selectedAction,
    className,
}: QValueChartProps) {
    const data = qValues.map((value, index) => ({
        action: ACTION_NAMES[index] || `Action ${index}`,
        value: value,
        shortName: ACTION_NAMES[index]?.split(" ")[0] || `A${index}`,
    }));

    const maxValue = Math.max(...qValues);

    return (
        <Card className={cn("", className)}>
            <CardHeader className="pb-2">
                <CardTitle className="text-base">Q-Value Analysis</CardTitle>
            </CardHeader>
            <CardContent>
                {qValues.length === 0 ? (
                    <p className="text-center text-muted-foreground py-8 text-sm">
                        No Q-values available yet
                    </p>
                ) : (
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={data} layout="vertical">
                            <XAxis type="number" domain={["auto", "auto"]} fontSize={12} />
                            <YAxis
                                type="category"
                                dataKey="shortName"
                                width={80}
                                fontSize={11}
                            />
                            <Tooltip
                                formatter={(value: number) => [value.toFixed(4), "Q-Value"]}
                                labelFormatter={(label) =>
                                    data.find((d) => d.shortName === label)?.action || label
                                }
                                contentStyle={{
                                    backgroundColor: "hsl(var(--card))",
                                    border: "1px solid hsl(var(--border))",
                                    borderRadius: "var(--radius)",
                                }}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {data.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={
                                            index === selectedAction
                                                ? "hsl(var(--primary))"
                                                : entry.value === maxValue
                                                    ? "hsl(var(--chart-1))"
                                                    : "hsl(var(--muted-foreground) / 0.3)"
                                        }
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                )}
            </CardContent>
        </Card>
    );
}
