"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatsCardProps {
    title: string;
    value: string | number;
    description?: string;
    icon?: React.ReactNode;
    trend?: "up" | "down" | "neutral";
    className?: string;
}

export function StatsCard({
    title,
    value,
    description,
    icon,
    trend,
    className,
}: StatsCardProps) {
    return (
        <Card className={cn("", className)}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                    {title}
                </CardTitle>
                {icon && <div className="text-muted-foreground">{icon}</div>}
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{value}</div>
                {description && (
                    <p
                        className={cn(
                            "text-xs text-muted-foreground mt-1",
                            trend === "up" && "text-green-500",
                            trend === "down" && "text-red-500"
                        )}
                    >
                        {description}
                    </p>
                )}
            </CardContent>
        </Card>
    );
}
