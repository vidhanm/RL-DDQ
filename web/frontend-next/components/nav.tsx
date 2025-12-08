"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
    { href: "/", label: "Demo" },
    { href: "/train", label: "Train" },
    { href: "/evaluate", label: "Evaluate" },
    { href: "/test", label: "Test" },
    { href: "/adversarial", label: "⚔️ Adversarial" },
    { href: "/history", label: "📜 History" },
];

export function Nav() {
    const pathname = usePathname();

    return (
        <header className="border-b bg-card">
            <div className="container mx-auto px-4 py-4">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h1 className="text-2xl font-bold">DDQ Debt Collection Agent</h1>
                        <p className="text-sm text-muted-foreground">
                            Interactive RL Agent Demo | NLU + Domain Randomization
                        </p>
                    </div>
                    <nav className="flex gap-1">
                        {navItems.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    "px-4 py-2 rounded-md text-sm font-medium transition-colors",
                                    pathname === item.href
                                        ? "bg-primary text-primary-foreground"
                                        : "hover:bg-accent hover:text-accent-foreground"
                                )}
                            >
                                {item.label}
                            </Link>
                        ))}
                    </nav>
                </div>
            </div>
        </header>
    );
}
