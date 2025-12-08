import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DDQ Debt Collection Agent",
  description: "Interactive RL Agent Demo | Deep Q-Learning with Dyna",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} antialiased font-sans min-h-screen`}>
        <div className="min-h-screen flex flex-col">
          <Nav />
          <main className="flex-1 container mx-auto px-4 py-6">
            {children}
          </main>
          <footer className="border-t py-4 text-center text-sm text-muted-foreground">
            DDQ Debt Collection Agent Demo | Reinforcement Learning in Conversational AI
          </footer>
        </div>
      </body>
    </html>
  );
}
