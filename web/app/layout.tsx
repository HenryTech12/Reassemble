import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Reassemble",
  description: "Digital reconstruction of shredded documents",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-lab-line/80 bg-lab-bg/80 backdrop-blur sticky top-0 z-30">
          <div className="max-w-[1500px] mx-auto px-4 h-12 flex items-center gap-3">
            <Link href="/" className="font-mono font-semibold tracking-tight text-white">
              <span className="text-lab-accent">▮▯▮▮▯▮</span> REASSEMBLE
            </Link>
            <span className="chip hidden sm:inline">document restoration lab</span>
            <div className="flex-1" />
            <a className="text-xs text-lab-dim hover:text-white" href="https://github.com/henrytech12/reassemble">source</a>
          </div>
        </header>
        <main className="max-w-[1500px] mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
