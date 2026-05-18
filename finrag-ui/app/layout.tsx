import type { Metadata, Viewport } from "next";
import { Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "FinRAG — AI Financial Research",
  description: "Citation-enforced financial research over SEC EDGAR 10-K, 10-Q, and 8-K filings.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#000000" },
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${ibmPlexMono.variable} antialiased dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen font-sans" style={{ background: "#000", color: "rgba(255,255,255,0.9)", overflow: "hidden" }}>
        {children}
      </body>
    </html>
  );
}
