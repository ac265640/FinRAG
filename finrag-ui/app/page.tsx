"use client";

import Link from "next/link";
import Galaxy from "@/components/Galaxy";
import GradientText from "@/components/GradientText";
import FinRAGLogo from "@/components/FinRAGLogo";
import { motion } from "motion/react";

export default function LandingPage() {
  return (
    <div className="min-h-screen w-full bg-black text-white selection:bg-primary/30 relative overflow-hidden flex flex-col">
      
      {/* ── Background ── */}
      <div className="absolute inset-0 z-0 opacity-80">
        <Galaxy 
          mouseRepulsion={true}
          mouseInteraction={true}
          density={1.2}
          glowIntensity={0.4}
          saturation={0}
          hueShift={140}
          twinkleIntensity={0.4}
          rotationSpeed={0.1}
          repulsionStrength={2}
          starSpeed={0.3}
          speed={0.8}
        />
      </div>

      {/* ── Header ── */}
      <header className="relative z-10 flex items-center justify-between px-6 py-6 max-w-7xl mx-auto w-full">
        <div className="flex items-center gap-2">
          <FinRAGLogo size="lg" />
        </div>
        <Link 
          href="/chat" 
          className="text-sm font-medium px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20 transition-all backdrop-blur-md"
        >
          Enter Platform
        </Link>
      </header>

      {/* ── Main Content ── */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 sm:px-6 text-center max-w-5xl mx-auto w-full py-12">
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="flex flex-col items-center gap-6"
        >
          {/* Title with Gradient Text */}
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold tracking-tight text-white mb-2 leading-tight">
            <GradientText
              colors={["#ff4444", "#ff8888", "#ff4444"]}
              animationSpeed={6}
              showBorder={false}
              className="inline-block mt-2"
            >
              Financial Intelligence.
            </GradientText>
          </h1>

          {/* Description */}
          <p className="text-lg sm:text-xl text-white/60 max-w-3xl leading-relaxed mt-4 font-light">
            FinRAG is an enterprise-grade financial data intelligence platform driven by advanced RAG architectures. 
            Automate your financial workflows and perform predictive risk analysis over SEC EDGAR 10-K, 10-Q, and 8-K filings. 
            Leverage AI-powered analytics to extract real-time market intelligence with scalable, data-driven decision systems.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center gap-4 mt-8">
            <Link 
              href="/chat"
              className="group relative px-8 py-4 bg-primary text-white font-medium rounded-full overflow-hidden transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(239,68,68,0.4)]"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
              <span className="relative flex items-center gap-2">
                Start Chatting
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:translate-x-1 transition-transform">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </span>
            </Link>
          </div>
        </motion.div>

        {/* ── Features Grid ── */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full mt-32 mb-16"
          id="features"
        >
          {/* Feature 1 */}
          <div className="flex flex-col items-center md:items-start text-center md:text-left p-8 rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-sm hover:bg-white/[0.04] transition-colors">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary mb-6 border border-primary/20">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Citation Enforced</h3>
            <p className="text-sm text-white/50 leading-relaxed">
              Every answer generated is explicitly linked to the exact paragraph in the SEC filing. Total transparency, zero hallucinations.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="flex flex-col items-center md:items-start text-center md:text-left p-8 rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-sm hover:bg-white/[0.04] transition-colors">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary mb-6 border border-primary/20">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M3 9h18M9 21V9" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Deep SEC Integration</h3>
            <p className="text-sm text-white/50 leading-relaxed">
              Native understanding of complex 10-K, 10-Q, and 8-K document structures, financial tables, and corporate disclosures.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="flex flex-col items-center md:items-start text-center md:text-left p-8 rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-sm hover:bg-white/[0.04] transition-colors">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary mb-6 border border-primary/20">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Real-time Analysis</h3>
            <p className="text-sm text-white/50 leading-relaxed">
              Instantly summarize reports, extract risk factors, and analyze management's future guidance at lightning speed.
            </p>
          </div>
        </motion.div>

      </main>

      {/* ── Footer ── */}
      <footer className="relative z-10 border-t border-white/10 py-8 text-center text-sm text-white/40">
        <p>© {new Date().getFullYear()} FinRAG. Advanced Financial Research System.</p>
      </footer>
      
    </div>
  );
}
