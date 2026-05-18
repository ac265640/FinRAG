"use client";

/**
 * FinRAGLogo — Modular logo component.
 *
 * TO REPLACE WITH FINAL LOGO:
 *   Option A (SVG):  Replace the <svg> element inside LogoMark with your own SVG.
 *   Option B (Image): Replace <svg> with <img src="/logo.png" alt="FinRAG" className="w-9 h-9 object-contain" />
 *                     and place your logo file in /public/logo.png
 *
 * The glow animation and sizing are controlled by the wrapper — no changes needed there.
 */

export default function FinRAGLogo({ size = "lg" }: { size?: "sm" | "md" | "lg" }) {
  const sizeMap = {
    sm: { icon: 28, text: "text-lg" },
    md: { icon: 34, text: "text-xl" },
    lg: { icon: 40, text: "text-2xl" },
  };
  const { icon, text } = sizeMap[size];

  return (
    <div className="finrag-logo-wrapper flex items-center gap-3 select-none">
      {/* ── Logo Mark ── swap this block with your final logo ── */}
      <div
        className="finrag-logo-icon"
        style={{ width: icon, height: icon }}
        aria-hidden="true"
      >
        {/* Robot icon from the FinRAG brand */}
        <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" width={icon} height={icon}>
          {/* Head */}
          <rect x="8" y="12" width="24" height="20" rx="10" fill="currentColor" opacity="0.12" stroke="currentColor" strokeWidth="1.5"/>
          {/* Eyes */}
          <circle cx="15" cy="21" r="2.5" fill="currentColor"/>
          <circle cx="25" cy="21" r="2.5" fill="currentColor"/>
          {/* Mouth line */}
          <path d="M15 27 Q20 29.5 25 27" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
          {/* Antenna */}
          <line x1="20" y1="12" x2="20" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          <circle cx="20" cy="6" r="1.5" fill="currentColor"/>
          {/* Ears */}
          <rect x="4" y="17" width="4" height="7" rx="2" fill="currentColor" opacity="0.7" stroke="currentColor" strokeWidth="1"/>
          <rect x="32" y="17" width="4" height="7" rx="2" fill="currentColor" opacity="0.7" stroke="currentColor" strokeWidth="1"/>
        </svg>
      </div>
      {/* ── Wordmark ── */}
      <span className={`finrag-logo-text font-bold tracking-tight ${text}`}>
        FinRAG
      </span>
    </div>
  );
}
