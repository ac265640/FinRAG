"use client";

import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  radius: number;
  opacity: number;
  twinkleSpeed: number;
  twinklePhase: number;
}

interface Streak {
  x: number;
  y: number;
  length: number;
  angle: number;
  speed: number;
  opacity: number;
  life: number;
  maxLife: number;
}

export default function StarfieldBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let stars: Star[] = [];
    let streaks: Streak[] = [];
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initStars();
    };

    const initStars = () => {
      const count = Math.floor((canvas.width * canvas.height) / 6000);
      stars = Array.from({ length: Math.min(count, 200) }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 1.2 + 0.2,
        opacity: Math.random() * 0.7 + 0.1,
        twinkleSpeed: Math.random() * 0.015 + 0.005,
        twinklePhase: Math.random() * Math.PI * 2,
      }));
    };

    const spawnStreak = () => {
      // Randomly spawn a shooting streak from upper area
      const angle = (Math.random() * 30 + 20) * (Math.PI / 180); // 20–50 degrees
      streaks.push({
        x: Math.random() * canvas.width * 0.8 + canvas.width * 0.1,
        y: Math.random() * canvas.height * 0.4,
        length: Math.random() * 120 + 80,
        angle,
        speed: Math.random() * 3 + 2,
        opacity: Math.random() * 0.6 + 0.2,
        life: 0,
        maxLife: Math.random() * 80 + 60,
      });
    };

    let streakTimer = 0;
    const STREAK_INTERVAL = 200; // frames between spawns

    const draw = () => {
      // Clear with pure black background
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      time += 1;
      streakTimer += 1;

      // Occasionally spawn a streak
      if (streakTimer > STREAK_INTERVAL && Math.random() < 0.02) {
        spawnStreak();
        streakTimer = 0;
      }

      // Draw stars
      for (const star of stars) {
        const twinkle = Math.sin(time * star.twinkleSpeed + star.twinklePhase);
        const currentOpacity = star.opacity * (0.6 + 0.4 * twinkle);

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);

        // Slightly cool/white tint
        const brightness = Math.floor(200 + 55 * (0.5 + 0.5 * twinkle));
        ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${Math.min(255, brightness + 20)}, ${currentOpacity})`;
        ctx.fill();
      }

      // Draw & update streaks
      streaks = streaks.filter((s) => s.life < s.maxLife);
      for (const streak of streaks) {
        streak.life += 1;
        const progress = streak.life / streak.maxLife;
        const fadeOpacity = streak.opacity * (1 - progress) * (progress < 0.3 ? progress / 0.3 : 1);

        const dx = Math.cos(streak.angle) * streak.speed;
        const dy = Math.sin(streak.angle) * streak.speed;
        streak.x += dx;
        streak.y += dy;

        const tailX = streak.x - Math.cos(streak.angle) * streak.length;
        const tailY = streak.y - Math.sin(streak.angle) * streak.length;

        const grad = ctx.createLinearGradient(tailX, tailY, streak.x, streak.y);
        grad.addColorStop(0, `rgba(255,255,255,0)`);
        grad.addColorStop(0.7, `rgba(200,220,255,${fadeOpacity * 0.5})`);
        grad.addColorStop(1, `rgba(255,255,255,${fadeOpacity})`);

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(streak.x, streak.y);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      animationId = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    draw();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
