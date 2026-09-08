/**
 * LiveSystemPanel — ambient "operating system" status readout for the hero.
 * Numbers count up via Anime.js on mount. Status dot pulses via Anime.js.
 * Pure DM Mono typography, zero color.
 */
import { useEffect, useRef } from "react";
import { animate } from "animejs";

interface Metric {
  label: string;
  value: string;
  raw?: number;
  suffix?: string;
}

const METRICS: Metric[] = [
  { label: "GPU fleet", value: "1,284", raw: 1284, suffix: "" },
  { label: "Active workloads", value: "47", raw: 47, suffix: "" },
  { label: "Verified savings", value: "$2.41M", raw: 2.41, suffix: "M" },
  { label: "System status", value: "NOMINAL" },
];

export default function LiveSystemPanel() {
  const panelRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLSpanElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (hasAnimated.current) return;
    hasAnimated.current = true;

    // Pulse the status dot
    if (dotRef.current) {
      animate(dotRef.current, {
        opacity: [0.3, 1, 0.3],
        scale: [0.85, 1.1, 0.85],
        duration: 2400,
        ease: "inOutSine",
        loop: true,
      });
    }

    // Count up numeric values
    if (!panelRef.current) return;
    const numberEls = panelRef.current.querySelectorAll<HTMLElement>("[data-countup]");
    numberEls.forEach((el) => {
      const target = parseFloat(el.dataset.countup ?? "0");
      const suffix = el.dataset.suffix ?? "";
      const prefix = el.dataset.prefix ?? "";
      const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
      const obj = { n: 0 };
      animate(obj, {
        n: target,
        duration: 1600,
        delay: 300,
        ease: "outQuart",
        onUpdate() {
          el.textContent =
            prefix +
            (decimals > 0
              ? obj.n.toFixed(decimals)
              : Math.floor(obj.n).toLocaleString()) +
            suffix;
        },
      });
    });
  }, []);

  return (
    <div ref={panelRef} className="live-system-panel" aria-label="GhostLayer system status">
      <div className="lsp-header">
        <span ref={dotRef} className="lsp-dot" aria-hidden="true" />
        <span className="lsp-label">GHOSTLAYER ENGINE</span>
        <span className="lsp-status-tag">ONLINE</span>
      </div>
      <div className="lsp-metrics">
        {METRICS.map((metric) => (
          <div key={metric.label} className="lsp-row">
            <span className="lsp-row__label">{metric.label}</span>
            {metric.raw !== undefined ? (
              <strong
                className="lsp-row__value"
                style={metric.label === "Verified savings" ? { color: "#10b981", fontWeight: 600 } : undefined}
                data-countup={metric.value === "$2.41M" ? "2.41" : metric.raw}
                data-suffix={metric.value === "$2.41M" ? "M" : ""}
                data-prefix={metric.value === "$2.41M" ? "$" : ""}
                data-decimals={metric.value === "$2.41M" ? "2" : "0"}
              >
                0
              </strong>
            ) : (
              <strong 
                className="lsp-row__value lsp-row__value--text"
                style={metric.value === "NOMINAL" ? { color: "#10b981", fontWeight: 600 } : undefined}
              >
                {metric.value}
              </strong>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
