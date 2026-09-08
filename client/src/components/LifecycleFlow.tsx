/**
 * LifecycleFlow — scroll-triggered OBSERVE → DIAGNOSE → EXPERIMENT → VERIFY → LEARN
 * node diagram. SVG connecting paths drawn by Anime.js. Node activation via
 * IntersectionObserver. Motion.dev handles text panel transitions.
 */
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { animate } from "animejs";

export type LifecyclePhase =
  | "observe"
  | "diagnose"
  | "experiment"
  | "verify"
  | "learn"
  | null;

const NODES = [
  {
    id: "observe",
    index: "01",
    label: "OBSERVE",
    title: "Watch without touching.",
    body:
      "GhostWatcherHook attaches to a standard PyTorch loop and records GPU utilization, VRAM pressure, step timing, data-loading timing, and precision state. The training run does not change. The hook does not change the gradient.",
    stat: { label: "Telemetry channels", value: "8" },
  },
  {
    id: "diagnose",
    index: "02",
    label: "DIAGNOSE",
    title: "Map the symptom to the mechanism.",
    body:
      "The decision engine cross-references the recorded telemetry against a rule library covering mixed precision, FlashAttention, gradient checkpointing, data pipeline saturation, operator fusion, and compiler flags. Each candidate recommendation carries an evidence label.",
    stat: { label: "Rule surface", value: "Training runtime" },
  },
  {
    id: "experiment",
    index: "03",
    label: "EXPERIMENT",
    title: "Apply one change. Measure the delta.",
    body:
      "A single intervention is applied within a defined policy boundary. The optimized run is measured against the baseline step time and loss trajectory simultaneously. No change is retained until verification completes.",
    stat: { label: "Safety gate", value: "Loss trajectory" },
  },
  {
    id: "verify",
    index: "04",
    label: "VERIFY",
    title: "A blocked path is as useful as a verified one.",
    body:
      "When the loss-shift delta exceeds the configured threshold (default 0.10), automatic application is revoked and the candidate is downgraded to recommendation-only evidence. Both outcomes — verified and blocked — are written to the replay log.",
    stat: { label: "Block threshold", value: "Δ > 0.10" },
  },
  {
    id: "learn",
    index: "05",
    label: "LEARN",
    title: "The system compounds its memory.",
    body:
      "Verified outcomes can update the shared knowledge base. The replay log retains OBSERVE, DIAGNOSE, VERIFY, and ROLLBACK events with full context. The ROI calculator converts a measured step-time improvement into a GPU-hour and cost receipt.",
    stat: { label: "Memory events", value: "4 types" },
  },
] as const;

interface LifecycleFlowProps {
  onPhaseChange?: (phase: LifecyclePhase) => void;
}

export default function LifecycleFlow({ onPhaseChange }: LifecycleFlowProps) {
  const [activePhase, setActivePhase] = useState<LifecyclePhase>(null);
  const [visibleNodes, setVisibleNodes] = useState<Set<string>>(new Set());
  const nodeRefs = useRef<Map<string, HTMLElement>>(new Map());
  const svgPathRefs = useRef<Map<string, SVGPathElement>>(new Map());
  const animatedPaths = useRef<Set<string>>(new Set());

  // IntersectionObserver to detect when each node enters viewport
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const id = (entry.target as HTMLElement).dataset.nodeId;
          if (!id) return;
          if (entry.isIntersecting) {
            setVisibleNodes((prev) => new Set([...prev, id]));
            setActivePhase(id as LifecyclePhase);
            onPhaseChange?.(id as LifecyclePhase);

            // Animate the SVG connector path into this node
            const pathEl = svgPathRefs.current.get(`path-${id}`);
            if (pathEl && !animatedPaths.current.has(id)) {
              animatedPaths.current.add(id);
              const length = pathEl.getTotalLength();
              animate(pathEl, {
                strokeDashoffset: [length, 0],
                duration: 600,
                ease: "outQuart",
                delay: 100,
              });
            }
          }
        });
      },
      { threshold: 0.5, rootMargin: "0px 0px -15% 0px" }
    );

    nodeRefs.current.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [onPhaseChange]);

  function registerNodeRef(id: string, el: HTMLElement | null) {
    if (el) nodeRefs.current.set(id, el);
  }

  function registerPathRef(id: string, el: SVGPathElement | null) {
    if (el) {
      svgPathRefs.current.set(id, el);
      // Initialize path to hidden
      const length = el.getTotalLength();
      el.style.strokeDasharray = String(length);
      el.style.strokeDashoffset = String(length);
    }
  }

  return (
    <section className="lifecycle-flow" aria-label="GhostLayer optimization lifecycle">
      <div className="lifecycle-flow__intro">
        <p className="orbit-eyebrow"><span />THE OPERATING LOOP</p>
        <h2>One training step.<br /><em>Five compounding moves.</em></h2>
        <p>
          GhostLayer is not a monitoring dashboard. It is a decision loop that
          observes, experiments, verifies, and remembers — building institutional
          memory for GPU infrastructure.
        </p>
      </div>

      <div className="lifecycle-flow__body">
        {/* SVG connector rail */}
        <svg
          className="lifecycle-rail"
          viewBox="0 0 2 500"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {NODES.slice(0, -1).map((node, i) => {
            const y1 = (i / (NODES.length - 1)) * 500;
            const y2 = ((i + 1) / (NODES.length - 1)) * 500;
            const pathId = `path-${NODES[i + 1].id}`;
            return (
              <path
                key={pathId}
                ref={(el) => registerPathRef(pathId, el)}
                d={`M 1 ${y1} L 1 ${y2}`}
                stroke={visibleNodes.has(NODES[i + 1].id) ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.1)"}
                strokeWidth="1"
                fill="none"
                style={{ transition: "stroke 400ms" }}
              />
            );
          })}
        </svg>

        <div className="lifecycle-nodes">
          {NODES.map((node) => {
            const isVisible = visibleNodes.has(node.id);
            const isActive = activePhase === node.id;
            return (
              <motion.div
                key={node.id}
                ref={(el) => registerNodeRef(node.id, el as HTMLElement | null)}
                data-node-id={node.id}
                className={`lifecycle-node ${isActive ? "is-active" : ""} ${isVisible ? "is-visible" : ""}`}
                initial={{ opacity: 0, x: -20 }}
                animate={isVisible ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
                transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
                onClick={() => {
                  setActivePhase(node.id as LifecyclePhase);
                  onPhaseChange?.(node.id as LifecyclePhase);
                }}
              >
                <div className="lifecycle-node__marker">
                  <span className="lifecycle-node__index">{node.index}</span>
                  <span className="lifecycle-node__dot" aria-hidden="true" />
                </div>
                <div className="lifecycle-node__content">
                  <p className="lifecycle-node__phase">{node.label}</p>
                  <h3 className="lifecycle-node__title">{node.title}</h3>
                  <AnimatePresence>
                    {isActive && (
                      <motion.div
                        className="lifecycle-node__detail"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.38, ease: [0.23, 1, 0.32, 1] }}
                      >
                        <p>{node.body}</p>
                        <div className="lifecycle-node__stat">
                          <span>{node.stat.label}</span>
                          <strong>{node.stat.value}</strong>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
