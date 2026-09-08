/**
 * RoiProofTable — forensic ROI comparison table.
 * Anime.js counts numbers from 0 when the section enters viewport.
 * Thin hairline borders, monospace values, zero rounded corners.
 */
import { useEffect, useRef } from "react";
import { animate } from "animejs";

const BASELINE = {
  step_ms: 31.27,
  gpu_hours: 1840,
  throughput: 421,
  cost_per_job: 4920,
};

const OPTIMIZED = {
  step_ms: 10.74,
  gpu_hours: 1514,
  throughput: 493,
  cost_per_job: 4046,
};

const ROWS = [
  {
    label: "Step time",
    baseRaw: BASELINE.step_ms,
    optRaw: OPTIMIZED.step_ms,
    suffix: "ms",
    decimals: 2,
    lower_is_better: true,
  },
  {
    label: "GPU hours / job",
    baseRaw: BASELINE.gpu_hours,
    optRaw: OPTIMIZED.gpu_hours,
    suffix: "",
    decimals: 0,
    lower_is_better: true,
  },
  {
    label: "Throughput",
    baseRaw: BASELINE.throughput,
    optRaw: OPTIMIZED.throughput,
    suffix: "/s",
    decimals: 0,
    lower_is_better: false,
  },
  {
    label: "Cost / job",
    baseRaw: BASELINE.cost_per_job,
    optRaw: OPTIMIZED.cost_per_job,
    prefix: "$",
    suffix: "",
    decimals: 0,
    lower_is_better: true,
  },
];

const SAVINGS = BASELINE.cost_per_job - OPTIMIZED.cost_per_job;

export default function RoiProofTable() {
  const sectionRef = useRef<HTMLElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const targets = el.querySelectorAll<HTMLElement>("[data-roi-count]");
          targets.forEach((target) => {
            const to = parseFloat(target.dataset.roiCount ?? "0");
            const suffix = target.dataset.suffix ?? "";
            const prefix = target.dataset.prefix ?? "";
            const decimals = parseInt(target.dataset.decimals ?? "0");
            const obj = { n: 0 };
            animate(obj, {
              n: to,
              duration: 1400,
              delay: 200,
              ease: "outQuart",
              onUpdate() {
                target.textContent =
                  prefix +
                  (decimals > 0 ? obj.n.toFixed(decimals) : Math.floor(obj.n).toLocaleString()) +
                  suffix;
              },
            });
          });
        }
      },
      { threshold: 0.4 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} className="roi-proof" aria-label="ROI proof table">
      <div className="roi-proof__header">
        <p className="orbit-eyebrow"><span />VERIFIED ECONOMIC IMPACT</p>
        <h2>What a verified run actually produces.</h2>
        <p>
          These are measurements from a real hardware run on an NVIDIA RTX A2000,
          not modeled estimates. The safety verifier confirmed the loss delta stayed
          within the 0.10 threshold before any result was retained.
        </p>
      </div>

      <div className="roi-table-wrapper">
        <table className="roi-table" aria-label="Baseline vs optimized comparison">
          <thead>
            <tr>
              <th scope="col">METRIC</th>
              <th scope="col">BASELINE</th>
              <th scope="col">OPTIMIZED</th>
              <th scope="col">DELTA</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const deltaRaw = row.lower_is_better
                ? row.baseRaw - row.optRaw
                : row.optRaw - row.baseRaw;
              const deltaSign = deltaRaw >= 0 ? "+" : "–";
              const deltaMag = Math.abs(deltaRaw);
              const prefix = row.prefix ?? "";

              return (
                <tr key={row.label}>
                  <td className="roi-table__label">{row.label}</td>
                  <td className="roi-table__base">
                    <span
                      data-roi-count={row.baseRaw}
                      data-suffix={row.suffix}
                      data-prefix={prefix}
                      data-decimals={row.decimals}
                    >
                      {prefix}0{row.suffix}
                    </span>
                  </td>
                  <td className="roi-table__opt">
                    <span
                      data-roi-count={row.optRaw}
                      data-suffix={row.suffix}
                      data-prefix={prefix}
                      data-decimals={row.decimals}
                    >
                      {prefix}0{row.suffix}
                    </span>
                  </td>
                  <td className="roi-table__delta roi-table__delta--positive">
                    {deltaSign}&thinsp;
                    <span
                      data-roi-count={deltaMag}
                      data-suffix={row.suffix}
                      data-prefix=""
                      data-decimals={row.decimals}
                    >
                      0{row.suffix}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="roi-proof__bottom">
          <div className="roi-savings">
            <span className="roi-savings__label">REALIZED SAVINGS / JOB</span>
            <strong className="roi-savings__value">
              $
              <span
                data-roi-count={SAVINGS}
                data-suffix=""
                data-prefix=""
                data-decimals="0"
              >
                0
              </span>
            </strong>
          </div>
          <p className="roi-proof__footnote">
            Source: Physical GPU Hardware Evidence Audit. RTX A2000, 52.2M parameter LLM.
            Maximum recorded loss delta: 0.0012. These are isolated project-specific measurements,
            not fleet-wide guarantees. "Verified Safe" reflects single-run loss proxy compliance (&lt;&nbsp;0.10) only.
            Review our <a href="/legal" style={{ color: "#10b981", textDecoration: "underline" }}>Legal Disclaimers</a>.
          </p>
        </div>
      </div>
    </section>
  );
}
