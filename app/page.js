import HeroActions from "@/components/HeroActions";

const STEPS = [
  {
    n: "01",
    t: "Upload a drawing",
    d: "Drag in a floor plan image, PDF, or a 3D model file from your machine.",
  },
  {
    n: "02",
    t: "We store it",
    d: "Every upload lands in your account and shows up in your recent panel.",
  },
  {
    n: "03",
    t: "View in 3D",
    d: "Open any plan in the viewport and walk the space in three dimensions.",
  },
];

export default function HomePage() {
  return (
    <>
      {/* Hero */}
      <section className="container" style={{ padding: "72px 24px 56px" }}>
        <div style={{ maxWidth: 720 }}>
          <span className="eyebrow">3D Apartment Viewer</span>
          <h1
            style={{
              fontSize: "clamp(40px, 6vw, 68px)",
              lineHeight: 1.02,
              marginTop: 18,
            }}
          >
            From flat drawing to a room you can walk through.
          </h1>
          <p
            style={{
              fontSize: 18,
              color: "var(--ink-soft)",
              marginTop: 20,
              maxWidth: 560,
            }}
          >
            Plan/Space turns apartment blueprints into navigable 3D space.
            Upload a plan, keep every version in one place, and explore the
            layout the way a person actually moves through it.
          </p>
          <HeroActions />
        </div>
      </section>

      {/* How it works */}
      <section
        style={{
          background: "var(--paper-raised)",
          borderTop: "1px solid var(--line)",
          borderBottom: "1px solid var(--line)",
        }}
      >
        <div className="container" style={{ padding: "56px 24px" }}>
          <span className="eyebrow">How it works</span>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: 28,
              marginTop: 28,
            }}
          >
            {STEPS.map((s) => (
              <div key={s.n}>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 13,
                    color: "var(--accent)",
                    letterSpacing: "0.1em",
                  }}
                >
                  {s.n}
                </div>
                <h3 style={{ fontSize: 20, marginTop: 10 }}>{s.t}</h3>
                <p
                  style={{
                    color: "var(--ink-soft)",
                    fontSize: 14,
                    marginTop: 8,
                  }}
                >
                  {s.d}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
