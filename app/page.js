import Link from "next/link";
import HeroActions from "@/components/HeroActions";
import FloorPlanThumb from "@/components/FloorPlanThumb";

const SAMPLES = [
  { name: "Studio · Lozenets", area: "38 m²", rooms: "1 room", variant: 0 },
  { name: "Two-bed · Mladost", area: "74 m²", rooms: "2 rooms", variant: 1 },
  {
    name: "Maisonette · Vitosha",
    area: "121 m²",
    rooms: "4 rooms",
    variant: 2,
  },
  { name: "Loft · Center", area: "56 m²", rooms: "open plan", variant: 3 },
];

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

      {/* Sample gallery */}
      <section className="container" style={{ padding: "24px 24px 64px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            marginBottom: 24,
            borderBottom: "1px solid var(--line)",
            paddingBottom: 14,
          }}
        >
          <h2 style={{ fontSize: 24 }}>Sample models</h2>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: "var(--ink-faint)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            4 examples
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))",
            gap: 18,
          }}
        >
          {SAMPLES.map((s) => (
            <Link
              key={s.name}
              href="/upload"
              className="card framed"
              style={{
                overflow: "hidden",
                transition: "border-color 0.15s ease",
              }}
            >
              <FloorPlanThumb variant={s.variant} />
              <div
                style={{
                  padding: "14px 16px",
                  borderTop: "1px solid var(--line)",
                }}
              >
                <div style={{ fontWeight: 500 }}>{s.name}</div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--ink-faint)",
                    marginTop: 4,
                  }}
                >
                  {s.area} · {s.rooms}
                </div>
              </div>
            </Link>
          ))}
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
