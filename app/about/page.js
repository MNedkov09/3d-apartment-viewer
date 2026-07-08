export const metadata = {
  title: "About — Plan/Space",
};

const FACTS = [
  { k: "Team", v: "QuackStack interns, cohort 2026" },
  { k: "Base", v: "Sofia, Bulgaria" },
  { k: "Stack", v: "Next.js · SQLite · Three.js" },
  { k: "Status", v: "Work in progress" },
];

const LINKS = [
  { label: "GitHub organization", href: "https://github.com/QuackStack-LTD" },
  { label: "Project repository", href: "https://github.com/QuackStack-LTD/interns-2026-3d-apartment-viewer" },
  { label: "Email us", href: "mailto:hello@quackstack.example" },
];

export default function AboutPage() {
  return (
    <div className="container" style={{ padding: "72px 24px", maxWidth: 820 }}>
      <span className="eyebrow">About us</span>
      <h1 style={{ fontSize: "clamp(36px, 5vw, 52px)", lineHeight: 1.05, marginTop: 18 }}>
        A small team teaching apartments to stand up in 3D.
      </h1>

      <p style={{ fontSize: 18, color: "var(--ink-soft)", marginTop: 22 }}>
        Plan/Space started as a 2026 internship project at QuackStack. The idea is simple:
        floor plans are hard to read as flat lines, so we let you upload one and see the same
        space in three dimensions. Along the way we&apos;re learning how to build a real product —
        accounts, file storage, a viewer, and everything that holds it together.
      </p>

      <p style={{ fontSize: 16, color: "var(--ink-soft)", marginTop: 16 }}>
        We&apos;re students and junior developers, so this is as much a learning space as a product.
        Feedback, issues, and pull requests are all welcome.
      </p>

      {/* Facts */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 0,
          marginTop: 44,
          border: "1px solid var(--line)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
        }}
      >
        {FACTS.map((f, i) => (
          <div
            key={f.k}
            style={{
              padding: "18px 20px",
              borderRight: "1px solid var(--line)",
              borderBottom: "1px solid var(--line)",
              background: "var(--paper-raised)",
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--ink-faint)",
              }}
            >
              {f.k}
            </div>
            <div style={{ marginTop: 6, fontWeight: 500 }}>{f.v}</div>
          </div>
        ))}
      </div>

      {/* Links */}
      <h2 style={{ fontSize: 22, marginTop: 52 }}>Our links</h2>
      <ul style={{ listStyle: "none", padding: 0, margin: "18px 0 0", display: "grid", gap: 2 }}>
        {LINKS.map((l) => (
          <li key={l.href}>
            <a
              href={l.href}
              target="_blank"
              rel="noreferrer"
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "16px 4px",
                borderBottom: "1px solid var(--line)",
              }}
            >
              <span style={{ fontSize: 16 }}>{l.label}</span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--accent)" }}>↗</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
