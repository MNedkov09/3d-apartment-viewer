import Link from "next/link";

export default function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--line)", background: "var(--paper-raised)" }}>
      <div
        className="container"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 32,
          justifyContent: "space-between",
          padding: "40px 24px",
        }}
      >
        <div style={{ maxWidth: 280 }}>
          <span
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: 600,
              fontSize: 18,
              letterSpacing: "-0.03em",
            }}
          >
            Plan<span style={{ color: "var(--accent)" }}>/</span>Space
          </span>
          <p style={{ color: "var(--ink-soft)", fontSize: 13, marginTop: 10 }}>
            Upload a floor plan, walk through it in 3D. An internship project by QuackStack, 2026.
          </p>
        </div>

        <FooterCol
          title="Product"
          links={[
            { href: "/", label: "Home" },
            { href: "/upload", label: "Upload & View" },
            { href: "/about", label: "About" },
          ]}
        />

        <FooterCol
          title="Links"
          links={[
            { href: "https://github.com/QuackStack-LTD", label: "GitHub", external: true },
            { href: "mailto:hello@quackstack.example", label: "Email", external: true },
          ]}
        />
      </div>

      <div
        className="container"
        style={{
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
          padding: "16px 24px",
          borderTop: "1px solid var(--line)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.08em",
          color: "var(--ink-faint)",
          textTransform: "uppercase",
        }}
      >
        <span>© {new Date().getFullYear()} QuackStack Ltd.</span>
        <span>Sofia · Bulgaria</span>
      </div>
    </footer>
  );
}

function FooterCol({ title, links }) {
  return (
    <div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--ink-faint)",
          marginBottom: 12,
        }}
      >
        {title}
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
        {links.map((l) => (
          <li key={l.href}>
            {l.external ? (
              <a
                href={l.href}
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: 14, color: "var(--ink-soft)" }}
              >
                {l.label}
              </a>
            ) : (
              <Link href={l.href} style={{ fontSize: 14, color: "var(--ink-soft)" }}>
                {l.label}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
