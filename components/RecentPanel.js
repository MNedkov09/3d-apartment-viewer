"use client";

function timeAgo(iso) {
  // SQLite datetime('now') is UTC without a "Z" — mark it so JS parses correctly.
  const then = new Date(iso.replace(" ", "T") + "Z").getTime();
  const s = Math.floor((Date.now() - then) / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function RecentPanel({ items, activeId, onSelect, loading, onClear, onDelete }) {
  return (
    <aside
      className="card"
      style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid var(--line)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span className="eyebrow" style={{ margin: 0 }}>
          Recent
        </span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-faint)" }}>
            {items?.length || 0}
          </span>
          {items?.length > 0 && (
            <button
              onClick={onClear}
              title="Clear all"
              style={{
                fontSize: 11,
                color: "var(--ink-faint)",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 0,
                lineHeight: 1,
              }}
              onMouseEnter={(e) => (e.target.style.color = "var(--ink)")}
              onMouseLeave={(e) => (e.target.style.color = "var(--ink-faint)")}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div style={{ maxHeight: 480, overflowY: "auto" }}>
        {loading && (
          <p style={{ padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--ink-faint)" }}>
            Loading…
          </p>
        )}

        {!loading && (!items || items.length === 0) && (
          <p style={{ padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--ink-faint)" }}>
            No blueprints yet. Your uploads will appear here.
          </p>
        )}

        {!loading &&
          items?.map((b) => {
            const active = b.id === activeId;
            return (
              <div
                key={b.id}
                style={{
                  borderBottom: "1px solid var(--line)",
                  borderLeft: `3px solid ${active ? "var(--accent)" : "transparent"}`,
                  background: active ? "var(--accent-wash)" : "transparent",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <button
                  onClick={() => onSelect(b)}
                  style={{
                    flex: 1,
                    width: "100%",
                    textAlign: "left",
                    border: "none",
                    background: "transparent",
                    padding: "12px 16px",
                    cursor: "pointer",
                    display: "block",
                  }}
                >
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 500,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {b.title}
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      color: "var(--ink-faint)",
                      marginTop: 3,
                    }}
                  >
                    {timeAgo(b.created_at)}
                  </div>
                </button>
                {onDelete && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(b);
                    }}
                    title="Delete"
                    style={{
                      fontSize: 13,
                      color: "var(--ink-faint)",
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      padding: "4px 8px",
                      marginRight: 4,
                      lineHeight: 1,
                    }}
                    onMouseEnter={(e) => (e.target.style.color = "var(--ink)")}
                    onMouseLeave={(e) => (e.target.style.color = "var(--ink-faint)")}
                  >
                    ✕
                  </button>
                )}
              </div>
            );
          })}
      </div>
    </aside>
  );
}
