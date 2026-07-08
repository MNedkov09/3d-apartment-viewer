// Self-contained SVG "floor plan" so the gallery needs no external images.
// `variant` picks one of a few hand-tuned layouts.

const LAYOUTS = [
  // Each layout is a set of internal wall segments inside a 200x140 frame.
  [
    "M110 8 V78", "M110 78 H192", "M110 78 V132", "M8 88 H110", "M60 88 V132",
  ],
  [
    "M8 60 H120", "M120 8 V60", "M120 60 V132", "M120 96 H192", "M60 60 V132",
  ],
  [
    "M96 8 V70", "M96 70 H8", "M96 70 V132", "M96 96 H192", "M150 8 V70",
  ],
  [
    "M8 74 H140", "M140 8 V132", "M70 8 V74", "M140 70 H192", "M100 74 V132",
  ],
];

export default function FloorPlanThumb({ variant = 0, style }) {
  const walls = LAYOUTS[variant % LAYOUTS.length];
  return (
    <svg viewBox="0 0 200 140" style={{ display: "block", width: "100%", ...style }} role="img" aria-label="Floor plan preview">
      <rect x="0" y="0" width="200" height="140" fill="var(--paper-sunk)" />
      {/* faint grid */}
      <g stroke="var(--line)" strokeWidth="0.5">
        {Array.from({ length: 9 }).map((_, i) => (
          <line key={`v${i}`} x1={i * 22 + 4} y1="6" x2={i * 22 + 4} y2="134" />
        ))}
        {Array.from({ length: 6 }).map((_, i) => (
          <line key={`h${i}`} x1="6" y1={i * 22 + 8} x2="196" y2={i * 22 + 8} />
        ))}
      </g>
      {/* outer wall */}
      <rect x="8" y="8" width="184" height="124" fill="none" stroke="var(--ink)" strokeWidth="2.5" />
      {/* internal walls */}
      <g stroke="var(--ink-soft)" strokeWidth="2" fill="none" strokeLinecap="square">
        {walls.map((d, i) => (
          <path key={i} d={d} />
        ))}
      </g>
      {/* a door swing, accent */}
      <path d="M8 108 A22 22 0 0 0 30 130" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
    </svg>
  );
}
