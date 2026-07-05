import { useEffect, useRef } from "react";

/**
 * Draws the uploaded image on a canvas, and — once a polygon has been
 * detected — draws the detected outline on top in red so you can visually
 * check accuracy against the original drawing.
 *
 * Two-canvas trick: `imageCanvasRef` holds a clean copy of the raw image
 * (this is what gets fed to OpenCV for detection). `overlayCanvasRef` sits
 * absolutely positioned on top and only draws the polygon + image, so we
 * never have to worry about the red outline itself getting picked up by
 * the next detection pass.
 */
export default function ImageCanvas({ image, polygon, width, height }) {
  const overlayCanvasRef = useRef(null);

  // Draw image + polygon overlay
  useEffect(() => {
    if (!image || !overlayCanvasRef.current) return;
    const canvas = overlayCanvasRef.current;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    if (polygon && polygon.length > 0) {
      ctx.strokeStyle = "#e5484d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      polygon.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();

      // Draw corner markers so it's easy to see exactly where each
      // detected vertex landed
      ctx.fillStyle = "#e5484d";
      polygon.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [image, polygon, width, height]);

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={overlayCanvasRef}
        style={{
          border: "1px solid #333",
          borderRadius: 4,
          maxWidth: "100%",
          background: "#fff",
        }}
      />
    </div>
  );
}
