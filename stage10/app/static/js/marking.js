// Farmer-guided symptom marking: pure geometry and list operations (no DOM). Marks are normalised [0,1] coordinates
// of the photo AS DISPLAYED (browsers apply EXIF orientation), so they survive resizing and rotation of the view.
// A mark is USER-SELECTED; it is never presented as AI-detected and never changes the assessment.

export const MAX_MARKS = 10;
export const TAP_RADIUS = 0.06;      // circle radius for a tap, as a fraction of the shorter image side
export const DRAG_MIN = 0.02;        // smaller drags count as taps
export const GRID = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"];

const clamp = (v) => Math.min(1, Math.max(0, v));
const r4 = (v) => Math.round(v * 10000) / 10000;

// Where the image content sits inside its element for object-fit: contain (letterboxing).
export function contentBox(elW, elH, natW, natH) {
  if (!(elW > 0 && elH > 0 && natW > 0 && natH > 0)) return null;
  const s = Math.min(elW / natW, elH / natH);
  const w = natW * s, h = natH * s;
  return { x: (elW - w) / 2, y: (elH - h) / 2, w, h };
}

// Pointer position (client coordinates) -> normalised image coordinates, or null outside the image content.
export function toNormalized(clientX, clientY, rect, natW, natH) {
  const cb = contentBox(rect.width, rect.height, natW, natH);
  if (!cb) return null;
  const x = (clientX - rect.left - cb.x) / cb.w;
  const y = (clientY - rect.top - cb.y) / cb.h;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x: r4(x), y: r4(y) };
}

// Normalised -> pixels for drawing at any display size.
export function toPixels(mark, w, h) {
  if (mark.shape === "circle") return { cx: mark.x * w, cy: mark.y * h, r: mark.r * Math.min(w, h) };
  return { x: mark.x0 * w, y: mark.y0 * h, width: (mark.x1 - mark.x0) * w, height: (mark.y1 - mark.y0) * h };
}

export function markFromGesture(start, end, via = "pointer") {
  if (!start) return null;
  if (!end || (Math.abs(end.x - start.x) < DRAG_MIN && Math.abs(end.y - start.y) < DRAG_MIN))
    return { shape: "circle", x: start.x, y: start.y, r: TAP_RADIUS, via };
  return { shape: "box", x0: r4(clamp(Math.min(start.x, end.x))), y0: r4(clamp(Math.min(start.y, end.y))),
           x1: r4(clamp(Math.max(start.x, end.x))), y1: r4(clamp(Math.max(start.y, end.y))), via };
}

export function gridMark(index) {
  if (!(index >= 0 && index < 9)) return null;
  const c = index % 3, r = Math.floor(index / 3);
  return { shape: "box", x0: r4(c / 3), y0: r4(r / 3), x1: r4((c + 1) / 3), y1: r4((r + 1) / 3), via: "grid" };
}

export function addMark(marks, m) {
  if (!m || marks.length >= MAX_MARKS) return marks;
  return [...marks, m];
}
export const undoMark = (marks) => marks.slice(0, -1);
export const removeMark = (marks, i) => marks.filter((_, k) => k !== i);
