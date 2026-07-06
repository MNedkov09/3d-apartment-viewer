import { useEffect, useState } from "react";

export const OPENCV_URL = "https://docs.opencv.org/4.x/opencv.js";

/**
 * OpenCV.js quirk #1: it's not a normal ES module. It's a big script that,
 * once downloaded, still needs to finish compiling its WASM binary before
 * `window.cv` is actually usable. If you call cv.* too early you get cryptic
 * "cv is not a function" style errors.
 *
 * The reliable pattern is: load the script tag, then wait for
 * cv['onRuntimeInitialized'] to fire (OpenCV.js's own readiness hook),
 * THEN flip a "ready" flag that the rest of the app watches.
 */
export function useOpenCv() {
  const [status, setStatus] = useState("loading"); // "loading" | "ready" | "error"

  useEffect(() => {
    // Guard against double-loading in React 18 StrictMode (effects run twice in dev)
    if (window.cv && window.cv.Mat) {
      setStatus("ready");
      return;
    }

    const existingScript = document.getElementById("opencv-script");
    if (existingScript) {
      // Script tag exists — check if it's already loaded
      if (window.cv && window.cv.Mat) {
        setStatus("ready");
        return;
      }
      // Not loaded yet; add listener (first effect's cleanup removed its listener)
      existingScript.addEventListener("load", handleScriptLoad);
      return () => existingScript.removeEventListener("load", handleScriptLoad);
    }

    const script = document.createElement("script");
    script.id = "opencv-script";
    script.src = OPENCV_URL;
    script.async = true;
    script.onload = handleScriptLoad;
    script.onerror = () => setStatus("error");
    document.body.appendChild(script);

    return () => {
      // no cleanup needed for the script tag itself; it's fine to keep it
      // cached across mounts
    };
  }, []);

  // Stable callback defined outside the effect so it doesn't need to be
  // listed in the dependency array — it doesn't close over any outer vars.
  function handleScriptLoad() {
    // script.onload fires when the JS glue code downloads, NOT when WASM
    // finishes compiling.  We must always register onRuntimeInitialized —
    // it is OpenCV.js's actual readiness signal.  If WASM is already ready
    // (e.g. script was cached), the hook fires synchronously and we pick it
    // up here; otherwise we set it up and wait for the async callback.
    if (window.cv && window.cv.Mat) {
      // Already ready (cached script or previously loaded)
      setStatus("ready");
      return;
    }
    // Always register the hook — it fires synchronously if WASM is already
    // initialized, or asynchronously once compilation finishes.
    window.Module = window.Module || {};
    const prev = window.Module.onRuntimeInitialized;
    window.Module.onRuntimeInitialized = () => {
      // Preserve any previously registered handler (e.g. from another load)
      if (prev) prev();
      setStatus("ready");
    };
  }

  return status;
}
