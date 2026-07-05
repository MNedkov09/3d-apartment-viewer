import { useEffect, useState } from "react";

const OPENCV_URL = "https://docs.opencv.org/4.x/opencv.js";

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

    function handleScriptLoad() {
      // At this point the JS glue code has loaded, but the WASM runtime
      // may still be compiling/instantiating. cv.onRuntimeInitialized is
      // the actual "safe to use cv.* now" signal.
      if (window.cv && window.cv.Mat) {
        setStatus("ready");
      } else if (window.cv) {
        window.cv["onRuntimeInitialized"] = () => setStatus("ready");
      } else {
        setStatus("error");
      }
    }

    return () => {
      // no cleanup needed for the script tag itself; it's fine to keep it
      // cached across mounts
    };
  }, []);

  return status;
}
