"use client";

import { useState, useEffect } from "react";

export const OPENCV_URL = "https://docs.opencv.org/4.x/opencv.js";

/**
 * React hook that loads OpenCV.js from the CDN and tracks loading/ready/error status.
 * Adapted from floorplan-detector/src/hooks/useOpenCv.js
 */
export function useOpenCv() {
  const [status, setStatus] = useState("idle"); // "idle" | "loading" | "ready" | "error"
  const [error, setError] = useState("");

  useEffect(() => {
    // Deduplicate: only load once per hook instance
    let loaded = false;

    async function load() {
      if (window.cv) {
        setStatus("ready");
        return;
      }

      // Check if a script tag already exists
      if (document.querySelector(`script[src="${OPENCV_URL}"]`)) {
        setStatus("loading");
        return;
      }

      setStatus("loading");

      return new Promise((resolve, reject) => {
        let settled = false;
        const settle = (fn) => (val) => {
          if (settled) return;
          settled = true;
          fn(val);
        };

        const timeoutId = setTimeout(() => {
          if (!settled) {
            settle(reject)(new Error("OpenCV.js load timed out"));
          }
        }, 30000);

        window.Module = window.Module || {};
        window.Module.onRuntimeInitialized = () => {
          clearTimeout(timeoutId);
          if (!settled) {
            settled = true;
            setStatus("ready");
            resolve();
          }
        };
        window.Module.onAbort = () => {
          clearTimeout(timeoutId);
          if (!settled) {
            settled = true;
            setStatus("error");
            setError("OpenCV failed to initialize");
            reject(new Error("OpenCV failed to initialize"));
          }
        };

        const script = document.createElement("script");
        script.src = OPENCV_URL;
        script.onload = () => {
          // Script loaded but runtime may not be ready yet;
          // onRuntimeInitialized will fire later
        };
        script.onerror = () => {
          clearTimeout(timeoutId);
          if (!settled) {
            settled = true;
            setStatus("error");
            setError("Failed to load OpenCV.js");
            reject(new Error("Failed to load OpenCV.js"));
          }
        };

        document.head.appendChild(script);
      }).catch((err) => {
        if (!loaded) {
          setStatus("error");
          setError(err.message);
        }
      });
    }

    load().then(() => {
      loaded = true;
    });

    return () => {
      loaded = true;
    };
  }, []);

  return { status, error };
}
