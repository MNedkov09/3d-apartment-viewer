const fs = require("fs");
const path = require("path");

const modelsDir = path.join(__dirname, "..", "public", "models");
const outputFile = path.join(__dirname, "..", "public", "models-manifest.json");

try {
  if (fs.existsSync(modelsDir)) {
    const files = fs.readdirSync(modelsDir).filter((f) => f.endsWith(".glb"));
    fs.writeFileSync(outputFile, JSON.stringify(files, null, 2));
    console.log(`Generated manifest for ${files.length} models.`);
  } else {
    fs.writeFileSync(outputFile, "[]");
  }
} catch (err) {
  console.warn("Could not generate models manifest:", err.message);
}
