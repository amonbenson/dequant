const Max = require("max-api");
const { spawn } = require("child_process");
const os = require("os");

Max.outlet("status", "loading");

function getBinaryPath() {
  const platform = os.platform();
  const arch = os.arch();

  switch (platform) {
    case "win32": return "./DequantData/bin/dequant-windows-x86_64.exe";
    case "darwin": return arch === "arm64" ? "./DequantData/bin/dequant-macos-arm64" : "./DequantData/bin/dequant-macos-x86_64";
    case "linux": return "./DequantData/bin/dequant-linux-x86_64";
    default: throw new Error(`Unsupported platform: ${platform} ${arch}`);
  }
}

// Create child process object
const dequant = spawn(
  getBinaryPath(),
  [
    "--config.model.transformer.n_layers=1",
    "dequantize-rt",
    "./DequantData/checkpoints/dequant_l1.pt",
  ],
  { stdio: ["pipe", "pipe", "pipe"] },
);

// Pass stdout from the executable to the Max object outlet
let buf = "";
dequant.stdout.on("data", (data) => {
  buf += data.toString();
  let lines = buf.split("\n");
  buf = lines.pop(); // keep incomplete last line in buffer

  for (const line of lines) {
    // Divide the line into its individual parts
    const parts = line.trim().split(" ").map(x => x.trim());
    if (parts.length === 0) {
      continue;
    }

    // Handle command type conversions
    [cmd, ...args] = parts;
    switch (cmd) {
      case "ready":
        Max.outlet("status", "ready");
        break;
      case "note":
        Max.outlet("note", parseInt(args[0]), parseFloat(args[1]), parseFloat(args[2]));
        break;
      case "pos":
        Max.outlet("pos", parseInt(args[0]));
        break;
      default:
        Max.post(`Unknown internal command ${cmd}`, Max.POST_LEVELS.ERROR);
        break;
    }
  }
});

// Print errors from the executable to the Max console
dequant.stderr.on("data", (data) => {
  Max.post(data.toString(), Max.POST_LEVELS.ERROR);
});

// Executable should never exit
dequant.on("close", (code) => {
  Max.post(`Dequant process exited with code ${code}`, Max.POST_LEVELS.ERROR);
});

// Pass inlet commands to the dequant executable
function handleInlet(cmd, ...args) {
  const line = `${cmd} ${args.join(" ")}\n`;
  dequant.stdin.write(line);
}

// Register all required command types
Max.addHandler("note", (note) => handleInlet("note", note));
Max.addHandler("step", () => handleInlet("step"));
Max.addHandler("seek", (pos) => handleInlet("seek", pos));
