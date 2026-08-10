const apiBaseInput = document.querySelector("#apiBase");
const healthButton = document.querySelector("#healthButton");
const healthStatus = document.querySelector("#healthStatus");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const sourcePreview = document.querySelector("#sourcePreview");
const emptyPreview = document.querySelector("#emptyPreview");
const includeOverlay = document.querySelector("#includeOverlay");
const predictButton = document.querySelector("#predictButton");
const runtimeStatus = document.querySelector("#runtimeStatus");
const mainAngle = document.querySelector("#mainAngle");
const upperAngle = document.querySelector("#upperAngle");
const lowerAngle = document.querySelector("#lowerAngle");
const overlayImage = document.querySelector("#overlayImage");
const emptyOverlay = document.querySelector("#emptyOverlay");
const openOverlayLink = document.querySelector("#openOverlayLink");
const deviceValue = document.querySelector("#deviceValue");
const originalSize = document.querySelector("#originalSize");
const inputSize = document.querySelector("#inputSize");
const landmarkCount = document.querySelector("#landmarkCount");
const vertebraCount = document.querySelector("#vertebraCount");
const vertebraRows = document.querySelector("#vertebraRows");
const jsonOutput = document.querySelector("#jsonOutput");
const copyJsonButton = document.querySelector("#copyJsonButton");

const defaultApiBase =
  window.location.origin && window.location.origin !== "null"
    ? window.location.origin
    : "http://127.0.0.1:8000";
const storedApiBase = localStorage.getItem("vertebraApiBase");
apiBaseInput.value = storedApiBase || defaultApiBase;

let selectedFile = null;
let lastResponseText = "";
let overlayObjectUrl = "";

function getApiBase() {
  return apiBaseInput.value.trim().replace(/\/+$/, "");
}

function setStatus(element, text, kind = "muted") {
  element.textContent = text;
  element.classList.remove("muted", "ok", "warn", "error");
  element.classList.add(kind);
}

function formatAngle(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} deg` : "--";
}

function formatSize(size) {
  if (!size || !size.width || !size.height) {
    return "--";
  }
  return `${size.width} x ${size.height}`;
}

function clearOverlayObjectUrl() {
  if (overlayObjectUrl) {
    URL.revokeObjectURL(overlayObjectUrl);
    overlayObjectUrl = "";
  }
}

function base64ToBlob(base64, mimeType = "image/jpeg") {
  const cleanBase64 = base64.includes(",") ? base64.split(",").pop() : base64;
  const byteCharacters = atob(cleanBase64);
  const byteArrays = [];
  const sliceSize = 8192;

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);
    const byteNumbers = new Array(slice.length);

    for (let index = 0; index < slice.length; index += 1) {
      byteNumbers[index] = slice.charCodeAt(index);
    }

    byteArrays.push(new Uint8Array(byteNumbers));
  }

  return new Blob(byteArrays, { type: mimeType });
}

function resetResults() {
  mainAngle.textContent = "--";
  upperAngle.textContent = "--";
  lowerAngle.textContent = "--";
  deviceValue.textContent = "--";
  originalSize.textContent = "--";
  inputSize.textContent = "--";
  landmarkCount.textContent = "--";
  vertebraCount.textContent = "0 items";
  vertebraRows.innerHTML = '<tr><td colspan="4">No records</td></tr>';
  jsonOutput.textContent = "{}";
  lastResponseText = "";
  copyJsonButton.disabled = true;
  clearOverlayObjectUrl();
  overlayImage.hidden = true;
  overlayImage.removeAttribute("src");
  openOverlayLink.hidden = true;
  openOverlayLink.href = "#";
  openOverlayLink.setAttribute("aria-disabled", "true");
  emptyOverlay.hidden = false;
  emptyOverlay.textContent = "No prediction yet";
}

function setSelectedFile(file) {
  if (!file) {
    return;
  }

  selectedFile = file;
  fileName.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
  predictButton.disabled = false;
  setStatus(runtimeStatus, "Ready", "ok");
  resetResults();

  const previewUrl = URL.createObjectURL(file);
  sourcePreview.onload = () => URL.revokeObjectURL(previewUrl);
  sourcePreview.src = previewUrl;
  sourcePreview.hidden = false;
  emptyPreview.hidden = true;
}

function renderVertebrae(vertebrae = []) {
  vertebraCount.textContent = `${vertebrae.length} item${vertebrae.length === 1 ? "" : "s"}`;

  if (!vertebrae.length) {
    vertebraRows.innerHTML = '<tr><td colspan="4">No records</td></tr>';
    return;
  }

  vertebraRows.innerHTML = vertebrae
    .map((item) => {
      const [centerX, centerY] = item.center || [];
      return `
        <tr>
          <td>${item.index ?? "--"}</td>
          <td>${Number.isFinite(item.score) ? item.score.toFixed(4) : "--"}</td>
          <td>${Number.isFinite(centerX) ? centerX.toFixed(2) : "--"}</td>
          <td>${Number.isFinite(centerY) ? centerY.toFixed(2) : "--"}</td>
        </tr>
      `;
    })
    .join("");
}

function renderPrediction(data) {
  const angles = data.cobb_angles || {};
  mainAngle.textContent = formatAngle(angles.main);
  upperAngle.textContent = formatAngle(angles.upper);
  lowerAngle.textContent = formatAngle(angles.lower);
  deviceValue.textContent = data.device || "--";
  originalSize.textContent = formatSize(data.original_size);
  inputSize.textContent = formatSize(data.input_size);
  landmarkCount.textContent = Array.isArray(data.landmarks)
    ? String(data.landmarks.length)
    : "--";

  renderVertebrae(Array.isArray(data.vertebrae) ? data.vertebrae : []);

  if (data.overlay_image_base64 && data.overlay_image_mime_type) {
    clearOverlayObjectUrl();
    const overlayBlob = base64ToBlob(
      data.overlay_image_base64,
      data.overlay_image_mime_type,
    );
    const overlayFile = new File([overlayBlob], "cobb-overlay.jpg", {
      type: data.overlay_image_mime_type,
    });
    const previewUrl = URL.createObjectURL(overlayFile);
    overlayObjectUrl = URL.createObjectURL(overlayFile);

    overlayImage.onload = () => {
      URL.revokeObjectURL(previewUrl);
      overlayImage.hidden = false;
      emptyOverlay.hidden = true;
      setStatus(
        runtimeStatus,
        `Prediction complete - overlay ${overlayImage.naturalWidth} x ${overlayImage.naturalHeight}`,
        "ok",
      );
    };
    overlayImage.onerror = () => {
      overlayImage.hidden = true;
      emptyOverlay.hidden = false;
      emptyOverlay.textContent = "Overlay image failed to load";
      setStatus(runtimeStatus, "Overlay image failed to load", "warn");
    };
    overlayImage.src = previewUrl;
    overlayImage.hidden = false;
    emptyOverlay.hidden = true;
    openOverlayLink.href = overlayObjectUrl;
    openOverlayLink.hidden = false;
    openOverlayLink.setAttribute("aria-disabled", "false");
  } else {
    clearOverlayObjectUrl();
    overlayImage.hidden = true;
    overlayImage.removeAttribute("src");
    openOverlayLink.hidden = true;
    openOverlayLink.href = "#";
    openOverlayLink.setAttribute("aria-disabled", "true");
    emptyOverlay.hidden = false;
    emptyOverlay.textContent = "No overlay returned";
  }

  lastResponseText = JSON.stringify(data, null, 2);
  jsonOutput.textContent = lastResponseText;
  copyJsonButton.disabled = false;
}

async function checkHealth() {
  const apiBase = getApiBase();
  if (!apiBase) {
    setStatus(healthStatus, "Enter an API base URL", "warn");
    return;
  }

  localStorage.setItem("vertebraApiBase", apiBase);
  setStatus(healthStatus, "Checking...", "muted");

  try {
    const response = await fetch(`${apiBase}/health`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    setStatus(
      healthStatus,
      `Online - ${data.device}, ${formatSize(data.input_size)}`,
      "ok",
    );
  } catch (error) {
    setStatus(healthStatus, error.message || "Health check failed", "error");
  }
}

async function runPrediction() {
  const apiBase = getApiBase();
  if (!apiBase) {
    setStatus(runtimeStatus, "Enter an API base URL", "warn");
    return;
  }
  if (!selectedFile) {
    setStatus(runtimeStatus, "Select an image", "warn");
    return;
  }

  localStorage.setItem("vertebraApiBase", apiBase);
  predictButton.disabled = true;
  setStatus(runtimeStatus, "Running inference...", "muted");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(
      `${apiBase}/predict?include_overlay=${includeOverlay.checked}`,
      {
        method: "POST",
        body: formData,
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    renderPrediction(data);
  } catch (error) {
    setStatus(runtimeStatus, error.message || "Prediction failed", "error");
  } finally {
    predictButton.disabled = false;
  }
}

imageInput.addEventListener("change", (event) => {
  setSelectedFile(event.target.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    imageInput.files = event.dataTransfer.files;
    setSelectedFile(file);
  }
});

healthButton.addEventListener("click", checkHealth);
predictButton.addEventListener("click", runPrediction);

apiBaseInput.addEventListener("change", () => {
  localStorage.setItem("vertebraApiBase", getApiBase());
});

copyJsonButton.addEventListener("click", async () => {
  if (!lastResponseText) {
    return;
  }
  await navigator.clipboard.writeText(lastResponseText);
  copyJsonButton.textContent = "Copied";
  window.setTimeout(() => {
    copyJsonButton.textContent = "Copy";
  }, 1200);
});
