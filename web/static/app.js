const presetForms = document.querySelectorAll("[data-preset-form]");
const customForm = document.querySelector("#custom-form");
const runButtons = document.querySelectorAll("[data-run-button]");
const customFilesInput = document.querySelector("#custom-files-input");
const customUploadButton = document.querySelector("#custom-upload-button");
const customResetButton = document.querySelector("#custom-reset-button");
const customSelectionNoteEl = document.querySelector("#custom-selection-note");
const processedEl = document.querySelector("#processed");
const fpsEl = document.querySelector("#fps");
const detectionsEl = document.querySelector("#detections");
const elapsedEl = document.querySelector("#elapsed");
const jobLabelEl = document.querySelector("#job-label");
const jobStatusEl = document.querySelector("#job-status");
const progressBarEl = document.querySelector("#progress-bar");
const latestImageEl = document.querySelector("#latest-image");
const latestTextEl = document.querySelector("#latest-text");
const selectedImageShellEl = document.querySelector("#selected-image-shell");
const selectedImageEl = document.querySelector("#selected-image");
const selectedImageEmptyEl = document.querySelector("#selected-image-empty");
const thumbGridEl = document.querySelector("#thumb-grid");
const artifactsEl = document.querySelector("#artifacts");
const modeTabs = document.querySelectorAll("[data-mode-tab]");
const ocrSections = document.querySelectorAll(".ocr-section");
const modePanels = {
  preset1: document.querySelector("#preset1-panel"),
  preset2: document.querySelector("#preset2-panel"),
  custom: document.querySelector("#custom-panel"),
  architecture: document.querySelector("#architecture-panel"),
};
const maxThumbnails = 1000;
const runButtonLabel = "Start Workload";
const runButtonWaitingLabel = "Wait for a while";
const customUploadButtonLabel = "Upload and Process Images";
const uploadingCustomImagesLabel = "UPLOADING IMAGES...";
const preparingWorkloadLabel = "PREPARING WORKLOAD...";
const loadingOcrEngineLabel = "LOADING OCR ENGINE...";
const processingOcrLabel = "PROCESSING OCR...";
const hailoBusyMessage = "Tiny is already helping someone else. Please try again in a moment.";
const customImageLimit = 10;

let eventSource = null;
let activeJobId = null;
let activeJobPreset = null;
let selectedImageName = null;
const thumbnailFrames = new Map();
const customImageObjectUrls = new Map();

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.modeTab;

    modeTabs.forEach((candidate) => {
      const isActive = candidate === tab;
      candidate.classList.toggle("is-active", isActive);
      candidate.setAttribute("aria-selected", String(isActive));
    });

    Object.entries(modePanels).forEach(([panelMode, panel]) => {
      const isActive = panelMode === mode;
      panel.classList.toggle("is-active", isActive);
      panel.hidden = !isActive;
    });

    ocrSections.forEach((section) => {
      section.hidden = mode === "architecture";
    });
  });
});

function numberOrNull(value) {
  const trimmed = String(value || "").trim();
  return trimmed ? Number(trimmed) : null;
}

function formatSeconds(seconds) {
  return `${Number(seconds || 0).toFixed(2)}s`;
}

function setSelectedThumbnail(imageName) {
  [...thumbGridEl.children].forEach((frame) => {
    frame.classList.toggle("is-selected", frame.dataset.imageName === imageName);
  });
}

function revokeCustomImageObjectUrls() {
  customImageObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  customImageObjectUrls.clear();
}

function imageSource(imageName, endpoint = "images") {
  const customUrl = customImageObjectUrls.get(imageName);
  if (activeJobPreset === "custom" && customUrl) {
    return customUrl;
  }
  return `/api/jobs/${activeJobId}/${endpoint}/${encodeURIComponent(imageName)}`;
}

function setRunButtonsProcessing(isProcessing, label = runButtonWaitingLabel, phase = "processing") {
  runButtons.forEach((button) => {
    button.disabled = isProcessing;
    button.textContent = isProcessing ? label : runButtonLabel;
    button.classList.toggle("is-warmup-glow", isProcessing && phase === "warmup");
    button.classList.toggle("is-processing-glow", isProcessing && phase === "processing");
  });
}

function setButtonProcessingState(button, isProcessing, label, idleLabel, phase) {
  if (!button) {
    return;
  }
  button.disabled = isProcessing;
  button.textContent = isProcessing ? label : idleLabel;
  button.classList.toggle("is-warmup-glow", isProcessing && phase === "warmup");
  button.classList.toggle("is-processing-glow", isProcessing && phase === "processing");
}

function syncProcessingButtonsStatus(label, phase = "warmup") {
  [...runButtons, customUploadButton].forEach((button) => {
    if (button.disabled) {
      button.textContent = label;
      button.classList.toggle("is-warmup-glow", phase === "warmup");
      button.classList.toggle("is-processing-glow", phase === "processing");
    }
  });
}

function setBusyState(isBusy, customLabel = processingOcrLabel, runLabel = runButtonWaitingLabel, phase = "processing") {
  setRunButtonsProcessing(isBusy, runLabel, phase);
  setButtonProcessingState(customUploadButton, isBusy, customLabel, customUploadButtonLabel, phase);
}

function selectImage(imageName, textPreview) {
  if (!activeJobId || !imageName) {
    return;
  }

  selectedImageName = imageName;
  selectedImageEl.src = imageSource(imageName, "images");
  selectedImageEl.alt = imageName;
  selectedImageEl.hidden = false;
  selectedImageShellEl.classList.remove("is-empty");
  selectedImageEmptyEl.hidden = true;
  latestImageEl.textContent = "detected texts";
  latestTextEl.textContent = textPreview || "No text detected in this image.";
  setSelectedThumbnail(imageName);
}

function resetSelectedImage(imageName, textPreview) {
  selectedImageName = null;
  selectedImageEl.removeAttribute("src");
  selectedImageEl.alt = "";
  selectedImageEl.hidden = true;
  selectedImageShellEl.classList.add("is-empty");
  selectedImageEmptyEl.hidden = false;
  latestImageEl.textContent = "detected texts";
  latestTextEl.textContent = textPreview;
}

function setWorkloadStatus(message, phase = "warmup") {
  resetSelectedImage("detected texts", message);
  syncProcessingButtonsStatus(message, phase);
}

function isHailoBusyError(message) {
  const text = String(message || "").toLowerCase();
  return (
    text.includes("hailo_out_of_physical_devices") ||
    text.includes("out_of_physical_devices") ||
    text.includes("not enough free devices") ||
    text.includes("failed to create vdevice")
  );
}

async function readErrorDetail(response, fallbackMessage = "Request failed.") {
  const text = await response.text();
  if (!text) {
    return fallbackMessage;
  }

  try {
    const payload = JSON.parse(text);
    return payload.detail || fallbackMessage;
  } catch {
    return text;
  }
}

function showBusyPopup(message) {
  window.alert(message);
}

function resetTelemetryUi(message = "Ready to serve!") {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  activeJobId = null;
  activeJobPreset = null;
  selectedImageName = null;
  thumbGridEl.replaceChildren();
  thumbnailFrames.clear();
  artifactsEl.textContent = "No artifacts yet.";
  processedEl.textContent = "0 / 0";
  fpsEl.textContent = "0.00";
  detectionsEl.textContent = "0";
  elapsedEl.textContent = "0.00s";
  progressBarEl.style.width = "0%";
  jobLabelEl.textContent = "No active job";
  jobStatusEl.textContent = "idle";
  resetSelectedImage("detected texts", message);
}

function resetCustomUi(message = "Ready to serve!") {
  revokeCustomImageObjectUrls();
  resetTelemetryUi(message);
  if (customSelectionNoteEl) {
    customSelectionNoteEl.textContent = "";
  }
  setBusyState(false);
}

function updateCustomSelectionNote(selectedCount) {
  if (!customSelectionNoteEl) {
    return;
  }
  const skippedCount = Math.max(0, selectedCount - customImageLimit);
  customSelectionNoteEl.textContent = skippedCount
    ? `Limit exceeded. ${skippedCount} extra image${skippedCount === 1 ? "" : "s"} will be skipped.`
    : "";
}

function appendThumbnail(event) {
  if (!activeJobId || !event.image_name) {
    return;
  }

  let frame = thumbnailFrames.get(event.image_name);
  const isNewFrame = !frame;

  if (!frame) {
    frame = document.createElement("figure");
    frame.className = "thumb-frame";
    frame.dataset.imageName = event.image_name;
    frame.tabIndex = 0;
    frame.setAttribute("role", "button");
    frame.setAttribute("aria-label", `Select ${event.image_name}`);

    const image = document.createElement("img");
    image.src = imageSource(event.image_name, "thumbnails");
    image.alt = event.image_name;
    image.loading = "lazy";

    const status = document.createElement("span");
    status.className = "thumb-status";

    const caption = document.createElement("figcaption");

    frame.append(image, status, caption);
    thumbnailFrames.set(event.image_name, frame);
  }

  const isCompleted = event.type === "image_completed";
  const statusText = event.status_label || (isCompleted ? "Done" : "Uploaded");
  const captionText =
    event.caption ||
    (isCompleted
      ? `${event.processed_image_count}/${event.image_count} · ${event.detection_count} boxes`
      : `${event.upload_index || 0}/${event.image_count || 0} · ${statusText}`);
  const textPreview = event.text_preview || (isCompleted ? "" : "Uploaded. Waiting for OCR.");

  frame.querySelector(".thumb-status").textContent = statusText;
  frame.querySelector("figcaption").textContent = captionText;
  frame.onclick = () => {
    selectImage(event.image_name, textPreview);
  };
  frame.onkeydown = (keyboardEvent) => {
    if (keyboardEvent.key === "Enter" || keyboardEvent.key === " ") {
      keyboardEvent.preventDefault();
      selectImage(event.image_name, textPreview);
    }
  };

  if (selectedImageName === event.image_name) {
    latestTextEl.textContent = textPreview || "No text detected in this image.";
  }

  if (isNewFrame) {
    thumbGridEl.prepend(frame);
  }

  if (selectedImageName) {
    setSelectedThumbnail(selectedImageName);
  }

  while (thumbGridEl.children.length > maxThumbnails) {
    thumbGridEl.removeChild(thumbGridEl.lastChild);
  }
}

function appendUploadedThumbnails(job) {
  const uploadedImages = job.uploaded_images || [];
  uploadedImages.forEach((image, index) => {
    appendThumbnail({
      type: "custom_uploaded",
      image_name: image.image_name,
      image_count: uploadedImages.length,
      upload_index: index + 1,
      status_label: "Uploaded",
      text_preview: "Uploaded. Waiting for OCR.",
    });
  });
}

function setAllThumbnailStatus(statusText) {
  thumbnailFrames.forEach((frame) => {
    const status = frame.querySelector(".thumb-status");
    if (status && status.textContent !== "Done") {
      status.textContent = statusText;
    }
  });
}

function updateProgress(event) {
  const processed = event.processed_image_count || 0;
  const total = event.image_count || 0;
  const percent = total ? Math.min(100, (processed / total) * 100) : 0;

  processedEl.textContent = `${processed} / ${total}`;
  progressBarEl.style.width = `${percent}%`;

  if (event.total_detection_count !== undefined) {
    detectionsEl.textContent = event.total_detection_count;
  } else if (event.detection_count !== undefined) {
    detectionsEl.textContent = event.detection_count;
  }

  if (event.fps !== undefined) {
    fpsEl.textContent = Number(event.fps).toFixed(2);
  }
  if (event.elapsed_seconds !== undefined) {
    elapsedEl.textContent = formatSeconds(event.elapsed_seconds);
  }
}

function renderArtifacts(jobId, artifacts) {
  const entries = Object.keys(artifacts || {});
  if (!entries.length) {
    artifactsEl.textContent = "No artifacts yet.";
    return;
  }

  artifactsEl.replaceChildren(
    ...entries.map((name) => {
      const link = document.createElement("a");
      link.href = `/api/jobs/${jobId}/artifacts/${encodeURIComponent(name)}`;
      link.textContent = name;
      return link;
    }),
  );
}

async function refreshJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    return;
  }
  const job = await response.json();
  jobStatusEl.textContent = job.status;
  updateProgress({
    image_count: job.image_count,
    processed_image_count: job.processed_image_count,
    detection_count: job.detection_count,
    fps: job.fps,
    elapsed_seconds: job.elapsed_seconds,
  });
  renderArtifacts(jobId, job.artifacts);
}

function connectEvents(jobId) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    appendThumbnail(event);
    jobStatusEl.textContent = event.status || jobStatusEl.textContent;

    if (event.type === "image_completed") {
      updateProgress(event);
    } else if (event.type === "randomizing_workload") {
      updateProgress(event);
      setWorkloadStatus(preparingWorkloadLabel);
    } else if (event.type === "workload_randomized" || event.type === "discovered") {
      updateProgress(event);
      setWorkloadStatus(preparingWorkloadLabel);
    } else if (event.type === "job_started") {
      setWorkloadStatus(preparingWorkloadLabel);
      if (activeJobPreset === "custom") {
        setAllThumbnailStatus("Processing");
      }
    } else if (event.type === "images_loaded" || event.type === "engine_ready") {
      updateProgress(event);
      setWorkloadStatus(loadingOcrEngineLabel);
    } else if (event.type === "started") {
      updateProgress(event);
      setWorkloadStatus(processingOcrLabel, "processing");
    } else if (event.type === "job_completed") {
      setBusyState(false);
      jobStatusEl.textContent = "completed";
      updateProgress(event.summary || {});
      await refreshJob(jobId);
      if (!selectedImageName) {
        latestTextEl.textContent = "";
      }
      eventSource.close();
    } else if (event.type === "job_failed") {
      setBusyState(false);
      jobStatusEl.textContent = "failed";
      setAllThumbnailStatus("Failed");
      latestTextEl.textContent = isHailoBusyError(event.error)
        ? hailoBusyMessage
        : event.error || "Job failed.";
      eventSource.close();
    }
  };

  eventSource.onerror = () => {
    jobStatusEl.textContent = "stream reconnect";
  };
}

presetForms.forEach((presetForm) => {
  presetForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      preset: presetForm.dataset.presetForm,
      limit: numberOrNull(presetForm.querySelector("[name='limit']")?.value),
      work_block_size: Number(presetForm.querySelector("[name='workBlockSize']")?.value || 20),
      hef_batch_size: Number(presetForm.querySelector("[name='hefBatchSize']")?.value || 10),
    };
    revokeCustomImageObjectUrls();
    resetTelemetryUi(preparingWorkloadLabel);
    setBusyState(true, preparingWorkloadLabel, preparingWorkloadLabel, "warmup");

    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      setBusyState(false);
      if (response.status === 429) {
        const detail = await readErrorDetail(response, hailoBusyMessage);
        resetSelectedImage("detected texts", detail);
        showBusyPopup(detail);
        return;
      }
      const detail = await readErrorDetail(response);
      resetSelectedImage("Request failed", detail);
      return;
    }

    const job = await response.json();
    activeJobId = job.job_id;
    activeJobPreset = job.preset;
    jobLabelEl.textContent = `${presetForm.dataset.jobLabel || "Job"} ${job.job_id}`;
    jobStatusEl.textContent = job.status;
    connectEvents(job.job_id);
  });
});

if (customResetButton) {
  customResetButton.addEventListener("click", () => {
    if (customFilesInput) {
      customFilesInput.value = "";
    }
    resetCustomUi("Ready to serve!");
  });
}

if (customFilesInput) {
  customFilesInput.addEventListener("change", () => {
    updateCustomSelectionNote(customFilesInput.files.length);
  });
}

customForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selectedFiles = [...customFilesInput.files].sort((left, right) =>
    left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: "base" }),
  );
  const files = selectedFiles.slice(0, customImageLimit);
  if (!files.length) {
    resetSelectedImage("detected texts", "Choose at least one image to upload.");
    return;
  }
  const skippedCount = Math.max(0, selectedFiles.length - customImageLimit);
  updateCustomSelectionNote(selectedFiles.length);

  setBusyState(true, uploadingCustomImagesLabel, runButtonWaitingLabel, "warmup");
  thumbGridEl.replaceChildren();
  thumbnailFrames.clear();
  artifactsEl.textContent = "No artifacts yet.";
  resetSelectedImage("detected texts", uploadingCustomImagesLabel);
  progressBarEl.style.width = "0%";
  processedEl.textContent = `0 / ${files.length}`;
  fpsEl.textContent = "0.00";
  detectionsEl.textContent = "0";
  elapsedEl.textContent = "0.00s";

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("work_block_size", "20");
  formData.append("hef_batch_size", "10");

  const response = await fetch("/api/custom-jobs", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    setBusyState(false);
    if (response.status === 429) {
      const detail = await readErrorDetail(response, hailoBusyMessage);
      resetSelectedImage("detected texts", detail);
      showBusyPopup(detail);
      return;
    }
    const detail = await readErrorDetail(response);
    resetSelectedImage("Request failed", detail);
    return;
  }

  const job = await response.json();
  activeJobId = job.job_id;
  activeJobPreset = job.preset;
  revokeCustomImageObjectUrls();
  (job.uploaded_images || []).forEach((image, index) => {
    const file = files[index];
    if (file) {
      customImageObjectUrls.set(image.image_name, URL.createObjectURL(file));
    }
  });
  jobLabelEl.textContent = `Custom ${job.job_id}`;
  jobStatusEl.textContent = job.status;
  updateProgress({
    image_count: job.image_count,
    processed_image_count: 0,
    detection_count: 0,
    fps: 0,
    elapsed_seconds: 0,
  });
  appendUploadedThumbnails(job);
  setWorkloadStatus(preparingWorkloadLabel);
  connectEvents(job.job_id);
});
