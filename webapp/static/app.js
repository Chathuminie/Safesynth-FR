/**
 * SafeSynth-FR Frontend JavaScript
 * Handles upload recognition and page interactions
 * 
 * NOTE: Login and Register forms have their own inline handlers in their HTML files
 * This module handles: upload recognition page, alerts, and upload-specific logic
 */

// ========== UTILITY FUNCTIONS ==========

function showAlert(message, type = "info") {
  // Types: success, error, warning, info
  if (!message) return;
  
  // Create alert box
  const alert = document.createElement("div");
  alert.className = `alert alert-${type}`;
  alert.innerHTML = `
    <div style="display: flex; align-items: flex-start; gap: var(--spacing-md);">
      <span style="font-size: 18px; flex-shrink: 0;">
        ${type === "success" ? "✓" : type === "error" ? "✕" : type === "warning" ? "⚠" : "ℹ"}
      </span>
      <span>${message}</span>
    </div>
  `;
  alert.style.marginBottom = "var(--spacing-md)";
  
  // Add to top of container or body
  const container = document.querySelector(".main-content") || document.body;
  if (container.firstChild) {
    container.insertBefore(alert, container.firstChild);
  } else {
    container.appendChild(alert);
  }
  
  // Auto remove after 5 seconds
  setTimeout(() => alert.remove(), 5000);
}


// ========== UPLOAD RECOGNITION PAGE FUNCTIONALITY ==========

const fileInput = document.getElementById("file");
const previewImg = document.getElementById("preview");
const previewPlaceholder = document.querySelector(".preview-placeholder");
const recognizeBtn = document.getElementById("btn");
const statusMessage = document.getElementById("status");
const resultLabel = document.getElementById("label");
const resultConf = document.getElementById("conf");
const resultNote = document.getElementById("note");
const progressFill = document.getElementById("progressFill");

// Handle file input change
if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  });
}

function handleFileSelect(file) {
  if (!file.type.startsWith("image/")) {
    showAlert("Please select an image file (JPG, PNG, etc.)", "error");
    fileInput.value = "";
    return;
  }
  
  if (file.size > 10 * 1024 * 1024) { // 10MB limit
    showAlert("Image size must be less than 10MB", "error");
    fileInput.value = "";
    return;
  }
  
  const reader = new FileReader();
  reader.onload = (e) => {
    // Show image, hide placeholder
    previewImg.src = e.target.result;
    previewImg.classList.add("show");
    if (previewPlaceholder) previewPlaceholder.classList.add("hidden");
    
    // Reset results
    if (resultLabel) resultLabel.textContent = "-";
    if (resultConf) resultConf.textContent = "-";
    if (progressFill) progressFill.style.width = "0%";
    if (resultNote) {
      resultNote.textContent = "";
      resultNote.classList.remove("show");
    }
    
    showAlert("Image loaded successfully. Click 'Recognize Face' to analyze.", "success");
  };
  reader.readAsDataURL(file);
}

// Handle recognition button click
if (recognizeBtn) {
  recognizeBtn.addEventListener("click", async () => {
    if (!fileInput.files[0]) {
      showAlert("Please select an image first", "error");
      return;
    }
    
    recognizeBtn.disabled = true;
    const originalText = recognizeBtn.textContent;
    recognizeBtn.innerHTML = '<span style="display: inline-block; width: 14px; height: 14px; border: 2px solid white; border-right-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite;"></span> &nbsp; Analyzing...';
    
    if (statusMessage) statusMessage.textContent = "Processing image...";
    
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      
      const result = await response.json();
      
      if (result.ok) {
        // Display results
        if (result.label === "Unknown" || result.label === "unknown") {
          resultLabel.textContent = "Unknown Person";
          resultLabel.classList.remove("success", "error", "warning");
          resultLabel.classList.add("warning");
          if (resultConf) resultConf.textContent = "-";
          if (progressFill) progressFill.style.width = "0%";
          if (resultNote) {
            resultNote.textContent = result.detection_status === "multiple_faces"
              ? "Multiple faces detected. The system analyzed the largest face, but the person is not recognized in the database."
              : "This person is not recognized in the database.";
            resultNote.classList.add("show");
          }
          showAlert(
            result.detection_status === "multiple_faces"
              ? "Multiple faces detected. The largest face was analyzed, but the person was not recognized."
              : "Person not recognized in database",
            "warning"
          );
        } else if (result.label === "Uncertain") {
          const confidence = (result.confidence || 0) * 100;
          resultLabel.textContent = "Uncertain Match";
          resultLabel.classList.remove("success", "error", "warning");
          resultLabel.classList.add("warning");
          if (resultConf) resultConf.textContent = confidence.toFixed(1) + "%";
          if (progressFill) progressFill.style.width = confidence + "%";
          if (resultNote) {
            resultNote.textContent = result.detection_status === "multiple_faces"
              ? `Multiple faces detected. The largest face appears similar to ${result.raw_label}, but the result is uncertain.`
              : `The face appears similar to ${result.raw_label}, but the result is uncertain.`;
            resultNote.classList.add("show");
          }
          showAlert(
            result.detection_status === "multiple_faces"
              ? `Multiple faces detected. The largest face is similar to ${result.raw_label}, but the match is uncertain.`
              : `Possible match to ${result.raw_label}, but confidence is too close to another person.`,
            "warning"
          );
        } else if (!result.label || result.label === "No Face") {
          resultLabel.textContent = "No Face Detected";
          resultLabel.classList.remove("success", "error", "warning");
          resultLabel.classList.add("error");
          if (resultConf) resultConf.textContent = "-";
          if (progressFill) progressFill.style.width = "0%";
          if (resultNote) {
            resultNote.textContent = "No face was detected in the image. Please try with a clearer image showing a face.";
            resultNote.classList.add("show");
          }
          showAlert("No face detected in image. Please try with a clearer image.", "warning");
        } else {
          // Recognized person
          const confidence = (result.confidence || 0) * 100;
          resultLabel.textContent = result.label;
          resultLabel.classList.remove("success", "error", "warning");
          resultLabel.classList.add(result.detection_status === "multiple_faces" ? "warning" : "success");
          
          if (resultConf) resultConf.textContent = confidence.toFixed(1) + "%";
          if (progressFill) progressFill.style.width = confidence + "%";
          
          if (resultNote) {
            resultNote.textContent = result.detection_status === "multiple_faces"
              ? `Multiple faces detected. The largest face was analyzed. Match confidence: ${confidence.toFixed(2)}%`
              : `Match confidence: ${confidence.toFixed(2)}%`;
            resultNote.classList.add("show");
          }
          
          showAlert(
            result.detection_status === "multiple_faces"
              ? `Multiple faces detected. Recognized ${result.label} from the largest face (${confidence.toFixed(1)}% confidence).`
              : `Person recognized: ${result.label} (${confidence.toFixed(1)}% confidence)`,
            result.detection_status === "multiple_faces" ? "warning" : "success"
          );
        }
      } else {
        resultLabel.textContent = "Error";
        resultLabel.classList.remove("success", "warning");
        resultLabel.classList.add("error");
        if (resultNote) {
          resultNote.textContent = result.error || "Recognition failed. Please try another image.";
          resultNote.classList.add("show");
        }
        showAlert(result.error || "Recognition failed. Please try another image.", "error");
      }
    } catch (error) {
      resultLabel.textContent = "Error";
      resultLabel.classList.remove("success", "warning");
      resultLabel.classList.add("error");
      if (resultNote) {
        resultNote.textContent = "Network error: " + error.message;
        resultNote.classList.add("show");
      }
      showAlert("Network error: " + error.message, "error");
    } finally {
      recognizeBtn.disabled = false;
      recognizeBtn.textContent = originalText;
      if (statusMessage) statusMessage.textContent = "";
    }
  });
}
