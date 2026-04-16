let lastFilename = ""; // store the last uploaded filename

document.getElementById("predictBtn").addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        showMessage("Please select an image first!", "error");
        return;
    }

    // Show loading state
    const predictBtn = document.getElementById("predictBtn");
    predictBtn.innerText = "Analyzing...";
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            predictBtn.innerText = "Predict";
            predictBtn.disabled = false;

            if (data.error) {
                showMessage(data.error, "error");
                resetUI();
                return;
            }

            lastFilename = data.filename;

            // Update result text
            document.getElementById("result").innerText =
                "Prediction: " + data.prediction;

            document.getElementById("confidence").innerText =
                "Confidence: " + data.confidence + "%";

            // Show image
            const img = document.getElementById("uploadedImg");
            img.src = URL.createObjectURL(file);
            img.style.display = "block";

            // Show result box
            const resultBox = document.getElementById("resultBox");
            resultBox.style.display = "block";

            // Apply color based on prediction
            resultBox.classList.remove("pneumonia", "normal");

            if (data.prediction.toLowerCase().includes("pneumonia")) {
                resultBox.classList.add("pneumonia");
            } else {
                resultBox.classList.add("normal");
            }

            // Show download button
            document.getElementById("downloadBtn").style.display = "inline-block";

            clearMessage();
        })
        .catch(error => {
            console.error("Error:", error);

            predictBtn.innerText = "Predict";
            predictBtn.disabled = false;

            showMessage("Error connecting to API!", "error");
            resetUI();
        });
});

document.getElementById("downloadBtn").addEventListener("click", () => {
    if (!lastFilename) return;

    const downloadUrl = "/download/" + lastFilename;
    window.open(downloadUrl, "_blank");
});

// Helper functions

function showMessage(msg, type) {
    const msgBox = document.getElementById("messageBox");

    msgBox.innerText = msg;

    if (type === "error") {
        msgBox.style.color = "#e74c3c";
    } else {
        msgBox.style.color = "#27ae60";
    }
}

function clearMessage() {
    const msgBox = document.getElementById("messageBox");
    msgBox.innerText = "";
}

function resetUI() {
    document.getElementById("result").innerText = "";
    document.getElementById("confidence").innerText = "";

    document.getElementById("uploadedImg").style.display = "none";
    document.getElementById("downloadBtn").style.display = "none";

    const resultBox = document.getElementById("resultBox");
    resultBox.style.display = "none";
}