// Function to update the time and date
function updateDateTime() {
    const now = new Date();
    const date = now.toLocaleDateString();
    const time = now.toLocaleTimeString();
    document.getElementById("datetime").textContent = `${date} ${time}`;
}

// Update the time and date every second
setInterval(updateDateTime, 1000);
updateDateTime(); // Initial call

// Function to fetch and display sensor data
async function fetchSensorData() {
    try {
        const response = await fetch("/sensor_data");
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Debugging: Log the received data
        console.log("Sensor data received:", data);

        // Update system monitoring data
        document.getElementById("cpu-usage").textContent = `${data.cpu_usage.toFixed(2)}%`;
        document.getElementById("ram-usage").textContent = `${data.ram_usage.toFixed(2)}%`;
        document.getElementById("storage-usage").textContent = `${data.storage_usage.toFixed(2)}%`;

        // Update IP address
        document.getElementById("ip-address").textContent = `IP: ${data.ip_address}`;

        // Update analog sensor data
        document.getElementById("analog-value").textContent = `${data.analog_value !== null ? data.analog_value : "N/A"}`;

        // Update color sensor data
        document.getElementById("color-red").textContent = `${data.color_red !== null ? data.color_red : "N/A"}`;

        // Update motion sensor data
        document.getElementById("accel-x").textContent = `${data.accel_x !== null ? data.accel_x.toFixed(2) : "N/A"}`;

        // Update pressure and temperature data
        document.getElementById("pressure").textContent = `${data.pressure !== null ? data.pressure.toFixed(2) : "N/A"}`;

        // Update SHTC3 temperature data
        document.getElementById("temperature-sht").textContent = `${data.temperature_sht !== null ? data.temperature_sht.toFixed(2) : "N/A"}`;
    } catch (error) {
        console.error("Error fetching sensor data:", error);
    }
}

// Update sensor data every 2 seconds
setInterval(fetchSensorData, 2000);
fetchSensorData(); // Initial call

// Function to fetch an image of the detected species
async function fetchSpeciesImage(speciesName) {
    try {
        // Use a public API like Wikimedia Commons to fetch an image
        const response = await fetch(`https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(speciesName)}&prop=pageimages&format=json&pithumbsize=100&origin=*`);
        const data = await response.json();
        const pages = data.query.pages;
        const pageId = Object.keys(pages)[0];
        const imageUrl = pages[pageId].thumbnail?.source;

        return imageUrl || "https://via.placeholder.com/50"; // Fallback to a placeholder if no image is found
    } catch (error) {
        console.error("Error fetching species image:", error);
        return "https://via.placeholder.com/50"; // Fallback to a placeholder on error
    }
}

// Function to update Detection Insights and Last 5 Detections
async function updateDetectionData() {
    try {
        const response = await fetch("/inference_data");
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Find the detection with the highest confidence score
        const highestConfidenceDetection = data.reduce((prev, current) =>
            prev.confidence > current.confidence ? prev : current
        );

        // Highlight the detection insights box if confidence > 230
        const detectionInsightsBox = document.getElementById("detection-insights-box");
        if (highestConfidenceDetection.confidence > 230) {
            detectionInsightsBox.classList.add("highlight");
        } else {
            detectionInsightsBox.classList.remove("highlight");
        }

        // Fetch an image of the detected species
        const imageUrl = await fetchSpeciesImage(highestConfidenceDetection.label);

        // Create the insight card
        const insightCard = `
            <div class="insight-card">
                <img src="${imageUrl}" alt="${highestConfidenceDetection.label}">
                <div class="content">
                    <h4>Breaking: ${highestConfidenceDetection.label} detected with ${highestConfidenceDetection.confidence.toFixed(2)} confidence!</h4>
                    <p>Did you know? ${highestConfidenceDetection.label} is a fascinating species!</p>
                    <p>Stay tuned for more updates on detected objects in your environment.</p>
                </div>
            </div>
        `;

        // Update the Detection Insights section
        const detectionInsights = document.getElementById("detection-insights");
        detectionInsights.innerHTML = insightCard;

        // Update the Last 5 Detections table
        const detectionsTableBody = document.getElementById("detections-table-body");
        detectionsTableBody.innerHTML = data
            .map(
                (detection) => `
                <tr>
                    <td>${detection.category}</td>
                    <td>${detection.label}</td>
                    <td>${detection.confidence.toFixed(2)}</td>
                </tr>
            `
            )
            .join("");
    } catch (error) {
        console.error("Error updating detection data:", error);
    }
}

// Update detection data every 30 seconds
setInterval(updateDetectionData, 30000);
updateDetectionData(); // Initial call

// Function to fetch and display growth rate data
async function fetchGrowthData() {
    try {
        const response = await fetch("/growth_graph");
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Update the growth graph
        document.getElementById("growth-graph").src = `data:image/png;base64,${data.image}`;

        // Update the growth rate table
        const growthRateTableBody = document.getElementById("growth-rate-table-body");
        growthRateTableBody.innerHTML = `
            <tr>
                <td>Plant A</td>
                <td>10 cm</td>
                <td>30 cm</td>
            </tr>
            <tr>
                <td>Plant B</td>
                <td>5 cm</td>
                <td>25 cm</td>
            </tr>
            <tr>
                <td>Plant C</td>
                <td>8 cm</td>
                <td>24 cm</td>
            </tr>
        `;
    } catch (error) {
        console.error("Error fetching growth data:", error);
    }
}

// Update growth data every 5 seconds
setInterval(fetchGrowthData, 5000);
fetchGrowthData(); // Initial call

// Function to fetch and display seasonal status
async function fetchSeasonalStatus() {
    try {
        const response = await fetch("/seasonal_status");
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Update the seasonal status table
        const seasonalStatusTableBody = document.getElementById("seasonal-status-table-body");
        seasonalStatusTableBody.innerHTML = data
            .map(
                (status) => `
                <tr>
                    <td>${status.plant_name}</td>
                    <td>${status.start_date}</td>
                    <td>${status.harvest_date}</td>
                    <td>${status.current_stage}%</td>
                </tr>
            `
            )
            .join("");
    } catch (error) {
        console.error("Error fetching seasonal status:", error);
    }
}

// Update seasonal status every 10 seconds
setInterval(fetchSeasonalStatus, 10000);
fetchSeasonalStatus(); // Initial call

// Function to fetch and display harvest scheduler data
async function fetchHarvestScheduler() {
    try {
        const response = await fetch("/harvest_scheduler");
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        // Update the harvest scheduler table
        const harvestSchedulerTableBody = document.getElementById("harvest-scheduler-table-body");
        harvestSchedulerTableBody.innerHTML = data
            .map(
                (schedule) => `
                <tr>
                    <td>${schedule.plant_name}</td>
                    <td>${schedule.predicted_harvest_date}</td>
                </tr>
            `
            )
            .join("");
    } catch (error) {
        console.error("Error fetching harvest scheduler data:", error);
    }
}

// Update harvest scheduler data every 10 seconds
setInterval(fetchHarvestScheduler, 10000);
fetchHarvestScheduler(); // Initial call

// Function to take a snapshot of the live feed
function takeSnapshot() {
    const videoFeed = document.querySelector(".video-feed img");
    const canvas = document.createElement("canvas");
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

    // Convert the canvas to an image and download it
    const link = document.createElement("a");
    link.download = "snapshot.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
}