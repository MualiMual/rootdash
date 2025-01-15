import { endpoints } from './routing.js';

// Function to fetch an image of the detected species
export async function fetchSpeciesImage(speciesName) {
    try {
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

// Fetch and display detection insights
export async function updateDetectionData() {
    try {
        const response = await fetch(endpoints.inferenceData);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
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
				
			
                <div class="content">
                    <h6>Breaking: ${highestConfidenceDetection.label} detected with ${highestConfidenceDetection.confidence.toFixed(2)} confidence!</h6>
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