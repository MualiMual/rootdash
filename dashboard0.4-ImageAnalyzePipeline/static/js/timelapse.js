export function startTimeLapse() {
    fetch('/start_time_lapse', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then(data => {
            alert(data.message || "Time-lapse started successfully!");
        })
        .catch(error => {
            console.error("Error starting time-lapse:", error);
            alert("Failed to start time-lapse: " + error.message);
        });
}

// Attach the function to the window object
window.startTimeLapse = startTimeLapse;