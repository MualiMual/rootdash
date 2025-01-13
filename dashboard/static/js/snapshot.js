// Function to take a snapshot of the live feed
export function takeSnapshot() {
    const videoFeed = document.querySelector(".video-feed img, .video-feed video");
    const canvas = document.createElement("canvas");
    canvas.width = videoFeed.videoWidth || videoFeed.width;
    canvas.height = videoFeed.videoHeight || videoFeed.height;
    const context = canvas.getContext("2d");
    context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

    const link = document.createElement("a");
    link.download = "snapshot.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
}