const video = document.getElementById('video');

Promise.all([
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
  faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
  faceapi.nets.faceExpressionNet.loadFromUri('./models'), // Load face expression model
]).then(start);

async function start() {
  // Initialize container for the bounding boxes
  const container = document.createElement('div');
  container.style.position = 'relative';
  document.body.append(container);

  // Start the camera stream
  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => console.error('Error accessing camera:', err));

  video.addEventListener('playing', () => {
    // Create bounding box boundaries with the video dimensions
    const canvas = faceapi.createCanvasFromMedia(video)
    video.after(canvas);
    const displaySize = { width: video.videoWidth, height: video.videoHeight }
    faceapi.matchDimensions(canvas, displaySize, true)

    // Detect faces every second (reduce lag)
    setInterval(async () => {
      // Detect the faces with facial expressions
      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();
    
      // Resize the bounding box
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
    
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    
      // For each face detected in detections variable, draw the bounding box around them.
      resizedDetections.forEach((detection, i) => {
        const box = detection.detection.box;
        const expressions = detection.expressions; // Get facial expressions
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: `Expression: ${JSON.stringify(expressions)}`, // Display expressions in the label
        });
        drawBox.draw(canvas);
      });
    }, 500);
  });
}
