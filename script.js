const video = document.getElementById("video");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceExpressionNet.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
    let results = [];
    if (resizedDetections.length > 0) {
      results = resizedDetections.map((d) =>
        faceMatcher.findBestMatch(d.descriptor)
      );
    }
    results.forEach((result, i) => {
      if (result) {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvas);
      }
    });
  }, 100);
});

function loadLabeledImages() {
  const labels = ["Ivan Kristiawan"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      const img = await faceapi.fetchImage(
        `https://res.cloudinary.com/dbtag5lau/image/upload/v1683079265/Ivan%20Kristiawan.jpg`
      );
      for (let i = 1; i <= 2; i++) {
        setInterval(async () => {
          try {
            const img = await faceapi.fetchImage(
              `https://raw.githubusercontent.com/IvanKristiawan/html_face_recognition/main/labeled_images/${label}/${i}.jpg`
            );

            const detections = await faceapi
              .detectSingleFace(img)
              .withFaceDescriptor();
            descriptions.push(detections.descriptor);
          } catch (error) {
            console.log(error);
          }
        }, 100);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
