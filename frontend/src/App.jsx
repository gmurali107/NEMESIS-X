
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [systemStatus, setSystemStatus] = useState(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [cameraResult, setCameraResult] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const cameraFeedRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const busyRef = useRef(false);
  const activeRef = useRef(false);

  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");

  const imageRef = useRef(null);
  const imageCanvasRef = useRef(null);
  const imageObjectUrlRef = useRef(null);

  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState("");
  const [analysisMode, setAnalysisMode] = useState(
    "Object Detection and Tracking"
  );

  // Temporal-spatial and anomaly dashboard data
  const [events, setEvents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [temporalStatus, setTemporalStatus] = useState(null);
  const [historyError, setHistoryError] = useState("");

  // =====================================
  // BACKEND STATUS
  // =====================================

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/api/health`);
        setBackendStatus(response.ok ? "online" : "offline");
      } catch {
        setBackendStatus("offline");
      }
    };

    const checkSystem = async () => {
      try {
        const response = await fetch(`${API}/api/system-status`);

        if (response.ok) {
          setSystemStatus(await response.json());
        }
      } catch {
        setSystemStatus(null);
      }
    };

    checkBackend();
    checkSystem();

    const interval = setInterval(() => {
      checkBackend();
      checkSystem();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // =====================================
  // EVENT HISTORY, ALERTS AND STATISTICS
  // =====================================

  useEffect(() => {
    let cancelled = false;

    const loadIntelligenceData = async () => {
      try {
        const [eventsResponse, alertsResponse, statusResponse] =
          await Promise.all([
            fetch(`${API}/api/events?limit=100`),
            fetch(`${API}/api/alerts?limit=100`),
            fetch(`${API}/api/temporal-status`),
          ]);

        if (!eventsResponse.ok) {
          throw new Error("Could not load event history.");
        }

        if (!alertsResponse.ok) {
          throw new Error("Could not load anomaly alerts.");
        }

        if (!statusResponse.ok) {
          throw new Error("Could not load temporal statistics.");
        }

        const [eventsData, alertsData, statusData] = await Promise.all([
          eventsResponse.json(),
          alertsResponse.json(),
          statusResponse.json(),
        ]);

        if (cancelled) return;

        setEvents(eventsData.events || []);
        setAlerts(alertsData.alerts || []);
        setTemporalStatus(statusData);
        setHistoryError("");
      } catch (error) {
        if (!cancelled) {
          setHistoryError(error.message);
        }
      }
    };

    loadIntelligenceData();

    const interval = setInterval(loadIntelligenceData, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // =====================================
  // IMAGE PREVIEW
  // =====================================

  const handleImageChange = (event) => {
    const file = event.target.files?.[0] || null;

    if (imageObjectUrlRef.current) {
      URL.revokeObjectURL(imageObjectUrlRef.current);
      imageObjectUrlRef.current = null;
    }

    setSelectedImage(file);
    setImageResult(null);
    setImageError("");

    if (file) {
      const previewUrl = URL.createObjectURL(file);
      imageObjectUrlRef.current = previewUrl;
      setImagePreview(previewUrl);
    } else {
      setImagePreview("");
    }
  };

  // =====================================
  // DRAW IMAGE DETECTION BOXES
  // =====================================

  const drawImageDetections = (result = imageResult) => {
    const image = imageRef.current;
    const canvas = imageCanvasRef.current;

    if (!image || !canvas || !image.complete) return;

    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    if (!naturalWidth || !naturalHeight) return;

    const displayWidth = image.clientWidth;
    const displayHeight = image.clientHeight;

    if (!displayWidth || !displayHeight) return;

    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    (result?.detections || []).forEach((detection) => {
      const coordinates = detection.coordinates;

      if (!coordinates || coordinates.length !== 4) return;

      const [x1, y1, x2, y2] = coordinates;

      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;

      const confidence = (
        (detection.confidence || 0) * 100
      ).toFixed(1);

      const label = `${detection.object} ${confidence}%`;

      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.font = "bold 14px Arial";

      const textWidth = ctx.measureText(label).width;
      const labelHeight = 25;
      const labelY = Math.max(0, y - labelHeight);

      ctx.fillStyle = "rgba(0, 20, 35, 0.9)";
      ctx.fillRect(x, labelY, textWidth + 12, labelHeight);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + 6, labelY + 17);
    });
  };

  useEffect(() => {
    if (imageResult && imagePreview) {
      const image = imageRef.current;

      if (image?.complete) {
        drawImageDetections(imageResult);
      }
    }
  }, [imageResult, imagePreview]);

  useEffect(() => {
    const image = imageRef.current;

    if (!image) return;

    const observer = new ResizeObserver(() => {
      drawImageDetections();
    });

    observer.observe(image);

    return () => observer.disconnect();
  }, [imagePreview, imageResult]);

  // =====================================
  // IMAGE DETECTION
  // =====================================

  const handleImageDetection = async () => {
    if (!selectedImage) {
      setImageError("Please select an image first.");
      return;
    }

    setImageLoading(true);
    setImageError("");
    setImageResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch(`${API}/api/detect-image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Image detection failed");
      }

      setImageResult(data);
    } catch (error) {
      setImageError(error.message);
    } finally {
      setImageLoading(false);
    }
  };

  // =====================================
  // VIDEO ANALYSIS
  // =====================================

  const handleVideoChange = (event) => {
    setSelectedVideo(event.target.files?.[0] || null);
    setVideoResult(null);
    setVideoError("");
  };

  const handleVideoAnalysis = async () => {
    if (!selectedVideo) {
      setVideoError("Please select a video first.");
      return;
    }

    setVideoLoading(true);
    setVideoError("");
    setVideoResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedVideo);
      formData.append("analysis_mode", analysisMode);

      const response = await fetch(`${API}/api/analyze-video`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Video analysis failed");
      }

      setVideoResult(data);
    } catch (error) {
      setVideoError(error.message);
    } finally {
      setVideoLoading(false);
    }
  };

  // =====================================
  // DRAW LIVE CAMERA DETECTIONS + ZONE
  // =====================================

  const drawDetections = (data) => {
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;
    const feed = cameraFeedRef.current;

    if (!video || !canvas || !feed) return;

    const width = feed.clientWidth;
    const height = feed.clientHeight;

    if (!width || !height || !video.videoWidth || !video.videoHeight) {
      return;
    }

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const scale = Math.min(
      width / video.videoWidth,
      height / video.videoHeight
    );

    const displayWidth = video.videoWidth * scale;
    const displayHeight = video.videoHeight * scale;

    const offsetX = (width - displayWidth) / 2;
    const offsetY = (height - displayHeight) / 2;

    // Draw monitored zone returned by backend
    const zoneCoordinates = data?.zone?.coordinates;

    if (zoneCoordinates?.length === 4) {
      const [zx1, zy1, zx2, zy2] = zoneCoordinates;

      const zoneX = offsetX + zx1 * scale;
      const zoneY = offsetY + zy1 * scale;
      const zoneWidth = (zx2 - zx1) * scale;
      const zoneHeight = (zy2 - zy1) * scale;

      ctx.strokeStyle = "#ffd54f";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(zoneX, zoneY, zoneWidth, zoneHeight);
      ctx.setLineDash([]);

      ctx.fillStyle = "#ffd54f";
      ctx.font = "bold 13px Arial";
      ctx.fillText("MONITORED ZONE", zoneX + 5, zoneY + 18);
    }

    // Draw detected objects
    (data?.detections || []).forEach((detection) => {
      const coordinates = detection.coordinates;

      if (!coordinates || coordinates.length !== 4) return;

      const [x1, y1, x2, y2] = coordinates;

      const boxX = offsetX + x1 * scale;
      const boxY = offsetY + y1 * scale;
      const boxWidth = (x2 - x1) * scale;
      const boxHeight = (y2 - y1) * scale;

      const label = detection.object || "Object";

      const confidence = (
        (detection.confidence || 0) * 100
      ).toFixed(1);

      const trackId =
        detection.track_id !== undefined &&
        detection.track_id !== null
          ? `ID: ${detection.track_id}`
          : "ID: --";

      const text = `${label} ${confidence}% | ${trackId}`;

      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

      ctx.font = "bold 13px Arial";

      const textWidth = ctx.measureText(text).width;
      const labelHeight = 24;
      const labelY = Math.max(0, boxY - labelHeight);

      ctx.fillStyle = "rgba(0, 20, 35, 0.88)";
      ctx.fillRect(boxX, labelY, textWidth + 12, labelHeight);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, boxX + 6, labelY + 16);
    });
  };

  // =====================================
  // START CAMERA
  // =====================================

  const startCamera = async () => {
    try {
      setCameraError("");
      setCameraResult(null);

      const overlay = overlayCanvasRef.current;

      if (overlay) {
        const ctx = overlay.getContext("2d");
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera access is not supported by this browser."
        );
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      activeRef.current = true;
      setCameraActive(true);

      timerRef.current = setInterval(sendCameraFrame, 700);
    } catch (error) {
      setCameraError(error.message || "Unable to start camera.");
      stopCamera();
    }
  };

  // =====================================
  // SEND CAMERA FRAME TO YOLO
  // =====================================

  const sendCameraFrame = async () => {
    if (
      !activeRef.current ||
      busyRef.current ||
      !videoRef.current ||
      !canvasRef.current
    ) {
      return;
    }

    const video = videoRef.current;

    if (video.readyState < 2 || video.videoWidth === 0) {
      return;
    }

    busyRef.current = true;
    setCameraLoading(true);

    try {
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.75);
      });

      if (!blob || !activeRef.current) return;

      const formData = new FormData();
      formData.append("file", blob, "camera-frame.jpg");

      const response = await fetch(`${API}/api/detect-frame`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Camera detection failed");
      }

      if (activeRef.current) {
        setCameraResult(data);
        drawDetections(data);
        setCameraError("");
      }
    } catch (error) {
      if (activeRef.current) {
        setCameraError(error.message);
      }
    } finally {
      busyRef.current = false;
      setCameraLoading(false);
    }
  };

  // =====================================
  // CAMERA OVERLAY RESIZE
  // =====================================

  useEffect(() => {
    const feed = cameraFeedRef.current;

    if (!feed) return;

    const observer = new ResizeObserver(() => {
      if (cameraResult && activeRef.current) {
        drawDetections(cameraResult);
      }
    });

    observer.observe(feed);

    return () => observer.disconnect();
  }, [cameraResult]);

  // =====================================
  // STOP CAMERA
  // =====================================

  const stopCamera = () => {
    activeRef.current = false;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");

      ctx.clearRect(
        0,
        0,
        overlayCanvasRef.current.width,
        overlayCanvasRef.current.height
      );
    }

    busyRef.current = false;
    setCameraActive(false);
    setCameraLoading(false);
    setCameraResult(null);
  };

  // =====================================
  // CLEANUP
  // =====================================

  useEffect(() => {
    return () => {
      activeRef.current = false;

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (imageObjectUrlRef.current) {
        URL.revokeObjectURL(imageObjectUrlRef.current);
      }
    };
  }, []);

  // =====================================
  // DASHBOARD
  // =====================================

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h2>NEMESIS-X</h2>
        <p className="sidebar-subtitle">Intelligence Platform</p>

        <nav>
          <a href="#dashboard">Dashboard</a>
          <a href="#live-camera">Live Detection</a>
          <a href="#video-analysis">Video Analysis</a>
          <a href="#image-detection">Image Detection</a>
          <a href="#event-history">Event History</a>
          <a href="#anomaly-alerts">Anomaly Alerts</a>
          <a href="#system-status">System Status</a>
        </nav>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>NEMESIS-X Dashboard</h1>
            <p>AI Surveillance and Intelligence System</p>
          </div>

          <span className={`status ${backendStatus}`}>
            Backend: {backendStatus}
          </span>
        </header>

        {/* DASHBOARD STATISTICS */}

        <section id="dashboard" className="stats-grid">
          <div className="stat-card">
            <h3>Image Objects</h3>
            <h2>{imageResult?.total_objects ?? 0}</h2>
          </div>

          <div className="stat-card">
            <h3>Video Tracks</h3>
            <h2>{videoResult?.unique_tracks ?? 0}</h2>
          </div>

          <div className="stat-card">
            <h3>Live Objects</h3>
            <h2>{cameraResult?.total_objects ?? 0}</h2>
          </div>

          <div className="stat-card">
            <h3>Tracked Objects</h3>
            <h2>{temporalStatus?.tracked_objects ?? 0}</h2>
          </div>

          <div className="stat-card">
            <h3>Inside Zone</h3>
            <h2>{temporalStatus?.objects_inside_zone ?? 0}</h2>
          </div>

          <div className="stat-card">
            <h3>Anomaly Alerts</h3>
            <h2>{alerts.length}</h2>
          </div>
        </section>

        {/* LIVE CAMERA */}

        <section id="live-camera" className="panel live-camera-panel">
          <h2>Live Camera Detection & Tracking</h2>

          <div className="camera-screen">
            <div
              ref={cameraFeedRef}
              style={{
                position: "relative",
                width: "100%",
                lineHeight: 0,
                background: "#050b12",
                overflow: "hidden",
              }}
            >
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="camera-video"
                style={{
                  display: "block",
                  width: "100%",
                  height: "auto",
                  maxHeight: "480px",
                  objectFit: "contain",
                }}
              />

              <canvas
                ref={overlayCanvasRef}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                }}
              />
            </div>

            {!cameraActive && (
              <div className="camera-placeholder">
                Camera is currently stopped
              </div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />

          <div className="camera-controls">
            <button onClick={startCamera} disabled={cameraActive}>
              Start Camera
            </button>

            <button onClick={stopCamera} disabled={!cameraActive}>
              Stop Camera
            </button>
          </div>

          <p>
            Camera status:{" "}
            <strong>{cameraActive ? "Running" : "Stopped"}</strong>
          </p>

          {cameraLoading && <p>Analyzing camera frame...</p>}

          {cameraError && (
            <p className="error-message">{cameraError}</p>
          )}

          {cameraResult && (
            <div className="detection-results">
              <h3>Live Tracking Results</h3>

              <p>
                Total objects: <strong>{cameraResult.total_objects}</strong>
              </p>

              <p>
                People detected:{" "}
                <strong>{cameraResult.person_count ?? 0}</strong>
              </p>

              <p>
                People inside zone:{" "}
                <strong>{cameraResult.people_inside_zone ?? 0}</strong>
              </p>

              <h3>Object Counts</h3>

              {Object.entries(cameraResult.class_counts || {}).map(
                ([name, count]) => (
                  <p key={name}>
                    {name}: {count}
                  </p>
                )
              )}

              <h3>Tracked Objects</h3>

              {cameraResult.detections?.length > 0 ? (
                cameraResult.detections.map((detection, index) => (
                  <div
                    key={`${detection.track_id ?? "object"}-${index}`}
                    style={{
                      padding: "10px",
                      margin: "8px 0",
                      borderRadius: "6px",
                      background: "rgba(0, 229, 255, 0.08)",
                      borderLeft: "3px solid #00e5ff",
                    }}
                  >
                    <strong>{detection.object}</strong>
                    <p>Tracking ID: {detection.track_id ?? "Not assigned"}</p>
                    <p>
                      Confidence:{" "}
                      {(detection.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                ))
              ) : (
                <p>No objects detected.</p>
              )}
            </div>
          )}
        </section>

        {/* IMAGE DETECTION */}

        <section id="image-detection" className="panel">
          <h2>Image Detection & Analysis</h2>

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />

          {imagePreview && (
            <div
              style={{
                position: "relative",
                width: "100%",
                maxWidth: "900px",
                marginTop: "20px",
                overflow: "hidden",
                background: "#050b12",
              }}
            >
              <img
                ref={imageRef}
                src={imagePreview}
                alt="Uploaded image for NEMESIS-X analysis"
                onLoad={() => drawImageDetections()}
                style={{
                  display: "block",
                  width: "100%",
                  height: "auto",
                }}
              />

              <canvas
                ref={imageCanvasRef}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                }}
              />
            </div>
          )}

          <button
            onClick={handleImageDetection}
            disabled={imageLoading || !selectedImage}
          >
            {imageLoading ? "Analyzing Image..." : "Analyze Image"}
          </button>

          {imageError && (
            <p className="error-message">{imageError}</p>
          )}

          {imageResult && (
            <div className="detection-results">
              <h3>Image Analysis Results</h3>

              <p>
                Total objects: <strong>{imageResult.total_objects}</strong>
              </p>

              {Object.entries(imageResult.class_counts || {}).map(
                ([name, count]) => (
                  <p key={name}>
                    {name}: {count}
                  </p>
                )
              )}

              <h3>Detected Objects</h3>

              {imageResult.detections?.map((detection, index) => (
                <p key={index}>
                  {detection.object} —{" "}
                  {(detection.confidence * 100).toFixed(1)}% confidence
                </p>
              ))}
            </div>
          )}
        </section>

        {/* VIDEO ANALYSIS */}

        <section id="video-analysis" className="panel video-analysis-panel">
          <h2>Video Analysis</h2>

          <input
            type="file"
            accept="video/*"
            onChange={handleVideoChange}
          />

          <select
            value={analysisMode}
            onChange={(event) => setAnalysisMode(event.target.value)}
          >
            <option>Object Detection and Tracking</option>
            <option>Object Detection Only</option>
          </select>

          <button
            onClick={handleVideoAnalysis}
            disabled={videoLoading || !selectedVideo}
          >
            {videoLoading ? "Analyzing Video..." : "Analyze Video"}
          </button>

          {videoError && (
            <p className="error-message">{videoError}</p>
          )}

          {videoResult && (
            <div className="detection-results">
              <h3>Video Analysis Results</h3>

              <p>Frames processed: {videoResult.frames_processed}</p>
              <p>Unique tracks: {videoResult.unique_tracks}</p>

              <video
                controls
                className="analysed-video"
                src={`${API}${videoResult.annotated_video_url}`}
              />

              <p>
                <a
                  href={`${API}${videoResult.report_csv_url}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download CSV Report
                </a>
              </p>
            </div>
          )}
        </section>

        {/* TEMPORAL TRACKING STATUS */}

        <section id="temporal-status" className="panel">
          <h2>Temporal-Spatial Intelligence</h2>

          {historyError && (
            <p className="error-message">{historyError}</p>
          )}

          {temporalStatus ? (
            <div className="stats-grid">
              <div className="stat-card">
                <h3>Total Tracked IDs</h3>
                <h2>{temporalStatus.tracked_objects}</h2>
              </div>

              <div className="stat-card">
                <h3>Active Objects</h3>
                <h2>{temporalStatus.active_objects}</h2>
              </div>

              <div className="stat-card">
                <h3>Inside Monitored Zone</h3>
                <h2>{temporalStatus.objects_inside_zone}</h2>
              </div>
            </div>
          ) : (
            <p>Waiting for temporal analysis data...</p>
          )}

          <p>
            Long stay threshold:{" "}
            {temporalStatus?.long_stay_threshold_seconds ?? 15} seconds
          </p>

          <p>
            Rapid entry/exit threshold:{" "}
            {temporalStatus?.rapid_event_threshold_seconds ?? 5} seconds
          </p>
        </section>

        {/* ANOMALY ALERTS */}

        <section id="anomaly-alerts" className="panel">
          <h2>Anomaly Alerts</h2>

          <p>
            Total alerts: <strong>{alerts.length}</strong>
          </p>

          {alerts.length === 0 ? (
            <p>No anomalies detected in the available event history.</p>
          ) : (
            alerts.map((alert, index) => (
              <div
                key={`${alert.Track_ID}-${alert.Timestamp}-${index}`}
                style={{
                  padding: "12px",
                  margin: "10px 0",
                  borderRadius: "6px",
                  background: "rgba(255, 70, 70, 0.10)",
                  borderLeft: "4px solid #ff5252",
                }}
              >
                <strong>{alert.Anomaly}</strong>
                <p>Tracking ID: {alert.Track_ID}</p>
                <p>Time: {alert.Timestamp}</p>
                <p>{alert.Explanation}</p>
              </div>
            ))
          )}
        </section>

        {/* EVENT HISTORY */}

        <section id="event-history" className="panel">
          <h2>Event History</h2>

          <p>
            Showing the latest {events.length} events.
          </p>

          {events.length === 0 ? (
            <p>No events recorded yet. Start the live camera to collect events.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  textAlign: "left",
                }}
              >
                <thead>
                  <tr>
                    <th style={{ padding: "10px" }}>Timestamp</th>
                    <th style={{ padding: "10px" }}>Track ID</th>
                    <th style={{ padding: "10px" }}>Object</th>
                    <th style={{ padding: "10px" }}>Event</th>
                    <th style={{ padding: "10px" }}>Details</th>
                  </tr>
                </thead>

                <tbody>
                  {events.map((event, index) => (
                    <tr key={`${event.Track_ID}-${event.Timestamp}-${index}`}>
                      <td style={{ padding: "10px" }}>{event.Timestamp}</td>
                      <td style={{ padding: "10px" }}>{event.Track_ID}</td>
                      <td style={{ padding: "10px" }}>{event.Object}</td>
                      <td style={{ padding: "10px" }}>{event.Event}</td>
                      <td style={{ padding: "10px" }}>{event.Details}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* SYSTEM STATUS */}

        <section id="system-status" className="panel">
          <h2>System Status</h2>

          <p>Backend: {backendStatus}</p>
          <p>YOLO11: {systemStatus?.yolo11 ?? "checking"}</p>
          <p>Model file: {systemStatus?.model_file ?? "checking"}</p>
          <p>
            Temporal Analysis:{" "}
            {systemStatus?.temporal_analysis ?? "checking"}
          </p>
          <p>
            Anomaly Detection:{" "}
            {systemStatus?.anomaly_detection ?? "checking"}
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
