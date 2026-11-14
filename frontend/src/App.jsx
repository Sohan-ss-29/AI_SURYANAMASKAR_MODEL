
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Play,
  History,
  Info,
  ChevronRight,
  Camera,
  CheckCircle2,
  RotateCcw,
  AlertCircle,
} from "lucide-react";

import "./App.css";


function App() {

  const [page, setPage] = useState("home");

  return (
    <div className="app">

      {/* =====================================================
          NAVBAR
      ===================================================== */}

      <nav className="navbar">

        <div
          className="logo"
          onClick={() => setPage("home")}
        >
          <Activity size={26} />
          <span>SuryaNamaskar AI</span>
        </div>


        <div className="nav-links">

          <button
            onClick={() => setPage("home")}
          >
            Home
          </button>


          <button
            onClick={() => setPage("history")}
          >
            <History size={18} />
            History
          </button>


          <button
            onClick={() => setPage("about")}
          >
            <Info size={18} />
            About
          </button>

        </div>

      </nav>


      {/* =====================================================
          HOME
      ===================================================== */}

      {page === "home" && (

        <section className="hero">

          <div className="hero-content">

            <div className="badge">

              <Activity size={16} />

              AI-Powered Yoga Assistant

            </div>


            <h1>

              Practice Surya Namaskara

              <span>
                with AI Guidance
              </span>

            </h1>


            <p>
              Real-time posture detection, intelligent correction,
              step-by-step guidance, and personalized performance
              scoring using computer vision.
            </p>


            <div className="buttons">

              <button
                className="primary-button"
                onClick={() => setPage("practice")}
              >

                <Play size={20} />

                Start Practice

              </button>


              <button
                className="secondary-button"
                onClick={() => setPage("about")}
              >

                Learn More

                <ChevronRight size={18} />

              </button>

            </div>


            <div className="features">

              <div>

                <strong>
                  12
                </strong>

                <span>
                  Yoga Steps
                </span>

              </div>


              <div>

                <strong>
                  AI
                </strong>

                <span>
                  Pose Detection
                </span>

              </div>


              <div>

                <strong>
                  Real-Time
                </strong>

                <span>
                  Feedback
                </span>

              </div>

            </div>

          </div>


          <div className="hero-visual">

            <div className="visual-card">

              <Activity size={80} />

              <h2>
                Surya Namaskara
              </h2>

              <p>
                Your AI-powered personal yoga assistant
              </p>


              <div className="pose-indicator">

                <span></span>

                Ready to practice

              </div>

            </div>

          </div>

        </section>

      )}


      {/* =====================================================
          PRACTICE
      ===================================================== */}

      {page === "practice" && (

        <PracticePage
          onHome={() => setPage("home")}
        />

      )}


      {/* =====================================================
          HISTORY
      ===================================================== */}

      {page === "history" && (

        <section className="practice-page">

          <h1>
            Session History
          </h1>

          <p>
            Your completed Surya Namaskara sessions will appear here.
          </p>

        </section>

      )}


      {/* =====================================================
          ABOUT
      ===================================================== */}

      {page === "about" && (

        <section className="practice-page">

          <h1>
            About SuryaNamaskar AI
          </h1>


          <p>
            SuryaNamaskar AI uses computer vision, MediaPipe pose
            estimation, and a MobileNetV2-based classifier to analyse
            Surya Namaskara postures in real time.
          </p>


          <button
            className="primary-button"
            onClick={() => setPage("home")}
          >
            Back to Home
          </button>

        </section>

      )}

    </div>
  );
}



/* ============================================================
   PRACTICE PAGE
============================================================ */

function PracticePage({ onHome }) {

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);
  const socketRef = useRef(null);

  const captureIntervalRef = useRef(null);

  const [started, setStarted] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const [aiResult, setAiResult] = useState({

    pose_detected: false,

    pose_id: null,

    pose_name: "",

    classification_confidence: 0,

    posture_valid: false,

    posture_score: 0,

    message:
      "Position yourself in front of the camera."

  });

  const [sequence, setSequence] = useState({
    expected_step: 1,
    expected_pose: 1,
    sequence_state: "waiting",
    sequence_message: "Perform Pose 1.",
    hold_elapsed: 0,
    completed_step: null,
    session_complete: false
  });



  /* ============================================================
     BACKEND URL
  ============================================================ */

  const BACKEND_HOST =
    window.location.hostname || "localhost";

  const WEBSOCKET_URL =
    `ws://${BACKEND_HOST}:8000/ws/practice`;



  /* ============================================================
     POSE NAME
  ============================================================ */

  const getPoseDisplayName = (poseName) => {

    if (!poseName) {

      return "Waiting for pose";

    }

    return poseName
      .replace(/^\d+_/, "")
      .replaceAll("_", " ");

  };



  /* ============================================================
     START CAMERA
  ============================================================ */

  const startCamera = async () => {

    try {

      setCameraError("");

      const stream =
        await navigator.mediaDevices.getUserMedia({

          video: {
            width: 640,
            height: 480,
            facingMode: "user"
          },

          audio: false

        });


      streamRef.current = stream;

      /*
       * First mount the video element.
       * WebSocket connection will happen after
       * React has rendered the video.
       */

      setStarted(true);

    }

    catch (error) {

      console.error(
        "Camera error:",
        error
      );

      setCameraError(
        "Camera access was denied or is unavailable."
      );

    }

  };



  /* ============================================================
     ATTACH CAMERA
  ============================================================ */

  useEffect(() => {

    if (
      !started ||
      !videoRef.current ||
      !streamRef.current
    ) {

      return;

    }


    videoRef.current.srcObject =
      streamRef.current;


    videoRef.current
      .play()
      .then(() => {

        console.log(
          "Camera started successfully."
        );

      })
      .catch((error) => {

        console.error(
          "Video playback error:",
          error
        );

      });

  }, [started]);



  /* ============================================================
     CONNECT TO AI BACKEND
  ============================================================ */

  useEffect(() => {

    if (!started) {

      return;

    }


    /*
     * Small delay ensures that:
     *
     * 1. React mounted the video
     * 2. video element exists
     * 3. camera stream is attached
     */

    const timer =
      setTimeout(() => {

        connectWebSocket();

      }, 300);


    return () => {

      clearTimeout(timer);

    };

  }, [started]);



  /* ============================================================
     WEBSOCKET
  ============================================================ */

  const connectWebSocket = () => {

    /*
     * Prevent duplicate connections.
     */

    if (
      socketRef.current &&
      (
        socketRef.current.readyState ===
        WebSocket.OPEN ||

        socketRef.current.readyState ===
        WebSocket.CONNECTING
      )
    ) {

      console.log(
        "WebSocket already connected/connecting."
      );

      return;

    }


    console.log(
      "Connecting to:",
      WEBSOCKET_URL
    );


    const socket =
      new WebSocket(
        WEBSOCKET_URL
      );


    socketRef.current =
      socket;


    socket.onopen = () => {

      console.log(
        "================================="
      );

      console.log(
        "Connected to Surya AI backend."
      );

      console.log(
        "WebSocket ready."
      );

      console.log(
        "================================="
      );


      setCameraError("");

      startFrameCapture();

    };



    socket.onmessage = (event) => {

      try {

        const result =
          JSON.parse(
            event.data
          );


        console.log(
          "AI RESULT:",
          result
        );


        /*
         * Backend sends an initial
         * connection confirmation.
         */

        if (
          result.message ===
          "WebSocket connected successfully"
        ) {

          console.log(
            "Backend connection confirmed."
          );

          return;

        }


        setAiResult(result);

        setSequence({
          expected_step:
            result.expected_step ?? 1,

          expected_pose:
            result.expected_pose ?? 1,

          sequence_state:
            result.sequence_state ?? "waiting",

          sequence_message:
            result.sequence_message ??
            "Perform Pose 1.",

          hold_elapsed:
            result.hold_elapsed ?? 0,

          completed_step:
            result.completed_step,

          session_complete:
            result.session_complete ?? false
        });

      }

      catch (error) {

        console.error(
          "Invalid AI response:",
          error
        );

      }

    };



    socket.onerror = (error) => {

      console.error(
        "WebSocket error:",
        error
      );


      setCameraError(
        "Could not connect to the AI backend."
      );

    };



    socket.onclose = (event) => {

      console.log(
        "AI WebSocket closed."
      );

      console.log(
        "Close code:",
        event.code
      );

      console.log(
        "Close reason:",
        event.reason
      );


      /*
       * Only show backend error if
       * the user did not intentionally
       * stop the session.
       */

      if (
        started &&
        event.code !== 1000
      ) {

        setCameraError(
          "AI backend connection closed."
        );

      }

    };

  };



  /* ============================================================
     SEND CAMERA FRAMES
  ============================================================ */

  const startFrameCapture = () => {

    /*
     * Prevent duplicate intervals.
     */

    if (
      captureIntervalRef.current
    ) {

      return;

    }


    console.log(
      "Starting frame capture..."
    );


    captureIntervalRef.current =
      setInterval(() => {

        const video =
          videoRef.current;

        const canvas =
          canvasRef.current;

        const socket =
          socketRef.current;


        if (
          !video ||
          !canvas ||
          !socket
        ) {

          return;

        }


        if (
          socket.readyState !==
          WebSocket.OPEN
        ) {

          return;

        }


        if (
          video.readyState <
          HTMLMediaElement.HAVE_CURRENT_DATA
        ) {

          return;

        }


        if (
          video.videoWidth === 0 ||
          video.videoHeight === 0
        ) {

          return;

        }


        const context =
          canvas.getContext(
            "2d"
          );


        canvas.width =
          video.videoWidth;


        canvas.height =
          video.videoHeight;


        context.drawImage(
          video,
          0,
          0,
          canvas.width,
          canvas.height
        );


        const imageData =
          canvas.toDataURL(
            "image/jpeg",
            0.6
          );


        const base64Image =
          imageData.split(",")[1];


        /*
         * Send raw base64 JPEG.
         */

        socket.send(
          base64Image
        );

      }, 300);

  };



  /* ============================================================
     STOP CAMERA
  ============================================================ */

  const stopCamera = () => {

    console.log(
      "Stopping practice session..."
    );


    /* STOP FRAME CAPTURE */

    if (
      captureIntervalRef.current
    ) {

      clearInterval(
        captureIntervalRef.current
      );

      captureIntervalRef.current =
        null;

    }



    /* CLOSE WEBSOCKET */

    if (
      socketRef.current
    ) {

      try {

        socketRef.current.close(
          1000,
          "Session ended"
        );

      }

      catch (error) {

        console.error(
          error
        );

      }


      socketRef.current =
        null;

    }



    /* STOP CAMERA */

    if (
      streamRef.current
    ) {

      streamRef.current
        .getTracks()
        .forEach(
          (track) => {

            track.stop();

          }
        );


      streamRef.current =
        null;

    }



    /* REMOVE VIDEO STREAM */

    if (
      videoRef.current
    ) {

      videoRef.current.srcObject =
        null;

    }



    setStarted(false);


    setAiResult({

      pose_detected:
        false,

      pose_id:
        null,

      pose_name:
        "",

      classification_confidence:
        0,

      posture_valid:
        false,

      posture_score:
        0,

      message:
        "Position yourself in front of the camera."

    });


    setCameraError("");

  };



  /* ============================================================
     CLEANUP
  ============================================================ */

  useEffect(() => {

    return () => {

      /*
       * Do not call the React state-setting
       * stopCamera() here.
       *
       * Just clean the actual resources.
       */

      if (
        captureIntervalRef.current
      ) {

        clearInterval(
          captureIntervalRef.current
        );

      }


      if (
        socketRef.current
      ) {

        try {

          socketRef.current.close(
            1000
          );

        }

        catch (error) {}

      }


      if (
        streamRef.current
      ) {

        streamRef.current
          .getTracks()
          .forEach(
            track => track.stop()
          );

      }

    };

  }, []);



  /* ============================================================
     EXIT SESSION
  ============================================================ */

  const handleExit = () => {

    stopCamera();

    onHome();

  };



  /* ============================================================
     DISPLAY VALUES
  ============================================================ */

  const displayPoseId =
    aiResult.posture_valid
      ? aiResult.pose_id
      : null;

  const poseName =
    displayPoseId
      ? getPoseDisplayName(
          aiResult.pose_name
        )
      : "Waiting for pose";


  const classificationPercent =
    Math.round(
      (
        aiResult.classification_confidence ||
        0
      ) * 100
    );


  const postureScorePercent =
    Math.round(
      (
        aiResult.posture_score ||
        0
      ) * 100
    );



  /* ============================================================
     UI
  ============================================================ */

  return (

    <section className="practice-container">


      {/* ==================================================
          HEADER
      ================================================== */}

      <div className="practice-header">

        <div>

          <div className="practice-label">

            <span className="live-dot"></span>

            AI PRACTICE SESSION

          </div>


          <h1>
            Surya Namaskara
          </h1>


          <p>
            Follow the sequence and hold each posture correctly.
          </p>

        </div>


        <button
          className="exit-button"
          onClick={handleExit}
        >
          Exit Session
        </button>

      </div>



      {/* ==================================================
          ERROR
      ================================================== */}

      {cameraError && (

        <div className="camera-error">

          <AlertCircle size={20} />

          <span>
            {cameraError}
          </span>

        </div>

      )}



      {/* ==================================================
          MAIN GRID
      ================================================== */}

      <div className="practice-grid">

  <div className="practice-main">


        {/* ==================================================
            CAMERA
        ================================================== */}

        <div className="camera-card">

          {!started ? (

            <div className="camera-placeholder">

              <div className="camera-icon">

                <Camera size={48} />

              </div>


              <h2>
                Camera Ready
              </h2>


              <p>
                Your webcam will be used for real-time
                posture analysis.
              </p>


              <button
                className="primary-button"
                onClick={startCamera}
              >

                <Camera size={19} />

                Start Camera

              </button>

            </div>

          ) : (

            <div className="camera-active">


              {/* WEBCAM */}

              <video
                ref={videoRef}
                className="webcam"
                autoPlay
                muted
                playsInline
              />



              {/* HIDDEN CANVAS */}

              <canvas
                ref={canvasRef}
                style={{
                  display: "none"
                }}
              />



              {/* CAMERA OVERLAY */}

              <div className="camera-overlay">

                <div className="camera-status">

                  <span className="live-dot"></span>

                  LIVE

                </div>


                <div className="camera-step">

                  {displayPoseId
                    ? `Pose ${displayPoseId}`
                    : "Waiting..."}

                </div>

              </div>

            </div>

          )}

        </div>



    {/* SESSION PROGRESS */}

    <div className="info-card session-progress-card">

      <div className="hold-header">

        <span className="small-label">
          SESSION PROGRESS
        </span>

        <span>
          {sequence.expected_step} / 12
        </span>

      </div>

      <div className="step-progress">

        {Array.from({
          length: 12
        }).map((_, index) => (

          <div
            key={index}
            className={
              index ===
              (sequence.expected_step - 1)
                ? "step-dot active"
                : index <
                  (sequence.expected_step - 1)
                  ? "step-dot completed"
                  : "step-dot"
            }
          >
            {index + 1}
          </div>

        ))}

      </div>

      <div className="hold-section">

        <div className="hold-header">

          <span className="small-label">
            HOLD PROGRESS
          </span>

          <span>
            {Math.min(
              sequence.hold_elapsed,
              2
            ).toFixed(1)} / 2.0 sec
          </span>

        </div>

        <div className="progress-track">

          <div
            className="progress-fill"
            style={{
              width:
                `${Math.min(
                  (sequence.hold_elapsed / 2) * 100,
                  100
                )}%`
            }}
          />

        </div>

      </div>

      <div className="sequence-message">
        {sequence.sequence_message}
      </div>

    </div>

  </div>


        {/* ==================================================
            SIDE PANEL
        ================================================== */}

        <div className="practice-side">


          {/* CURRENT POSE */}

          <div className="info-card">

            <div className="card-header">

              <div>

                <span className="small-label">
                  CURRENT DETECTED POSE
                </span>


                <h2>

                  {displayPoseId
                    ? `Pose ${displayPoseId}`
                    : "Waiting for pose"}

                </h2>

              </div>


              <div className="step-number">

                {displayPoseId
                  ? String(
                      displayPoseId
                    ).padStart(
                      2,
                      "0"
                    )
                  : "--"}

              </div>

            </div>


            <p className="pose-name">

              {poseName}

            </p>


            <div className="instruction">

              {aiResult.posture_valid ? (

                <CheckCircle2
                  size={18}
                />

              ) : (

                <AlertCircle
                  size={18}
                />

              )}


              <span>

                {aiResult.posture_valid
                  ? "Correct posture"
                  : aiResult.message}

              </span>

            </div>

          </div>



          {/* AI FEEDBACK */}

          <div className="feedback-card">

            <span className="small-label">
              AI FEEDBACK
            </span>


            <div className="feedback-content">

              <div className="feedback-icon">

                {aiResult.posture_valid ? (

                  <CheckCircle2
                    size={22}
                  />

                ) : (

                  <Activity
                    size={22}
                  />

                )}

              </div>


              <div>

                <h3>

                  {aiResult.posture_valid
                    ? "Good posture"
                    : aiResult.pose_detected
                      ? "Waiting for correct pose"
                      : "Ready to begin"}

                </h3>


                <p>

                  {aiResult.message ||
                    "Position yourself in front of the camera."}

                </p>

              </div>

            </div>

          </div>



          {/* AI CONFIDENCE */}

          <div className="info-card">

            <div className="hold-header">

              <span className="small-label">
                AI CONFIDENCE
              </span>


              <span>
                {classificationPercent}%
              </span>

            </div>


            <div className="progress-track">

              <div
                className="progress-fill"
                style={{
                  width:
                    `${classificationPercent}%`
                }}
              />

            </div>

          </div>



          {/* POSTURE SCORE */}

          <div className="info-card">

            <div className="hold-header">

              <span className="small-label">
                POSTURE SCORE
              </span>


              <span>
                {postureScorePercent}%
              </span>

            </div>


            <div className="progress-track">

              <div
                className="progress-fill"
                style={{
                  width:
                    `${postureScorePercent}%`
                }}
              />

            </div>

          </div>



    {/* CURRENT POSE SCORE */}

    <div className="score-card">

      <div>

        <span className="small-label">
          CURRENT POSE SCORE
        </span>

        <div className="score">
          {postureScorePercent}%
        </div>

      </div>

      <div className="score-ring">
        {postureScorePercent}
      </div>

    </div>


          {/* RESTART */}

          {started && (

            <button
              className="reset-button"
              onClick={stopCamera}
            >

              <RotateCcw size={17} />

              Restart Camera

            </button>

          )}

        </div>

  </div>

    </section>

  );

}


export default App;
