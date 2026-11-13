import os
import json
import time
from datetime import datetime


class SessionManager:

    def __init__(self, base_dir):

        self.base_dir = base_dir

        self.history_dir = os.path.join(
            base_dir,
            "session_history"
        )

        os.makedirs(
            self.history_dir,
            exist_ok=True
        )

        self.history_file = os.path.join(
            self.history_dir,
            "sessions.json"
        )

        self.reset()

    # ========================================================
    # RESET CURRENT SESSION
    # ========================================================

    def reset(self):

        self.start_time = time.time()

        self.pose_scores = {}

        self.completed_steps = []

    # ========================================================
    # RECORD COMPLETED POSE
    # ========================================================

    def record_pose(
        self,
        step,
        score
    ):

        self.pose_scores[
            str(step)
        ] = round(
            float(score) * 100,
            2
        )

        if step not in self.completed_steps:

            self.completed_steps.append(
                step
            )

    # ========================================================
    # CALCULATE OVERALL SCORE
    # ========================================================

    def calculate_overall_score(self):

        if not self.pose_scores:

            return 0.0

        scores = list(
            self.pose_scores.values()
        )

        return round(
            sum(scores) / len(scores),
            2
        )

    # ========================================================
    # CALCULATE DURATION
    # ========================================================

    def get_duration(self):

        return round(
            time.time() -
            self.start_time,
            1
        )

    # ========================================================
    # SAVE SESSION
    # ========================================================

    def save_session(self):

        duration = (
            self.get_duration()
        )

        overall_score = (
            self.calculate_overall_score()
        )

        session = {

            "date":
                datetime.now().strftime(
                    "%Y-%m-%d"
                ),

            "time":
                datetime.now().strftime(
                    "%H:%M:%S"
                ),

            "duration_seconds":
                duration,

            "duration":
                self.format_duration(
                    duration
                ),

            "poses_completed":
                len(
                    self.completed_steps
                ),

            "total_poses":
                12,

            "overall_score":
                overall_score,

            "pose_scores":
                self.pose_scores.copy()
        }

        history = []

        if os.path.exists(
            self.history_file
        ):

            try:

                with open(
                    self.history_file,
                    "r"
                ) as f:

                    history = (
                        json.load(f)
                    )

            except Exception:

                history = []

        history.append(
            session
        )

        with open(
            self.history_file,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )

        return session

    # ========================================================
    # LOAD HISTORY
    # ========================================================

    def load_history(self):

        if not os.path.exists(
            self.history_file
        ):

            return []

        try:

            with open(
                self.history_file,
                "r"
            ) as f:

                return json.load(f)

        except Exception:

            return []

    # ========================================================
    # FORMAT TIME
    # ========================================================

    @staticmethod
    def format_duration(
        seconds
    ):

        minutes = int(
            seconds // 60
        )

        remaining_seconds = int(
            seconds % 60
        )

        return (
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    # ========================================================
    # PRINT FINAL REPORT
    # ========================================================

    def print_final_report(
        self,
        session
    ):

        print("\n")
        print("=" * 60)
        print(
            "       SURYA NAMASKARA SESSION COMPLETE"
        )
        print("=" * 60)

        print(
            f"Overall Score : "
            f"{session['overall_score']:.1f}%"
        )

        print(
            f"Poses         : "
            f"{session['poses_completed']}"
            f"/12"
        )

        print(
            f"Duration      : "
            f"{session['duration']}"
        )

        print("\nPose Scores:")

        for step in range(1, 13):

            score = (
                session[
                    "pose_scores"
                ].get(
                    str(step),
                    0
                )
            )

            print(
                f"Pose {step:2d} : "
                f"{score:5.1f}%"
            )

        print("\nSession saved to:")

        print(
            self.history_file
        )

        print("=" * 60)