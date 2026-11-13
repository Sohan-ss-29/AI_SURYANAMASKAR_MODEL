import time


class SequenceController:

    def __init__(
        self,
        total_poses=12,
        hold_time=2.0,
        wrong_pose_frames=4,
        hold_grace_time=0.8
    ):

        # ====================================================
        # 12-STEP SURYA NAMASKARA SEQUENCE
        # ====================================================

        self.sequence = [
            1,  # Step 1  - Pranamasana
            2,  # Step 2  - Hasta Uttanasana
            3,  # Step 3  - Padahastasana
            4,  # Step 4  - Ashwa Sanchalanasana
            5,  # Step 5  - Parvatasana
            6,  # Step 6  - Ashtanga Namaskara
            7,  # Step 7  - Bhujangasana
            5,  # Step 8  - Parvatasana
            4,  # Step 9  - Ashwa Sanchalanasana
            3,  # Step 10 - Padahastasana
            2,  # Step 11 - Hasta Uttanasana
            1   # Step 12 - Pranamasana
        ]

        self.total_poses = total_poses
        self.hold_time = hold_time
        self.wrong_pose_frames = wrong_pose_frames
        self.hold_grace_time = hold_grace_time

        # Current sequence step
        self.current_step = 1

        # Hold timing
        self.hold_start_time = None

        # Last time posture was valid
        self.last_valid_time = None

        # Last time expected pose was detected
        self.last_expected_pose_time = None

        # Wrong pose tracking
        self.wrong_pose_count = 0
        self.last_raw_pose = None

        self.completed = False

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.current_step = 1

        self.hold_start_time = None
        self.last_valid_time = None
        self.last_expected_pose_time = None

        self.wrong_pose_count = 0
        self.last_raw_pose = None

        self.completed = False

    # ========================================================
    # CURRENT STEP
    # ========================================================

    def get_expected_step(self):

        return self.current_step

    # ========================================================
    # EXPECTED CLASSIFIER POSE
    # ========================================================

    def get_expected_pose(self):

        if self.completed:
            return None

        return self.sequence[
            self.current_step - 1
        ]

    # ========================================================
    # HOLD ELAPSED
    # ========================================================

    def get_hold_elapsed(self):

        if self.hold_start_time is None:
            return 0.0

        return (
            time.time() -
            self.hold_start_time
        )

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def update(
        self,
        raw_pose,
        posture_valid
    ):

        current_time = time.time()

        expected_pose = (
            self.get_expected_pose()
        )

        result = {

            "expected_step":
                self.current_step,

            "expected_pose":
                expected_pose,

            "state":
                "waiting",

            "message":
                "",

            "hold_elapsed":
                0.0,

            "completed_step":
                None,

            "session_complete":
                False
        }

        # ====================================================
        # SESSION COMPLETE
        # ====================================================

        if self.completed:

            result["state"] = "complete"

            result["message"] = (
                "Excellent! "
                "Surya Namaskara completed."
            )

            result["session_complete"] = True

            return result

        # ====================================================
        # EXPECTED POSE DETECTED
        # ====================================================

        if raw_pose == expected_pose:

            # Expected pose is back.
            self.last_expected_pose_time = (
                current_time
            )

            # Reset wrong-pose counter.
            self.wrong_pose_count = 0

            self.last_raw_pose = raw_pose

            # ------------------------------------------------
            # POSTURE CORRECT
            # ------------------------------------------------

            if posture_valid:

                # Start timer if necessary.
                if self.hold_start_time is None:

                    self.hold_start_time = (
                        current_time
                    )

                self.last_valid_time = (
                    current_time
                )

                elapsed = (
                    current_time -
                    self.hold_start_time
                )

                result["state"] = (
                    "holding"
                )

                result["hold_elapsed"] = (
                    elapsed
                )

                result["message"] = (
                    f"Good! Hold for "
                    f"{elapsed:.1f} / "
                    f"{self.hold_time:.1f} sec"
                )

                # ============================================
                # COMPLETE CURRENT STEP
                # ============================================

                if elapsed >= self.hold_time:

                    completed_step = (
                        self.current_step
                    )

                    self.current_step += 1

                    self._reset_hold()

                    result["completed_step"] = (
                        completed_step
                    )

                    # ========================================
                    # ALL 12 COMPLETE
                    # ========================================

                    if (
                        self.current_step >
                        self.total_poses
                    ):

                        self.completed = True

                        result["state"] = (
                            "complete"
                        )

                        result["message"] = (
                            "Excellent! "
                            "Surya Namaskara "
                            "completed."
                        )

                        result[
                            "session_complete"
                        ] = True

                    # ========================================
                    # NEXT STEP
                    # ========================================

                    else:

                        result["state"] = (
                            "pose_complete"
                        )

                        result["message"] = (
                            f"Step "
                            f"{completed_step} "
                            f"complete. "
                            f"Now perform Step "
                            f"{self.current_step}."
                        )

                return result

            # ------------------------------------------------
            # EXPECTED POSE BUT POSTURE NOT VALID
            # ------------------------------------------------

            if (
                self.hold_start_time is not None
                and
                self.last_valid_time is not None
                and
                (
                    current_time -
                    self.last_valid_time
                ) <= self.hold_grace_time
            ):

                elapsed = (
                    current_time -
                    self.hold_start_time
                )

                result["state"] = (
                    "holding"
                )

                result["hold_elapsed"] = (
                    elapsed
                )

                result["message"] = (
                    f"Keep holding. "
                    f"{elapsed:.1f} / "
                    f"{self.hold_time:.1f} sec"
                )

                return result

            # Posture has been invalid too long.
            self._reset_hold()

            result["state"] = (
                "correct_pose_adjust"
            )

            result["message"] = (
                f"Step "
                f"{self.current_step} "
                f"detected. "
                f"Adjust your posture."
            )

            return result

        # ====================================================
        # TEMPORARY UNKNOWN / DETECTION LOSS
        # ====================================================

        if raw_pose is None:

            self.wrong_pose_count = 0

            # If we were already holding the expected pose,
            # allow a short classifier dropout.

            if (
                self.hold_start_time is not None
                and
                self.last_expected_pose_time is not None
            ):

                dropout_time = (
                    current_time -
                    self.last_expected_pose_time
                )

                if (
                    dropout_time <=
                    self.hold_grace_time
                ):

                    elapsed = (
                        current_time -
                        self.hold_start_time
                    )

                    result["state"] = (
                        "holding"
                    )

                    result["hold_elapsed"] = (
                        elapsed
                    )

                    result["message"] = (
                        f"Detection unstable - "
                        f"keep holding. "
                        f"{elapsed:.1f} / "
                        f"{self.hold_time:.1f} sec"
                    )

                    return result

            # Grace period expired.
            self._reset_hold()

            result["state"] = (
                "waiting"
            )

            result["message"] = (
                f"Perform Step "
                f"{self.current_step}."
            )

            return result

        # ====================================================
        # ANOTHER POSE DETECTED
        # ====================================================

        # IMPORTANT:
        #
        # DO NOT reset the hold immediately.
        #
        # First determine whether the other pose persists
        # for the required number of frames.

        if raw_pose == self.last_raw_pose:

            self.wrong_pose_count += 1

        else:

            self.wrong_pose_count = 1

            self.last_raw_pose = raw_pose

        # ====================================================
        # TEMPORARY WRONG CLASSIFICATION
        # ====================================================

        if (
            self.wrong_pose_count <
            self.wrong_pose_frames
        ):

            # If we were already holding the expected pose,
            # preserve the timer while checking whether this
            # wrong classification is temporary.

            if self.hold_start_time is not None:

                elapsed = (
                    current_time -
                    self.hold_start_time
                )

                result["state"] = (
                    "holding"
                )

                result["hold_elapsed"] = (
                    elapsed
                )

                result["message"] = (
                    f"Checking pose... "
                    f"keep holding. "
                    f"{elapsed:.1f} / "
                    f"{self.hold_time:.1f} sec"
                )

            else:

                result["state"] = (
                    "checking"
                )

                result["message"] = (
                    f"Checking pose... "
                    f"perform Step "
                    f"{self.current_step}."
                )

            return result

        # ====================================================
        # WRONG POSE CONFIRMED
        # ====================================================

        self._reset_hold()

        result["state"] = (
            "wrong_pose"
        )

        result["message"] = (
            f"Wrong pose! "
            f"Step {self.current_step} "
            f"should be performed here."
        )

        return result

    # ========================================================
    # RESET HOLD
    # ========================================================

    def _reset_hold(self):

        self.hold_start_time = None

        self.last_valid_time = None

        self.last_expected_pose_time = None