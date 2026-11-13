from collections import deque


class TemporalStability:
    def __init__(self, window_size=7, required_count=5):
        self.window_size = window_size
        self.required_count = required_count
        self.history = deque(maxlen=window_size)

    def update(self, pose_id):
        self.history.append(pose_id)

        if len(self.history) < self.required_count:
            return None

        counts = {}

        for pose in self.history:
            counts[pose] = counts.get(pose, 0) + 1

        stable_pose = max(counts, key=counts.get)

        if counts[stable_pose] >= self.required_count:
            return stable_pose

        return None

    def reset(self):
        self.history.clear()