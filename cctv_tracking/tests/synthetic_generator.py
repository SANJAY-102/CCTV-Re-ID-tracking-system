"""
Synthetic Video and Detection Scenario Generator for CCTV Tracking Tests.
Generates realistic multi-person trajectories, crossing paths, occlusions,
disappearances, fast movements, and re-entries.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

from ..detector.yolo_detector import PersonDetection


class SyntheticPerson:
    """
    Simulated person with distinct visual appearance patterns and trajectory.
    """
    def __init__(
        self,
        person_id: int,
        initial_pos: Tuple[float, float],
        velocity: Tuple[float, float],
        size: Tuple[int, int] = (40, 100),  # width, height
        primary_color: Tuple[int, int, int] = (200, 50, 50),
        secondary_color: Tuple[int, int, int] = (50, 200, 50)
    ):
        self.person_id = person_id
        self.x, self.y = float(initial_pos[0]), float(initial_pos[1])
        self.vx, self.vy = float(velocity[0]), float(velocity[1])
        self.width, self.height = size
        self.primary_color = primary_color
        self.secondary_color = secondary_color
        self.is_visible = True

    @property
    def bbox(self) -> np.ndarray:
        x1 = self.x - self.width / 2.0
        y1 = self.y - self.height / 2.0
        x2 = self.x + self.width / 2.0
        y2 = self.y + self.height / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (self.x, self.y + self.height / 2.0)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def step(self):
        """Move one time step."""
        if self.is_visible:
            self.x += self.vx
            self.y += self.vy

    def draw(self, canvas: np.ndarray):
        """Draw realistic silhouette with distinctive torso and limbs on canvas."""
        if not self.is_visible:
            return

        x1, y1, x2, y2 = [int(round(v)) for v in self.bbox]
        h_img, w_img = canvas.shape[:2]

        if x2 <= 0 or y2 <= 0 or x1 >= w_img or y1 >= h_img:
            return

        # Torso
        torso_y1 = y1 + int(self.height * 0.2)
        torso_y2 = y1 + int(self.height * 0.65)
        cv2.rectangle(canvas, (max(0, x1), max(0, torso_y1)), (min(w_img, x2), min(h_img, torso_y2)), self.primary_color, -1)

        # Head
        head_radius = int(self.width * 0.35)
        head_center = (int(self.x), int(y1 + self.height * 0.12))
        cv2.circle(canvas, head_center, max(1, head_radius), (210, 180, 140), -1)

        # Legs
        legs_y1 = torso_y2
        legs_y2 = y2
        cv2.rectangle(canvas, (max(0, x1), max(0, legs_y1)), (min(w_img, x2), min(h_img, legs_y2)), self.secondary_color, -1)


class SyntheticScenarioGenerator:
    """
    Produces repeatable synthetic sequences for test scenarios.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def _simulate(self, people: List[SyntheticPerson], num_frames: int) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        frames: List[np.ndarray] = []
        detections_seq: List[List[PersonDetection]] = []

        for f in range(num_frames):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dets: List[PersonDetection] = []
            for p in people:
                if p.is_visible:
                    p.draw(canvas)
                    det = PersonDetection(
                        bbox=p.bbox.copy(),
                        confidence=0.92,
                        center=p.center,
                        bottom_center=p.bottom_center,
                        frame_id=f
                    )
                    det.extract_crop(canvas)
                    dets.append(det)
                p.step()

            frames.append(canvas)
            detections_seq.append(dets)

        return frames, detections_seq

    def generate_single_moving_person(self, num_frames: int = 40) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 1: Single person walking smoothly left to right."""
        person = SyntheticPerson(1, (100, 360), (5.0, 0.0), (50, 120), (220, 50, 50), (50, 50, 220))
        return self._simulate([person], num_frames)

    def generate_multiple_moving_people(self, num_frames: int = 60) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 2: Multiple people moving parallel and diagonal."""
        p1 = SyntheticPerson(1, (100, 200), (4.0, 1.0), (50, 120), (220, 30, 30), (30, 30, 220))
        p2 = SyntheticPerson(2, (100, 450), (3.5, -0.5), (50, 120), (30, 200, 30), (200, 200, 30))
        p3 = SyntheticPerson(3, (800, 300), (-4.0, 0.0), (50, 120), (200, 30, 200), (30, 200, 200))
        return self._simulate([p1, p2, p3], num_frames)

    def generate_crossing_people(self, num_frames: int = 60) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 3: Two people walking toward each other, crossing paths at center, continuing."""
        p1 = SyntheticPerson(1, (200, 360), (4.0, 0.0), (50, 120), (220, 20, 20), (20, 20, 220))
        p2 = SyntheticPerson(2, (600, 360), (-4.0, 0.0), (50, 120), (20, 220, 20), (220, 220, 20))
        return self._simulate([p1, p2], num_frames)

    def generate_fast_movement(self, num_frames: int = 30) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 4: Fast moving person with rapid velocity."""
        p1 = SyntheticPerson(1, (100, 360), (16.0, 0.0), (50, 120), (220, 100, 20), (20, 100, 220))
        return self._simulate([p1], num_frames)

    def generate_temporary_detection_miss(self, num_frames: int = 35, miss_start: int = 15, miss_len: int = 4) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 5: Person tracked, miss for a few frames, then detected again."""
        person = SyntheticPerson(1, (100, 360), (4.0, 0.0), (50, 120), (220, 50, 50), (50, 50, 220))
        frames: List[np.ndarray] = []
        detections_seq: List[List[PersonDetection]] = []

        for f in range(num_frames):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            is_miss = (miss_start <= f < miss_start + miss_len)
            person.draw(canvas)

            dets: List[PersonDetection] = []
            if not is_miss:
                det = PersonDetection(
                    bbox=person.bbox.copy(),
                    confidence=0.92,
                    center=person.center,
                    bottom_center=person.bottom_center,
                    frame_id=f
                )
                det.extract_crop(canvas)
                dets.append(det)

            frames.append(canvas)
            detections_seq.append(dets)
            person.step()

        return frames, detections_seq

    def generate_person_exit_and_return(self, num_frames: int = 140) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Scenario 6 & 8: Complete tracker failure -> person leaves, track deleted, person returns."""
        person = SyntheticPerson(1, (100, 360), (4.0, 0.0), (50, 120), (220, 50, 50), (50, 50, 220))
        frames: List[np.ndarray] = []
        detections_seq: List[List[PersonDetection]] = []

        for f in range(num_frames):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            is_present = (f <= 30) or (f >= 100)
            if f == 100:
                person.x = 900
                person.y = 360
                person.vx = -4.0
                person.vy = 0.0

            person.is_visible = is_present
            person.draw(canvas)

            dets: List[PersonDetection] = []
            if is_present:
                det = PersonDetection(
                    bbox=person.bbox.copy(),
                    confidence=0.94,
                    center=person.center,
                    bottom_center=person.bottom_center,
                    frame_id=f
                )
                det.extract_crop(canvas)
                dets.append(det)

            frames.append(canvas)
            detections_seq.append(dets)
            if is_present:
                person.step()

        return frames, detections_seq

    def generate_6_moving_people(self, num_frames: int = 1000) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """Generate 6 distinct moving people tracked continuously in horizontal lanes."""
        colors = [
            ((220, 50, 50), (50, 220, 50)),
            ((50, 50, 220), (220, 220, 50)),
            ((50, 220, 220), (220, 50, 220)),
            ((200, 100, 50), (50, 100, 200)),
            ((100, 200, 50), (200, 50, 100)),
            ((150, 150, 50), (50, 150, 150))
        ]
        people = []
        for i in range(6):
            p = SyntheticPerson(
                person_id=i + 1,
                initial_pos=(150 + i * 160, 120 + i * 85),
                velocity=((1.5 if i % 2 == 0 else -1.5), 0.0),
                size=(50, 120),
                primary_color=colors[i][0],
                secondary_color=colors[i][1]
            )
            people.append(p)

        frames: List[np.ndarray] = []
        detections_seq: List[List[PersonDetection]] = []

        for f in range(num_frames):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dets: List[PersonDetection] = []

            for p in people:
                if p.x <= 90 or p.x >= self.width - 90:
                    p.vx *= -1.0

                p.draw(canvas)
                det = PersonDetection(
                    bbox=p.bbox.copy(),
                    confidence=0.94,
                    center=p.center,
                    bottom_center=p.bottom_center,
                    frame_id=f
                )
                det.extract_crop(canvas)
                dets.append(det)
                p.step()

            frames.append(canvas)
            detections_seq.append(dets)

        return frames, detections_seq

    def generate_6_people_with_single_drop(
        self,
        num_frames: int = 1000,
        drop_person_idx: int = 0,
        drop_start: int = 150,
        drop_duration: int = 100
    ) -> Tuple[List[np.ndarray], List[List[PersonDetection]]]:
        """
        Generate 6 moving people where exactly 1 person artificially disappears,
        exceeds max_lost_frames, and then returns / respawns later.
        """
        colors = [
            ((220, 50, 50), (50, 220, 50)),
            ((50, 50, 220), (220, 220, 50)),
            ((50, 220, 220), (220, 50, 220)),
            ((200, 100, 50), (50, 100, 200)),
            ((100, 200, 50), (200, 50, 100)),
            ((150, 150, 50), (50, 150, 150))
        ]
        people = []
        for i in range(6):
            p = SyntheticPerson(
                person_id=i + 1,
                initial_pos=(150 + i * 160, 120 + i * 85),
                velocity=((1.5 if i % 2 == 0 else -1.5), 0.0),
                size=(50, 120),
                primary_color=colors[i][0],
                secondary_color=colors[i][1]
            )
            people.append(p)

        frames: List[np.ndarray] = []
        detections_seq: List[List[PersonDetection]] = []

        for f in range(num_frames):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dets: List[PersonDetection] = []

            for idx, p in enumerate(people):
                if idx == drop_person_idx:
                    is_dropped = (drop_start <= f < drop_start + drop_duration)
                    p.is_visible = not is_dropped
                    if f == drop_start + drop_duration:
                        p.x = 250.0
                        p.y = 120.0
                        p.vx = 1.5
                        p.vy = 0.0

                if p.is_visible:
                    if p.x <= 90 or p.x >= self.width - 90:
                        p.vx *= -1.0

                    p.draw(canvas)
                    det = PersonDetection(
                        bbox=p.bbox.copy(),
                        confidence=0.94,
                        center=p.center,
                        bottom_center=p.bottom_center,
                        frame_id=f
                    )
                    det.extract_crop(canvas)
                    dets.append(det)
                    p.step()

            frames.append(canvas)
            detections_seq.append(dets)

        return frames, detections_seq
