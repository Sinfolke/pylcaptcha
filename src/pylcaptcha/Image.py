import cv2
import numpy as np


class Image:
    def __init__(self, path: str, size=3):
        self.path = str(path)
        self.img = cv2.imread(self.path)
        if self.img is None: raise ValueError("Image not found")
        self.size = size
        self.rows = size
        self.cols = size

    def _find_thick_grid(self):
        # 1. Isolate bright areas (Grid + Sky + Bag)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        # 2. Use a "Vertical Kernel" to find ONLY vertical lines
        # This ignores the white bag because the bag isn't a perfectly straight tall line
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detected_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

        # 3. Use a "Horizontal Kernel" to find ONLY horizontal lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detected_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)

        # 4. Get the coordinates where these lines peak
        v_proj = np.mean(detected_v, axis=0)
        h_proj = np.mean(detected_h, axis=1)

        # We look for peaks that are near our expected 1/3, 2/3 marks
        v_lines = self._get_peaks(v_proj, self.img.shape[1])
        h_lines = self._get_peaks(h_proj, self.img.shape[0])

        return v_lines, h_lines

    def _get_peaks(self, projection, dimension):
        indices = np.where(projection > 150)[0]
        if len(indices) == 0: return []

        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)

        # Precompute invariant factors outside the loop
        expected_marks = [(i / self.size) * dimension for i in range(1, self.size)]
        tolerance = dimension * 0.05
        best_lines = []

        for group in groups:
            center = int(np.mean(group))
            for expected in expected_marks:
                if abs(center - expected) < tolerance:
                    best_lines.extend([center - 1, center, center + 1])
                    break

        return list(set(best_lines))

    def clean(self, overwrite=True):
        v_indices, h_indices = self._find_thick_grid()

        # Create a mask of the pixels we want to "remove"
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        if v_indices:
            mask[:, v_indices] = 255
        if h_indices:
            mask[h_indices, :] = 255

        # Use inpainting to fill the grid lines naturally
        # 3 is the neighborhood radius; INPAINT_TELEA is fast and effective
        result = cv2.inpaint(self.img, mask, 3, cv2.INPAINT_TELEA)

        if overwrite:
            self.img = result
            cv2.imwrite(self.path, result)
        return result