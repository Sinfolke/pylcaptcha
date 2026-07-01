import asyncio
import random
import secrets
from pathlib import Path
from typing import Optional

from playwright.async_api import Page
from ultralytics import YOLO
import time
import cv2
import numpy as np
from pylcaptcha.ShyMouse import ShyMouse
from pylcaptcha.Image import Image
from pylcaptcha.playwright_wrapper import DOMNode, DOM

PARENT = Path(__file__).resolve().parent
# Map your custom repository tags to exact standard COCO strings
REPO_TO_COCO_MAPPING = {
    "Bus": "bus",
    "Traffic_Light": "traffic light",
    "Hydrant": "fire hydrant",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Car": "car"
}
class Captcha:
    DEFAULT_OPTIONS = {
        "models_root": PARENT / 'models',
        "detection_model": "Detectionv4.pt",
    }
    # Wrapper Captcha keyword -> Model class name
    CAPTCHA_MAPPING = {
        "stair": "Stair", "сходи": "Stair", "лестниц": "Stair",
        "bus": "Bus", "автобус": "Bus",
        "traffic light": "Traffic_Light", "светофор": "Traffic_Light",
        "hydrant": "Hydrant", "гидрант": "Hydrant",
        "crosswalk": "Crosswalk", "пешеход": "Crosswalk",
        "bicycle": "Bicycle", "велосипед": "Bicycle",
        "motorcycle": "Motorcycle", "мотоцикл": "Motorcycle",
        "car": "Car", "автомобил": "Car", "машин": "Car",
        "taxi": "Car",
        "palm": "Palm", "пальм": "Palm",
        "chimney": "Chimney", "дымоход": "Chimney",
        "bridge": "Bridge", "мост": "Bridge"
    }
    CLASS_THRESHOLD: dict[str, float] = {
        # High Noise Classes (Requires high certainty to avoid selecting fences/trees)
        "bicycle": 0.78,
        "chimney": 0.80,
        "stair": 0.75,
        "palm": 0.75,

        # Large Objects (Easier to detect, but can be confused with other vehicles)
        "bus": 0.72,
        "bridge": 0.70,
        "car": 0.65,
        "motorcycle": 0.72,

        # Small/Specific Objects (Often missed, so we allow slightly lower confidence)
        "traffic_light": 0.68,
        "hydrant": 0.65,
        "crosswalk": 0.60,

        # Fallback for any class not listed
        "default": 0.70
    }
    def __init__(self, page: Page, mouse = None, YoloSeg: bool = True, options: dict[str, str] = None, captcha_wrapper: dict[str, str] = None, class_threshold: dict[str, str] = None):
        """
        Note: mouse defaults to ShyMouse, but can be accessible if  you provided the 'page' option, otherwise invoke openPage to have mouse filled
        :param page: the page that contains google captcha
        :param mouse: the mouse handler provides human clicks. Must contain .move and .click methods, the rest is on your own
        """
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self.captcha_mapper = {**self.CAPTCHA_MAPPING, **(captcha_wrapper or {})}
        self.class_threshold = {**self.CLASS_THRESHOLD, **(class_threshold or {})}
        self.page = page
        self.mouse = mouse if mouse else ShyMouse(page)
        self.models = {}
        self.model_paths = {}
        models_root = Path(self.options.get("models_root"))
        if YoloSeg:
            self.yolo_seg = YOLO(models_root / 'yolo26l.pt')
            self.yolo_cls = YOLO(models_root / 'yolo26l-cls.pt')
        else:
            # Map the available models without building them yet
            for model in models_root.iterdir():
                if model.name == self.options.get("detection_model"):
                    self.models["detection"] = YOLO(str(model))  # Load detection model eagerly
                    continue
                if model.name.startswith("Detection"):
                    continue
                self.model_paths[model.stem] = str(model)

    def _get_model(self, target_label: str) -> YOLO:
        """Retrieves a cached model instance or lazy-loads it on demand."""
        if self.yolo_seg:
            return self.yolo_seg
        if target_label not in self.models:
            path = self.model_paths.get(target_label)
            if not path:
                raise ValueError(f"No local model configured for class target: {target_label}")
            print(f"⚡ Lazy-loading model for class: {target_label}")
            self.models[target_label] = YOLO(path)
        return self.models[target_label]
    def _get_classification_prob(self, tile_img, target: str):
        """Helper to get a probability from the specific target model"""
        if tile_img.size == 0: return 0.0
        res = self._get_model(target).predict(tile_img, imgsz=128, verbose=False)[0]
        # Reusing your existing pos/neg logic
        probs = res.probs.data.tolist()
        for j, name in enumerate(res.names.values()):
            if name == "pos":
                return probs[j]
        return 0.0
    def _expand_box(self, x1, y1, x2, y2, img_w, img_h, scale=1.15):
        w = x2 - x1
        h = y2 - y1

        cx = x1 + w / 2
        cy = y1 + h / 2

        nw = w * scale
        nh = h * scale

        x1 = max(0, cx - nw / 2)
        y1 = max(0, cy - nh / 2)
        x2 = min(img_w, cx + nw / 2)
        y2 = min(img_h, cy + nh / 2)

        return x1, y1, x2, y2

    def _box_to_tiles(self, x1, y1, x2, y2, tile_w, tile_h, cols, rows):
        tiles = set()

        c1 = int(x1 // tile_w)
        c2 = int(x2 // tile_w)
        r1 = int(y1 // tile_h)
        r2 = int(y2 // tile_h)

        for r in range(max(0, r1), min(rows, r2 + 1)):
            for c in range(max(0, c1), min(cols, c2 + 1)):
                tiles.add(r * cols + c)

        return tiles

    def _extract_tile(self, img, idx, tile_w, tile_h, size):
        r = idx // size
        c = idx % size

        y1 = int(r * tile_h)
        y2 = int((r + 1) * tile_h)
        x1 = int(c * tile_w)
        x2 = int((c + 1) * tile_w)

        return img[y1:y2, x1:x2]

    def _get_pos_from_result(self, res):
        probs = res.probs.data

        # safer than name matching
        pos_idx = list(res.names.values()).index("pos")

        return float(probs[pos_idx])

    def _aggregate_tiles(self, meta, probs, img, rows, cols, threshold):
        h, w, _ = img.shape
        tile_h = h / rows
        tile_w = w / cols

        scores = [0.0] * (rows * cols)
        counts = [0] * (rows * cols)

        for (y, x), prob in zip(meta, probs):
            if prob < threshold:
                continue

            r1 = int(y // tile_h)
            r2 = int((y + tile_h) // tile_h)
            c1 = int(x // tile_w)
            c2 = int((x + tile_w) // tile_w)

            for r in range(max(0, r1), min(rows, r2 + 1)):
                for c in range(max(0, c1), min(cols, c2 + 1)):
                    idx = r * cols + c
                    scores[idx] += prob
                    counts[idx] += 1

        # normalize score
        for i in range(len(scores)):
            if counts[i] > 0:
                scores[i] /= counts[i]

        return scores, counts

    def _predict_pos_scores(self, tiles, target_label):
        """Internal helper to get raw 'pos' confidence scores for a batch of tiles."""
        model = self._get_model(target_label)
        if not model:
            return [0.0] * len(tiles)

        # Batch inference
        results = model.predict(tiles, imgsz=128, verbose=False)
        scores = []

        for res in results:
            if res.probs is None:
                scores.append(0.0)
                continue

            # Safely map the 'pos' class score
            names_inv = {v: k for k, v in res.names.items()}
            pos_idx = names_inv.get("pos")
            scores.append(float(res.probs.data[pos_idx]) if pos_idx is not None else 0.0)

        return scores

    # 2. Update _ai_verify_single_tile to safely accept both bytes and paths:
    def _ai_verify_single_tile(self, tile_source: bytes | str | Path, target_label: str) -> bool:
        if isinstance(tile_source, bytes):
            # Decode the image directly from the RAM buffer
            tile_img = cv2.imdecode(np.frombuffer(tile_source, np.uint8), cv2.IMREAD_COLOR)
        else:
            tile_img = cv2.imread(str(tile_source))

        if tile_img is None:
            return False

        scores = self._predict_pos_scores([tile_img], target_label)
        return scores[0] >= 0.55

    def _solve_with_detection_model(self, img, target: str):
        det_model = self.models["detection"]

        results = det_model.predict(
            img.img,
            imgsz=416,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )

        h, w, _ = img.img.shape
        rows, cols = img.rows, img.cols
        tile_w, tile_h = w / cols, h / rows
        tile_area = tile_w * tile_h

        selected_tiles = set()

        # MINIMUM OVERLAP THRESHOLD (Adjust this based on performance)
        # 0.15 means the bounding box must cover at least 15% of the tile to count.
        MIN_TILE_OVERLAP_RATIO = 0.12

        for res in results:
            for box in res.boxes:
                cls_name = res.names[int(box.cls[0])].lower()
                if cls_name != target.lower():
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])

                # OPTIONAL: Bounding Box Contraction (Shrink box by 5% to eliminate edge-bleed)
                box_w = x2 - x1
                box_h = y2 - y1
                x1 += box_w * 0.05
                x2 -= box_w * 0.05
                y1 += box_h * 0.05
                y2 -= box_h * 0.05

                # Get the rough grid boundaries the box spans
                c1 = max(0, int(x1 // tile_w))
                c2 = min(cols - 1, int(x2 // tile_w))
                r1 = max(0, int(y1 // tile_h))
                r2 = min(rows - 1, int(y2 // tile_h))

                # Evaluate each overlapping tile individually
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        # Define the absolute pixel boundaries of the current grid tile
                        tile_x1 = c * tile_w
                        tile_y1 = r * tile_h
                        tile_x2 = (c + 1) * tile_w
                        tile_y2 = (r + 1) * tile_h

                        # Calculate intersection coordinates
                        inter_x1 = max(x1, tile_x1)
                        inter_y1 = max(y1, tile_y1)
                        inter_x2 = min(x2, tile_x2)
                        inter_y2 = min(y2, tile_y2)

                        # Compute intersection dimensions
                        inter_w = inter_x2 - inter_x1
                        inter_h = inter_y2 - inter_y1

                        if inter_w > 0 and inter_h > 0:
                            inter_area = inter_w * inter_h

                            # Ratio 1: How much of the tile is covered by this detection?
                            tile_overlap_ratio = inter_area / tile_area

                            # Ratio 2: How much of the detection itself lies inside this tile?
                            # (Useful for tiny objects that don't fill 15% of a tile but are entirely inside it)
                            box_area = (x2 - x1) * (y2 - y1)
                            box_contained_ratio = inter_area / box_area if box_area > 0 else 0

                            # Select tile if it passes either threshold rule
                            if tile_overlap_ratio >= MIN_TILE_OVERLAP_RATIO or box_contained_ratio > 0.65:
                                selected_tiles.add(r * cols + c)

        return sorted(selected_tiles)

    def _solve_with_classification_model(self, img, threshold, target):
        h, w, _ = img.img.shape
        rows, cols = img.rows, img.cols

        # 🛡️ FIX 1: Transition to floats to prevent edge pixel loss
        tile_h = h / rows
        tile_w = w / cols

        # 1. Extract tiles using accurate coordinate boundaries
        tiles = []
        for r in range(rows):
            for c in range(cols):
                r_start = int(r * tile_h)
                r_end = int((r + 1) * tile_h) if r < rows - 1 else h
                c_start = int(c * tile_w)
                c_end = int((c + 1) * tile_w) if c < cols - 1 else w
                tiles.append(img.img[r_start:r_end, c_start:c_end])

        # 2. Get Scores from shared helper
        scores = self._predict_pos_scores(tiles, target)

        # 3. Dynamic Gapping
        sorted_scores = sorted(scores, reverse=True)
        max_score = sorted_scores[0] if sorted_scores else 0.0
        if max_score < 0.25:
            return []

        # 4. Adaptive Thresholding (🛡️ FIX 2: Isolated to 4x4 continuous frames)
        if rows > 3:
            # For continuous images sliced into 16 parts, scale threshold dynamically
            base_thresh = max(threshold, max_score * 0.7)
        else:
            # For 3x3 independent photo matrices, preserve the absolute baseline
            base_thresh = threshold

        selected = {i for i, s in enumerate(scores) if s >= base_thresh}

        # 5. Spatial Consistency (4x4 grids only)
        if rows > 3 and len(selected) > 0:
            def get_neighbors(idx):
                r, c = divmod(idx, cols)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        yield nr * cols + nc

            refined = set()
            for i in selected:
                neighbors = list(get_neighbors(i))
                has_selected_neighbor = any(n in selected for n in neighbors)

                # 🛡️ FIX 3: Safe-edge thresholds for small independent structures
                if has_selected_neighbor or scores[i] > 0.85:
                    refined.add(i)
                else:
                    # Local Peak Evaluation: If this tile scores higher than all its
                    # immediate neighbors, it is likely a standalone target (e.g., a traffic light)
                    is_local_max = all(scores[i] > scores[n] for n in neighbors)
                    if is_local_max and scores[i] >= threshold:
                        refined.add(i)

            selected = refined

        return sorted(selected)

    def _solve_with_segmentation_model(self, img, target: str):
        """
        Solves ReCaptcha using a generic YOLO Segmentation model.
        Checks if the actual object pixels overlap with grid sectors.
        """
        # 1. Fallback to the loaded segmentation model
        seg_model = self.yolo_seg

        # 2. Convert target prompt to standard COCO dataset string format
        normalized_target = REPO_TO_COCO_MAPPING.get(target.lower(), target.lower())

        # 3. Predict on the FULL image grid at once
        results = seg_model.predict(
            img.img,
            imgsz=640,  # 640px is highly recommended for segmentation models
            conf=0.25,  # Adjust confidence if it's missing trickier elements
            iou=0.45,
            retina_masks=True,  # CRITICAL: Forces mask array dimensions to match img.img shape perfectly
            verbose=False,
        )

        h, w, _ = img.img.shape
        rows, cols = img.rows, img.cols
        tile_w = w / cols
        tile_h = h / rows

        selected_tiles = set()

        # MINIMUM PIXEL THRESHOLD
        # Requires at least 15 pixels of the target object to exist in a tile to prevent noise selection
        MIN_PIXEL_THRESHOLD = 15

        for res in results:
            if res.masks is None:
                continue

            # Extract underlying binary masks array and bounding boxes
            masks = res.masks.data
            boxes = res.boxes

            for box, mask in zip(boxes, masks):
                cls_idx = int(box.cls[0])
                cls_name = res.names[cls_idx].lower()

                # Ignore everything except our active target class
                if cls_name != normalized_target:
                    continue

                # Convert PyTorch tensor mask to a Boolean NumPy array
                mask_np = mask.cpu().numpy().astype(bool) if hasattr(mask, 'cpu') else mask.astype(bool)

                # Evaluate which grid positions host the object pixels
                for r in range(rows):
                    for c in range(cols):
                        r_start, r_end = int(r * tile_h), int((r + 1) * tile_h)
                        c_start, c_end = int(c * tile_w), int((c + 1) * tile_w)

                        # Slice out the mask segment for this specific tile
                        tile_mask_zone = mask_np[r_start:r_end, c_start:c_end]

                        # Count how many target pixels inhabit this tile
                        if tile_mask_zone.sum() > MIN_PIXEL_THRESHOLD:
                            selected_tiles.add(r * cols + c)

        return sorted(list(selected_tiles))
    def _get_next_screenshot_index(self, path: Path):
        max_i = 0
        for file in path.glob("captcha_task (*.png"):
            try:
                num = int(file.stem.split("(")[-1].rstrip(")").split('_')[0])
                max_i = max(max_i, num)
            except:
                pass
        return max_i + 1
    async def _get_g_token(self, document) -> str | None:
        """Helper to extract the response token from the hidden field."""
        try:
            token = await document.getElementById('g-recaptcha-response').value()
            return token if token and len(token) > 10 else None
        except:
            return None
    async def _make_click(self, element: DOMNode, fast = True, is_last=False):
        await self.mouse.click(element.locator, fast=fast)
    async def check_g_captcha(self, wait_ms = 3000):
        """
            Check that g captcha is on the page by waiting for it
            increase wait_ms on bad connection
        """
        document = DOM(self.page)
        try:
            anchor_frame = document.frame("iframe[title*='reCAPTCHA'], iframe[src*='api2/anchor']")
            anchor_checkbox = anchor_frame.querySelector('#recaptcha-anchor')
            await anchor_checkbox.waitVisible(wait_ms)
            return True
        except Exception:
            return False
    async def solve_g_captcha(self) -> str | None:
        screenshot_path = str(PARENT / "captcha-debug" / "last_captcha.png")
        screenshot_error_path = str(PARENT / "captcha-debug" / "debug_error.png")

        try:
            # --- PHASE 1: The Initial Checkbox ---
            print("Ожидание чекбокса reCAPTCHA...")
            document = DOM(self.page)
            anchor_frame = document.frame("iframe[title*='reCAPTCHA'], iframe[src*='api2/anchor']")
            anchor_checkbox = anchor_frame.querySelector('#recaptcha-anchor')
            await anchor_checkbox.waitVisible(30000)

            await self.mouse.move_to_element_naturally(anchor_checkbox.locator)
            await self._make_click(anchor_checkbox, fast=False)

            # Check for passive verification bypass
            await asyncio.sleep(1.2)
            if await anchor_checkbox.getAttribute("aria-checked") == "true":
                print("Капча решена автоматически!")
                return await self._get_g_token(document)

            print("Переход к решению графической капчи...")
            tiles_to_monitor = {}

            # --- PHASE 2: The Solving Loop ---
            while True:
                try:
                    await self.page.mouse.up()
                except Exception:
                    pass

                document = DOM(self.page)

                challenge_frame = None
                for _ in range(10):
                    challenge_frame = document.frame('iframe[title*="challenge"], iframe[src*="api2/bframe"]')
                    if challenge_frame: break
                    await asyncio.sleep(0.15)

                if not challenge_frame: raise Exception("Challenge frame not available")

                image_frame = document.frame('iframe[src*="api2/bframe"], iframe[title*="recaptcha challenge"]')
                instruction_element = image_frame.querySelector('.rc-imageselect-instructions')

                if not instruction_element:
                    token = await self._get_g_token(document)
                    if token: return token
                    continue

                # Identify Task Parameters
                full_text = (await instruction_element.innerText()).lower().replace("\n", " ")
                target_label = next((v for k, v in self.captcha_mapper.items() if k in full_text), "Bus")
                is_disappearing = "none left" in full_text
                current_mode = "classify" if any(p in full_text for p in ["none left", "images"]) else "detect"

                cells = image_frame.querySelectorAll('.rc-imageselect-tile')
                target_area = image_frame.querySelector('#rc-imageselect-target')
                verify_button = image_frame.querySelector('#recaptcha-verify-button')
                grid_size = 4 if await cells.count() == 16 else 3

                try:
                    await target_area.locator.wait_for(state="attached", timeout=4000)
                    await target_area.locator.screenshot(path=screenshot_path)
                    img = Image(screenshot_path, grid_size)
                except Exception as e:
                    print(f"Предупреждение: Область капчи обновляется, ожидаем стабилизации... ({e})")
                    await asyncio.sleep(1.0)
                    continue

                # stop_wander = asyncio.Event()
                # wander_task = asyncio.create_task(self.mouse.wander_randomly(target_area.locator, stop_wander))

                try:
                    if self.yolo_seg:
                        tile_indices = self._solve_with_segmentation_model(img, target_label)
                    else:
                        if current_mode == "detect":
                            img.clean(overwrite=True)
                            tile_indices = self._solve_with_detection_model(img, target_label)
                        else:
                            tile_indices = self._solve_with_classification_model(img, 0.52, target_label)
                except Exception as e:
                    print(f"Ошибка ИИ-модели: {e}")
                    tile_indices = []
                finally:
                    # FIX 1: Stop the random wandering worker BEFORE human-like clicking begins
                    # stop_wander.set()
                    try:
                        await wander_task
                    except Exception:
                        pass
                # --- DYNAMIC BLANK GRID EXIT CONDITION ---
                if not tile_indices:
                    print("Целей не обнаружено. Пробуем завершить...")
                    try:
                        await self.mouse.move_to_element_naturally(verify_button.locator)
                        await self._make_click(verify_button, fast=False)
                    except Exception:
                        await asyncio.sleep(1.0)
                        continue

                    # 🌟 FIXED CRITICAL VERIFICATION CHECK LIFT
                    is_solved = False
                    for _ in range(8):
                        await asyncio.sleep(0.4)
                        try:
                            if await anchor_checkbox.getAttribute("aria-checked") == "true":
                                is_solved = True
                                break
                        except Exception:
                            pass
                        try:
                            if not await target_area.isVisible():
                                is_solved = True
                                break
                        except Exception:
                            is_solved = True  # Element detached/missing means frame closed
                            break

                    if is_solved:
                        print("Капча успешно пройдена!")
                        for _ in range(15):
                            token = await self._get_g_token(document)
                            if token: return token
                            await asyncio.sleep(0.5)
                        raise Exception("Captcha solved but token retrieval timed out")

                    print("Появилась новая задача...")
                    await asyncio.sleep(random.uniform(0.8, 1.3))
                    continue
                elif not is_disappearing and len(tile_indices) < 3:
                    allowed_numbers = [num for num in range(0, 10) if num not in tile_indices]
                    for i in range(3 - len(tile_indices)):
                        tile_indices.append(random.choice(allowed_numbers))
                # --- PHASE 3: Selection Execution ---
                try:
                    if is_disappearing:
                        clicked_indices = set(tile_indices)

                        for idx in tile_indices:
                            tile = cells.nth(idx)
                            tiles_to_monitor[idx] = await tile.locator.locator('img').get_attribute('src')

                            await self.mouse.move_to_tile_naturally(tile.locator)
                            await self._make_click(tile, fast=True)
                            await asyncio.sleep(random.uniform(0.12, 0.28))

                        while clicked_indices:
                            # If you want to wander while waiting for tiles to fade,
                            # manage its lifetime precisely within this inner block scope.
                            # stop_reactive = asyncio.Event()
                            # reactive_task = asyncio.create_task(
                            #     self.mouse.wander_randomly(target_area.locator, stop_reactive))

                            ready_to_check = []
                            start_poll = time.time()

                            while time.time() - start_poll < 7:
                                for idx in list(clicked_indices):
                                    tile = cells.nth(idx)
                                    if await self._check_tile_ready(tile, tiles_to_monitor.get(idx)):
                                        ready_to_check.append(idx)
                                        clicked_indices.remove(idx)

                                if len(ready_to_check) >= 2 or (not clicked_indices and ready_to_check):
                                    break
                                await asyncio.sleep(0.2)

                            await asyncio.sleep(0.5)
                            # Stop the wander task before executing clicks on fresh tiles
                            # stop_reactive.set()
                            try:
                                await reactive_task
                            except Exception:
                                pass

                            if time.time() - start_poll >= 7:
                                clicked_indices.clear()

                            for idx in ready_to_check:
                                tile = cells.nth(idx)
                                img_bytes = await tile.locator.screenshot()
                                if self._ai_verify_single_tile(img_bytes, target_label):
                                    tiles_to_monitor[idx] = await tile.locator.locator('img').get_attribute('src')
                                    # No collision now because reactive_task was joined above
                                    await self.mouse.move_to_element_naturally(tile.locator)
                                    await self._make_click(tile, fast=True)
                                    clicked_indices.add(idx)
                                    await asyncio.sleep(random.uniform(0.2, 0.4))

                        await asyncio.sleep(0.6)
                        try:
                            if await anchor_checkbox.getAttribute("aria-checked") == "true":
                                print("Капча успешно пройдена во время исчезающего раунда!")
                                for _ in range(15):
                                    token = await self._get_g_token(document)
                                    if token: return token
                                    await asyncio.sleep(0.5)
                        except Exception:
                            pass
                        continue

                    else:
                        # --- STATIC MODE TRACK ---
                        for i, index in enumerate(tile_indices):
                            tile = cells.nth(index)
                            if i > 0:
                                visual_search_delay = random.uniform(0.14, 0.28)
                                await asyncio.sleep(visual_search_delay)

                            await self.mouse.move_to_tile_naturally(tile.locator)
                            await self._make_click(tile, fast=True, is_last=(i == len(tile_indices) - 1))

                            if i < len(tile_indices) - 1:
                                await asyncio.sleep(random.uniform(0.18, 0.38))

                        await asyncio.sleep(random.uniform(0.4, 0.7))

                        print("Все выбранные статические плитки обработаны. Проверяем результат раунда...")
                        await self.mouse.move_to_element_naturally(verify_button.locator)
                        await asyncio.sleep(random.uniform(0.15, 0.30))
                        await self._make_click(verify_button, fast=True)

                        # 🌟 FIXED CRITICAL VERIFICATION CHECK LIFT
                        is_solved = False
                        for _ in range(8):
                            await asyncio.sleep(0.4)
                            try:
                                if await anchor_checkbox.getAttribute("aria-checked") == "true":
                                    is_solved = True
                                    break
                            except Exception:
                                pass
                            try:
                                if not await target_area.isVisible():
                                    is_solved = True
                                    break
                            except Exception:
                                is_solved = True
                                break

                        if is_solved:
                            print("Капча успешно пройдена!")
                            for _ in range(15):
                                token = await self._get_g_token(document)
                                if token: return token
                                await asyncio.sleep(0.5)
                            raise Exception("Captcha solved but token retrieval timed out")

                        print("Появилась новая задача (новый раунд картинок)...")
                        await asyncio.sleep(random.uniform(0.8, 1.3))
                        continue

                except Exception as e:
                    if "session is null" in str(e) or "detached" in str(e):
                        print(
                            "Предупреждение: Фрейм перегрузился во время взаимодействия. Мягкая перезагрузка раунда...")
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        raise e

        except Exception as e:
            print(f"Критическая ошибка: {e}")
            try:
                await self.page.screenshot(path=screenshot_error_path)
            except Exception:
                pass
            return await self._get_g_token(DOM(self.page))

    async def _check_tile_ready(self, tile, old_src):
        """Helper to check if a tile has finished its refresh animation."""
        try:
            img = tile.locator.locator('img')
            current_src = await img.get_attribute('src')
            opacity = await img.evaluate("el => window.getComputedStyle(el).opacity")
            return (current_src != old_src) and (float(opacity) >= 0.9)
        except:
            return False

    async def get_g_captcha_dataset(self) -> str | None:
        run_id = int(time.time() * 1000)
        print(f"[{run_id}] Starting reCAPTCHA verification track...")

        screenshots_path = Path("tasks")
        screenshots_path.mkdir(exist_ok=True)

        try:
            # 1. Locate and resolve the initial anchor iframe checkbox
            print("Locating reCAPTCHA checkbox frame...")
            anchor_frame = None
            for frame in self.page.frames:
                if "anchor" in frame.url or "recaptcha" in frame.name.lower():
                    anchor_frame = frame
                    break

            if not anchor_frame:
                print("❌ Could not resolve anchor frame context.")
                return None

            anchor_checkbox = await anchor_frame.wait_for_selector('#recaptcha-anchor', state="visible", timeout=10000)

            # Check if already solved via passive cookies/reputation profile
            if await anchor_checkbox.get_attribute("aria-checked") == "true":
                print("✅ Captcha passively pre-solved via browser profile reputation!")
                return await self.page.evaluate('() => document.getElementById("g-recaptcha-response")?.value')

            print("Clicking reCAPTCHA checkbox via interaction framework...")
            await self.mouse.click(anchor_checkbox)
            await asyncio.sleep(1.5)

            # 2. Transition into the Challenge Matrix Loop
            while True:
                # Direct break check: Did our previous step solve it completely?
                g_token = await self.page.evaluate('() => document.getElementById("g-recaptcha-response")?.value')
                if g_token and len(g_token) > 10:
                    print(f"🎉 Success! Token extracted: {g_token[:30]}...")
                    return g_token

                # Check if the puzzle visual wrapper is active
                challenge_frame = None
                for frame in self.page.frames:
                    if "bframe" in frame.url or "challenge" in frame.url:
                        challenge_frame = frame
                        break

                if not challenge_frame:
                    # If no challenge frame exists and token is empty, double check frame stability
                    await asyncio.sleep(1.0)
                    continue

                # Detect the puzzle text instructions
                try:
                    instruction_el = await challenge_frame.wait_for_selector('.rc-imageselect-instructions',
                                                                             timeout=4000)
                    full_text = (await instruction_el.inner_text()).lower().replace("\n", " ")
                except Exception:
                    print("Puzzle element transition state. Retrying loop check...")
                    await asyncio.sleep(0.5)
                    continue

                disappearing_mode = "none left" in full_text
                current_mode = "classify" if any(p in full_text for p in ["none left", "images"]) else "detect"
                print(f"[{run_id}] Detected puzzle track mode: {current_mode} (Disappearing={disappearing_mode})")

                # Capture matrix image target for your loaded YOLO weights
                image_target = await challenge_frame.wait_for_selector('#rc-imageselect-target')
                screenshot_file = screenshots_path / f"captcha_{run_id}_{secrets.token_hex(4)}.png"
                await image_target.screenshot(path=str(screenshot_file))

                cells = await challenge_frame.query_selector_all('.rc-imageselect-tile')
                grid_size = 4 if len(cells) == 16 else 3
                print(f"Grid detected: {grid_size}x{grid_size} ({len(cells)} active tiles)")

                # --- 🎯 INTEGRATION POINT FOR YOUR ML MODELS ---
                # Pass your screenshot path to your loaded YOLO architecture here
                # target_tiles = self.model_registry.predict(screenshot_file, current_mode)
                # Placeholder fallback simulating a list of target indexes (e.g., [0, 2, 5])
                target_tiles = random.sample(range(len(cells)), 2)

                if not target_tiles:
                    print("🔍 Model detected no targets. Triggering reload pattern...")
                    reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await self.mouse.click(reload_btn)
                    await asyncio.sleep(1.5)
                    continue

                # 3. Handle Tile Interaction Phases
                verify_button = await challenge_frame.query_selector('#recaptcha-verify-button')

                if disappearing_mode:
                    for index in target_tiles:
                        if index >= len(cells): continue
                        tile = cells[index]

                        old_html = await tile.inner_html()
                        await self.mouse.click(tile)

                        # Fine-tuned frame verification loop to wait for the new image slot to load
                        for _ in range(10):
                            await asyncio.sleep(0.1)
                            if await tile.inner_html() != old_html:
                                break
                        await asyncio.sleep(0.2)

                    print("Settle wait window for dynamic asset loading...")
                    await asyncio.sleep(2.0)

                else:
                    # Standard static detection grid selection matrix
                    for index in target_tiles:
                        if index >= len(cells): continue
                        await self.mouse.click(cells[index])
                        await asyncio.sleep(random.uniform(0.2, 0.4))

                # 4. Final Verification Dispatch Phase
                print("Dispatching resolution sequence to verification framework...")
                await self.mouse.click(verify_button)
                await asyncio.sleep(2.5)  # Let reCAPTCHA process network confirmation telemetry

        except Exception as e:
            print(f"❌ Critical exception fault in execution pipeline: {e}")
            await self.page.screenshot(path="tasks/debug_error_dump.png")
            # Final safety lookup attempt to grab the token if it passed in the background
            return await self.page.evaluate('() => document.getElementById("g-recaptcha-response")?.value')

    async def solve_interstitial_cf_captcha(self, timeout: int = 15, click: bool = True) -> Optional[str]:
        """
        Checks if the browser is stuck on a Cloudflare 'Just a moment' clearance page
        and resolves it before the target page loads. Returns the token string or bypass token on success.
        """
        page_title = await self.page.title()
        is_interstitial = "just a moment" in page_title.lower() or "cloudflare" in page_title.lower()

        if not is_interstitial:
            print("ℹ️ [Stage 1] Skipped. No initial blocking interstitial page encountered.")
            return "skipped"

        print("⚠️ [Stage 1] Cloudflare Interstitial Page detected. Bypassing protection gate...")
        stage1_token = await self._execute_frame_solve(timeout=timeout, click=click, check_title_fallback=True)

        if not stage1_token:
            print("❌ [Stage 1] Failed to clear initial Cloudflare intercept screen.")
            return None

        print("⏳ [Stage 1] Passed! Waiting for redirect to settle...")
        try:
            # Enforce a hard block until the URL shifts completely away from the challenge layout
            await self.page.wait_for_function(
                "() => !window.location.href.includes('challenges.cloudflare.com') && !document.title.toLowerCase().includes('just a moment')",
                timeout=8000
            )
            await self.page.wait_for_load_state("domcontentloaded", timeout=4000)
        except Exception:
            pass

        await asyncio.sleep(1.0)  # Quick landing buffer for target DOM element population
        return stage1_token

    # --- Function 2: Handle Embedded Form Captcha ---
    async def solve_embedded_cf_captcha(self, timeout: int = 6, click: bool = True) -> Optional[str]:
        """
        Scans the current target page for an embedded Turnstile widget (like the IMEI form check)
        and resolves it natively using ShyMouse. Returns the real token string on success.
        """
        print("🔍 [Stage 2] Scanning page for embedded target form Turnstile widget...")

        has_widget = False
        start_scan = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_scan < 4.0:
            for frame in self.page.frames:
                if "challenges.cloudflare.com" in frame.url:
                    has_widget = True
                    break
            if has_widget:
                break
            await asyncio.sleep(0.5)

        if not has_widget:
            print("✅ [Stage 2] No embedded form captcha present. Execution track clear!")
            return "skipped"

        print("⚠️ [Stage 2] Embedded Form Turnstile found. Simulating completion...")
        return await self._execute_frame_solve(timeout=timeout, click=click, check_title_fallback=False)

    async def _execute_frame_solve(self, timeout: int, click: bool, check_title_fallback: bool = False) -> Optional[str]:
        """
        Internal execution driver that isolates frames, stabilizes changing page layouts,
        clicks via ShyMouse, and handles concurrent background organic mouse wandering
        while waiting for token resolution signatures.

        Returns the token string on success, or None on failure.
        """
        target_frame = None
        start_time = asyncio.get_event_loop().time()

        # 1. Locate the iframe context
        while asyncio.get_event_loop().time() - start_time < timeout:
            for frame in self.page.frames:
                try:
                    url = frame.url
                    title = await frame.title()
                    if "challenges.cloudflare.com" in url or "turnstile" in url.lower() or "turnstile" in title.lower():
                        target_frame = frame
                        break
                except Exception:
                    continue
            if target_frame:
                break
            await asyncio.sleep(0.4)

        if not target_frame:
            print("❌ Target verification frame context missing from layout registry.")
            return None

        if click:
            try:
                frame_element = await target_frame.frame_element()
                if not frame_element:
                    return None

                # 2. Bring it into the viewport view track
                await frame_element.scroll_into_view_if_needed()
                await self.mouse.organic_idle(0.2)

                # 3. LAYOUT STABILIZATION ENGINE
                print("⏳ Waiting for Turnstile internal DOM components to mount...")
                try:
                    await target_frame.wait_for_selector("#challenge-stage", state="visible", timeout=4000)
                except Exception:
                    try:
                        await target_frame.wait_for_selector("body", state="visible", timeout=2000)
                    except Exception:
                        pass

                print("⏳ Waiting for page layout and ad wrappers to stabilize...")
                stable_box = None
                stable_retries = 0

                while stable_retries < 15:
                    current_box = await frame_element.bounding_box()
                    if current_box and current_box["width"] > 10 and current_box["height"] > 10:
                        if stable_box and current_box["x"] == stable_box["x"] and current_box["y"] == stable_box["y"]:
                            stable_box = current_box
                            break
                        stable_box = current_box
                    stable_retries += 1
                    await self.mouse.organic_idle(0.2)

                if not stable_box:
                    print("❌ Found the frame element, but it has no physical dimensions layout.")
                    return None

                # 4. TARGET KINEMATICS & STRIKE ENGINE
                target_x = stable_box["x"] + random.uniform(25.0, 37.0)
                target_y = stable_box["y"] + (stable_box["height"] * 0.5) + random.uniform(-5.0, 5.0)

                print(
                    f"🎯 Target stabilized. Driving Bézier path precisely to checkbox at: ({target_x:.1f}, {target_y:.1f})")
                await asyncio.sleep(random.uniform(0.15, 0.25))

                # Execute path movement to targeted coordinates
                await self.mouse.move_to_position(target_x, target_y)
                await self.page.mouse.move(target_x, target_y)  # Precision adjustment snap

                # Fire the click down/up timeline
                await self.page.mouse.down()
                await self.mouse.organic_idle(random.uniform(0.07, 0.14))
                await self.page.mouse.up()

                if hasattr(self.mouse, "update_action_count"):
                    self.mouse.update_action_count()

            except Exception as e:
                print(f"[-] Interaction framework event fault: {e}")
                return None

            # 5. CONCURRENT REAL-TOKEN VALIDATION LOGIC
            print("Checking live token registration status...")
            # stop_wandering = asyncio.Event()
            #
            # # Pass frame_element context to wander locally while waiting for token resolution
            # wander_task = asyncio.create_task(
            #     self.mouse.wander_randomly(frame_element, stop_wandering)
            # )

            resolved_token = None
            try:
                # 🛠️ wait_for_function returns a JSHandle containing the value string once it matches criteria
                token_handle = await self.page.wait_for_function(
                    """() => {
                        const el = document.querySelector('[name="cf-turnstile-response"]');
                        return (el && el.value && el.value.length > 10) ? el.value : null;
                    }""",
                    timeout=10000
                )
                # Unwrap the string primitive out of the browser JS execution sandbox
                resolved_token = await token_handle.json_value()
                print("✅ Challenge signature verified via real host page token registry.")

            except Exception as e:
                # Fall back to checking the main page title only if the token evaluation fails/times out
                if check_title_fallback:
                    try:
                        title = await self.page.title()
                        title = title.lower()
                        if "just a moment" not in title and "cloudflare" not in title and title != "":
                            print("✅ Challenge verified via page title transition fallback.")
                            try:
                                resolved_token = await self.page.eval_on_selector('[name="cf-turnstile-response"]',
                                                                                  "el => el.value")
                            except Exception:
                                pass
                            if not resolved_token:
                                resolved_token = "interstitial_passed"
                    except Exception:
                        pass

                if not resolved_token:
                    if any(k in str(e) for k in ["navigated", "destroyed", "detached"]):
                        print("✅ Context transitioned instantly. Treat as verified.")
                        resolved_token = "interstitial_passed"
                    else:
                        print("⚠️ Verification signature timed out.")
                        resolved_token = None

            # finally:
            #     stop_wandering.set()
            #     await wander_task

            return resolved_token

        else:
            # Non-click / Keyboard interaction tracking fallback
            try:
                await target_frame.click("body")
                await self.mouse.organic_idle(random.uniform(0.3, 0.5))
                await self.page.keyboard.press("Tab")
                await self.mouse.organic_idle(random.uniform(0.2, 0.4))
                await self.page.keyboard.press("Space")

                await asyncio.sleep(1.0)
                try:
                    token_value = await self.page.eval_on_selector('[name="cf-turnstile-response"]', "el => el.value")
                    if token_value and len(token_value) > 10:
                        return token_value
                except Exception:
                    pass
                return "passed"
            except Exception:
                return None