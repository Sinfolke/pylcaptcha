import asyncio
import math
import random
import time
from logging import config
from typing import Any, Dict, List, Optional


class ShyMouse:
    def __init__(self, page, options: Optional[Dict[str, Any]] = None):
        options = options or {}

        self.page = page
        self.last_pos: Optional[Dict[str, float]] = None
        self.last_move_time = self._now_ms()
        self.move_history: List[Dict[str, float]] = []
        self.max_history_length = 50
        self.cached_viewport: Optional[Dict[str, Any]] = None
        self.viewport_cache_time = 0
        self.viewport_cache_duration = 2000

        self.motion_state = {
            "lastVelocity": {"x": 0.0, "y": 0.0},
            "lastAcceleration": {"x": 0.0, "y": 0.0},
            "lastJerk": {"x": 0.0, "y": 0.0},
            "temporalCorrelation": 0.5,
            "entropyAccumulator": 0.0,
            "perlinSeed": random.random() * 10000,
            "pollingPhase": random.random(),
        }

        self.config = {
            "fatigueEnabled": options.get("fatigueEnabled", False),
            "fatigueThreshold": options.get("fatigueThreshold", 20),
            "actionCount": 0,
            "maxFatigue": options.get("maxFatigue", 100),
            "fatigueMultiplier": 1.0,
            "attentionSpan": 0.88 + random.random() * 0.10,
            "minAttentionSpan": 0.80,
            "baseReactionTime": options.get("baseReactionTime", 200),
            "reactionTimeVariance": options.get("reactionTimeVariance", 80),
            "curveComplexity": options.get("curveComplexity", "low"),
            "debug": options.get("debug", False),
            "hesitationProbability": 0.08,
            "microCorrectionFrequency": 0.15,
            "targetDriftEnabled": True,
            "minPollingInterval": 6.9,
            "maxPollingInterval": 16.6,
            "typicalPollingInterval": 15,
            "fittsA": 0.100,
            "fittsB": 0.050,
            "fractalDepth": 3,
            "entropyTarget": 0.65,
            "jerkSmoothness": 0.85,
        }

        self.setup_navigation_listener()

    # -----------------------------
    # Infra / helpers
    # -----------------------------
    async def _precise_async_sleep(self, duration_ms: float) -> None:
        """Sleeps with sub-millisecond accuracy by mixing asyncio and spin-waiting."""
        duration_sec = duration_ms / 1000.0
        start_time = time.perf_counter()

        # 1. Coarse Sleep Phase
        # Standard OS sleep quantization is ~15.6ms. Only use asyncio.sleep if
        # the duration is long enough, leaving a 10ms safety buffer to spin-wait.
        if duration_sec > 0.015:
            await asyncio.sleep(duration_sec - 0.010)

        # 2. Fine Spin-Wait Phase
        # Tight CPU lock for the remaining fraction of a millisecond to hit perfect timing.
        while (time.perf_counter() - start_time) < duration_sec:
            pass  # Intentionally blocking to guarantee high precision
    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def organic_idle(self, duration_sec: float):
        """
        Simulates human physiological tremor and lazy drift while waiting.
        Replaces dead static sleeps with continuous organic micro-movements.
        """
        if not self.last_pos:
            return

        start_time = time.perf_counter()
        accumulator = random.random() * 100

        while (time.perf_counter() - start_time) < duration_sec:
            # 1. High-frequency physiological tremor (8-12Hz muscle vibration)
            tremor_x = random.gauss(0, 0.22)
            tremor_y = random.gauss(0, 0.22)

            # 2. Slow, lazy cognitive drift (wandering thoughts/hand relaxation)
            accumulator += 0.05
            drift_x = self.perlin_noise(accumulator, 0, self.motion_state["perlinSeed"]) * 0.4
            drift_y = self.perlin_noise(0, accumulator, self.motion_state["perlinSeed"] + 5) * 0.4

            # Combine current position with minor noise
            current_x = self.last_pos["x"] + tremor_x + drift_x
            current_y = self.last_pos["y"] + tremor_y + drift_y

            # Use viewport or fallback boundaries to prevent clipping out of bounds
            await self.page.mouse.move(current_x, current_y)
            self.last_pos = {"x": current_x, "y": current_y}

            # Fast update rate (roughly 60Hz to match monitor refresh/human hand updates)
            await self.page.evaluate("() => new Promise(requestAnimationFrame)")
    def log(self, *args):
        if self.config["debug"]:
            ts = time.strftime("%H:%M:%S", time.localtime())
            print("[ShyMouse]", ts, *args)

    def setup_navigation_listener(self):
        try:
            self.page.on("framenavigated", lambda frame: self.invalidate_viewport_cache())
        except Exception as e:
            self.log("Navigation listener failed:", str(e))

    def invalidate_viewport_cache(self):
        self.cached_viewport = None
        self.viewport_cache_time = 0

    async def random_delay(self, min_ms: float, max_ms: float):
        micro_var = (random.random() - 0.5) * 10
        delay = min_ms + random.random() * (max_ms - min_ms) + micro_var
        await asyncio.sleep(max(0.0, delay) / 1000.0)

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    def calculate_distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        return math.sqrt(dx * dx + dy * dy)

    def random_gaussian(self, mean: float = 0, std_dev: float = 1) -> float:
        return random.gauss(mean, std_dev)

    def initialize_position(self, viewport: Dict[str, Any]):
        margin = 120
        x = margin + (random.random() ** 1.3) * (viewport["width"] - 2 * margin)
        y = margin + random.random() * (viewport["height"] - 2 * margin)
        self.last_pos = {"x": x, "y": y}
        self.last_move_time = self._now_ms()
        self.log("Position initialized:", self.last_pos)

    async def get_viewport(self, retries: int = 2) -> Dict[str, Any]:
        now = self._now_ms()

        if self.cached_viewport and (now - self.viewport_cache_time) < self.viewport_cache_duration:
            return self.cached_viewport

        for attempt in range(retries + 1):
            try:
                viewport_info = await self.page.evaluate(
                    """
                    () => {
                        try {
                            return {
                                width: window.innerWidth,
                                height: window.innerHeight,
                                scrollX: window.scrollX || window.pageXOffset || 0,
                                scrollY: window.scrollY || window.pageYOffset || 0,
                                devicePixelRatio: window.devicePixelRatio || 1,
                                documentWidth: Math.max(
                                    document.documentElement.scrollWidth || 0,
                                    document.documentElement.offsetWidth || 0,
                                    document.documentElement.clientWidth || 0,
                                    document.body?.scrollWidth || 0,
                                    document.body?.offsetWidth || 0
                                ),
                                documentHeight: Math.max(
                                    document.documentElement.scrollHeight || 0,
                                    document.documentElement.offsetHeight || 0,
                                    document.documentElement.clientHeight || 0,
                                    document.body?.scrollHeight || 0,
                                    document.body?.offsetHeight || 0
                                ),
                            };
                        } catch (e) {
                            return null;
                        }
                    }
                    """
                )
                if viewport_info:
                    self.cached_viewport = viewport_info
                    self.viewport_cache_time = now
                    return viewport_info

                if attempt < retries:
                    await self.random_delay(50, 100)

            except Exception as e:
                self.log(f"get_viewport attempt {attempt + 1} failed:", str(e))
                if attempt < retries:
                    await self.random_delay(100, 200)

        fallback = {
            "width": 1920,
            "height": 1080,
            "scrollX": 0,
            "scrollY": 0,
            "devicePixelRatio": 1,
            "documentWidth": 1920,
            "documentHeight": 1080,
        }
        self.cached_viewport = fallback
        self.viewport_cache_time = now - (self.viewport_cache_duration - 500)
        return fallback

    async def get_current_scroll_y(self) -> float:
        try:
            return await self.page.evaluate(
                "() => window.scrollY || window.pageYOffset || 0"
            )
        except Exception:
            return 0.0

    async def get_element_frame(self, element):
        try:
            frame = await element.owner_frame()
            return frame or self.page.main_frame
        except Exception as e:
            self.log("get_element_frame failed:", str(e))
            return self.page.main_frame

    async def get_element_bounding_box(self, element, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                box = await element.bounding_box()
                if box and box["width"] > 0 and box["height"] > 0:
                    return box
                if attempt < max_retries - 1:
                    await self.random_delay(50, 150)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log(f"Failed to get bounding box after {max_retries} attempts:", str(e))
                    return None
                await self.random_delay(100, 200)
        return None

    async def get_scroll_container(self, element):
        try:
            container_info = await element.evaluate(
                """
                (el) => {
                    try {
                        let parent = el.parentElement;
                        let depth = 0;

                        while (parent && parent !== document.documentElement && depth < 50) {
                            const style = window.getComputedStyle(parent);
                            const overflow = style.overflow + style.overflowY + style.overflowX;

                            if (/(auto|scroll)/.test(overflow)) {
                                const rect = parent.getBoundingClientRect();
                                return {
                                    isWindow: false,
                                    scrollTop: parent.scrollTop,
                                    scrollLeft: parent.scrollLeft,
                                    scrollHeight: parent.scrollHeight,
                                    scrollWidth: parent.scrollWidth,
                                    clientHeight: parent.clientHeight,
                                    clientWidth: parent.clientWidth,
                                    rectTop: rect.top,
                                    rectLeft: rect.left,
                                    rectWidth: rect.width,
                                    rectHeight: rect.height,
                                };
                            }

                            parent = parent.parentElement;
                            depth++;
                        }

                        return null;
                    } catch (e) {
                        return null;
                    }
                }
                """
            )

            if container_info:
                return {"info": container_info, "is_window": False}

            viewport = await self.get_viewport()
            return {
                "info": {
                    "isWindow": True,
                    "scrollTop": viewport["scrollY"],
                    "scrollLeft": viewport["scrollX"],
                    "scrollHeight": viewport["documentHeight"],
                    "scrollWidth": viewport["documentWidth"],
                    "clientHeight": viewport["height"],
                    "clientWidth": viewport["width"],
                },
                "is_window": True,
            }
        except Exception as e:
            self.log("get_scroll_container failed:", str(e))
            viewport = await self.get_viewport()
            return {
                "info": {
                    "isWindow": True,
                    "scrollTop": viewport["scrollY"],
                    "scrollLeft": viewport["scrollX"],
                    "scrollHeight": viewport["documentHeight"],
                    "scrollWidth": viewport["documentWidth"],
                    "clientHeight": viewport["height"],
                    "clientWidth": viewport["width"],
                },
                "is_window": True,
            }

    # -----------------------------
    # Clickability / viewport
    # -----------------------------

    async def is_element_clickable(self, element) -> bool:
        try:
            return await element.evaluate(
                """
                (el) => {
                    try {
                        if (!el.isConnected) return false;

                        const style = window.getComputedStyle(el);
                        if (style.display === 'none') return false;
                        if (style.visibility === 'hidden') return false;
                        if (parseFloat(style.opacity) < 0.1) return false;
                        if (style.pointerEvents === 'none') return false;
                        if (el.disabled) return false;

                        const rect = el.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0) return false;
                        if (rect.bottom < 0 || rect.right < 0) return false;
                        if (rect.top > window.innerHeight || rect.left > window.innerWidth) return false;

                        let ancestor = el.parentElement;
                        while (ancestor && ancestor !== document.body) {
                            const ancestorStyle = window.getComputedStyle(ancestor);
                            if (ancestorStyle.pointerEvents === 'none') return false;
                            ancestor = ancestor.parentElement;
                        }

                        const samplingPoints = [
                            { x: 0.5, y: 0.5 },
                            { x: 0.3, y: 0.5 },
                            { x: 0.7, y: 0.5 },
                            { x: 0.5, y: 0.3 },
                            { x: 0.5, y: 0.7 },
                            { x: 0.3, y: 0.3 },
                            { x: 0.7, y: 0.3 },
                            { x: 0.3, y: 0.7 },
                            { x: 0.7, y: 0.7 },
                        ];

                        let clickablePoints = 0;

                        for (const point of samplingPoints) {
                            const x = rect.left + rect.width * point.x;
                            const y = rect.top + rect.height * point.y;

                            const topElement = document.elementFromPoint(x, y);

                            if (topElement) {
                                if (topElement === el || el.contains(topElement)) {
                                    clickablePoints++;
                                } else {
                                    let current = topElement;
                                    while (current && current !== document.body) {
                                        if (current === el) {
                                            clickablePoints++;
                                            break;
                                        }
                                        current = current.parentElement;
                                    }
                                }
                            }
                        }

                        return clickablePoints >= samplingPoints.length * 0.5;
                    } catch (e) {
                        return false;
                    }
                }
                """
            )
        except Exception as e:
            self.log("is_element_clickable failed:", str(e))
            return False

    async def is_element_in_viewport(self, element, buffer: int = 10) -> bool:
        try:
            box = await self.get_element_bounding_box(element)
            if not box:
                return False

            scroll_container = await self.get_scroll_container(element)
            viewport = await self.get_viewport()

            if scroll_container["info"]["isWindow"]:
                view_top = viewport["scrollY"] - buffer
                view_bottom = viewport["scrollY"] + viewport["height"] + buffer
                view_left = viewport["scrollX"] - buffer
                view_right = viewport["scrollX"] + viewport["width"] + buffer

                has_vertical_overlap = not (box["y"] + box["height"] < view_top or box["y"] > view_bottom)
                has_horizontal_overlap = not (box["x"] + box["width"] < view_left or box["x"] > view_right)
                return has_vertical_overlap and has_horizontal_overlap

            return await element.evaluate(
                """
                ([el, buff]) => {
                    try {
                        let parent = el.parentElement;
                        while (parent && parent !== document.documentElement) {
                            const style = window.getComputedStyle(parent);
                            const overflow = style.overflow + style.overflowY + style.overflowX;

                            if (/(auto|scroll)/.test(overflow)) {
                                const parentRect = parent.getBoundingClientRect();
                                const elRect = el.getBoundingClientRect();

                                const hasVerticalOverlap = !(elRect.bottom < parentRect.top - buff || elRect.top > parentRect.bottom + buff);
                                const hasHorizontalOverlap = !(elRect.right < parentRect.left - buff || elRect.left > parentRect.right + buff);
                                return hasVerticalOverlap && hasHorizontalOverlap;
                            }
                            parent = parent.parentElement;
                        }
                        return true;
                    } catch (e) {
                        return false;
                    }
                }
                """,
                [element, buffer]
            )
        except Exception as e:
            self.log("is_element_in_viewport failed:", str(e))
            return False

    async def wait_for_element_stability(self, element, timeout: int = 1500):
        start_time = self._now_ms()

        try:
            has_animations = await element.evaluate(
                """
                (el) => {
                    try {
                        const style = window.getComputedStyle(el);
                        const hasTransition = style.transition !== 'all 0s ease 0s' && style.transition !== 'none';
                        const hasAnimation = style.animation !== 'none';
                        return hasTransition || hasAnimation;
                    } catch (e) {
                        return false;
                    }
                }
                """
            )
            if has_animations:
                await self.random_delay(300, 500)
        except Exception:
            pass

        last_box = None
        stable_count = 0
        required_stable_checks = 3

        while self._now_ms() - start_time < timeout:
            try:
                box = await element.bounding_box()
                if not box:
                    await self.random_delay(50, 100)
                    continue

                if last_box:
                    x_diff = abs(box["x"] - last_box["x"])
                    y_diff = abs(box["y"] - last_box["y"])
                    w_diff = abs(box["width"] - last_box["width"])
                    h_diff = abs(box["height"] - last_box["height"])

                    if x_diff < 1 and y_diff < 1 and w_diff < 1 and h_diff < 1:
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            return box
                    else:
                        stable_count = 0

                last_box = box
                await self.random_delay(50, 100)

            except Exception:
                await self.random_delay(100, 200)

        return last_box

    # -----------------------------
    # Fatigue / stats
    # -----------------------------

    def apply_fatigue(self, base_value: int) -> int:
        if not self.config["fatigueEnabled"]:
            return base_value

        if self.config["actionCount"] > self.config["maxFatigue"]:
            self.config["actionCount"] = math.floor(self.config["fatigueThreshold"] * 0.8)
            self.config["attentionSpan"] = min(0.96, self.config["attentionSpan"] + 0.08)
            self.config["fatigueMultiplier"] = 1.0
            self.log("Fatigue reset")

        if self.config["actionCount"] > self.config["fatigueThreshold"]:
            excess = self.config["actionCount"] - self.config["fatigueThreshold"]
            fatigue_level = excess / self.config["fatigueThreshold"]
            self.config["fatigueMultiplier"] = 1.0 + fatigue_level * 0.4
            return round(base_value * min(1 + fatigue_level * 0.018, 1.45))

        return base_value

    def update_action_count(self):
        self.config["actionCount"] += 1

        if self.config["actionCount"] % 45 == 0:
            recovery = math.floor(15 + random.random() * 10)
            self.config["actionCount"] = max(0, self.config["actionCount"] - recovery)
            self.config["attentionSpan"] = min(0.96, self.config["attentionSpan"] + 0.04)
            self.config["fatigueMultiplier"] = max(1.0, self.config["fatigueMultiplier"] * 0.85)
            self.log("Recovery applied")

        self.config["attentionSpan"] = max(
            self.config["minAttentionSpan"],
            self.config["attentionSpan"] - 0.0008
        )

    def add_to_history(self, position: Dict[str, float]):
        self.move_history.append(position)
        if len(self.move_history) > self.max_history_length:
            self.move_history.pop(0)

    def get_movement_stats(self):
        if len(self.move_history) < 2:
            return None

        distances = []
        time_diffs = []

        for i in range(1, len(self.move_history)):
            dist = self.calculate_distance(self.move_history[i - 1], self.move_history[i])
            time_diff = self.move_history[i]["time"] - self.move_history[i - 1]["time"]
            distances.append(dist)
            time_diffs.append(time_diff)

        avg_distance = sum(distances) / len(distances)
        avg_time = sum(time_diffs) / len(time_diffs)

        return {
            "averageDistance": avg_distance,
            "averageTime": avg_time,
            "averageSpeed": avg_distance / avg_time if avg_time else 0,
            "totalMoves": len(self.move_history),
            "actionCount": self.config["actionCount"],
            "attentionSpan": self.config["attentionSpan"],
            "fatigueLevel": max(0, self.config["actionCount"] - self.config["fatigueThreshold"]),
            "fatigueMultiplier": self.config["fatigueMultiplier"],
        }

    async def human_reaction_delay(self):
        base_time = self.config["baseReactionTime"]
        variance = self.config["reactionTimeVariance"]
        attention_factor = 1 + (1 - self.config["attentionSpan"]) * 0.6
        fatigue_factor = self.config["fatigueMultiplier"]
        reaction_time = max(85, self.random_gaussian(base_time * attention_factor * fatigue_factor, variance))
        await self.random_delay(reaction_time * 0.75, reaction_time * 1.25)

    # -----------------------------
    # Noise / physics
    # -----------------------------
    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)
    @staticmethod
    def grad(h, x_, y_):
        v = x_ if (int(h) & 1) == 0 else y_
        return -v if (int(h) & 2) == 0 else v
    def perlin_noise(self, x: float, y: float, seed: float) -> float:
        def hash_fn(n):
            n = math.sin(n + seed) * 43758.5453123
            return n - math.floor(n)

        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi


        a = hash_fn(xi + hash_fn(yi))
        b = hash_fn(xi + 1 + hash_fn(yi))
        c = hash_fn(xi + hash_fn(yi + 1))
        d = hash_fn(xi + 1 + hash_fn(yi + 1))

        u = self.fade(xf)
        v = self.fade(yf)

        x1 = self.lerp(self.grad(a * 255, xf, yf), self.grad(b * 255, xf - 1, yf), u)
        x2 = self.lerp(self.grad(c * 255, xf, yf - 1), self.grad(d * 255, xf - 1, yf - 1), u)

        return self.lerp(x1, x2, v)

    def calculate_entropy(self, points: List[Dict[str, float]]) -> float:
        if len(points) < 3:
            return 0.5

        velocities = []
        for i in range(1, len(points)):
            dx = points[i]["x"] - points[i - 1]["x"]
            dy = points[i]["y"] - points[i - 1]["y"]
            velocities.append(math.sqrt(dx * dx + dy * dy))

        mean = sum(velocities) / len(velocities)
        variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
        entropy = math.log2(1 + variance / (mean + 1))
        return min(1.0, entropy / 3)

    def calculate_smooth_jerk(self, prev_jerk, target_jerk):
        smoothness = self.config["jerkSmoothness"]
        return {
            "x": prev_jerk["x"] * smoothness + target_jerk["x"] * (1 - smoothness),
            "y": prev_jerk["y"] * smoothness + target_jerk["y"] * (1 - smoothness),
        }

    def ease_in_out_cubic(self, t: float) -> float:
        variance = (random.random() - 0.5) * 0.018
        t = self.clamp(t + variance, 0, 1)
        return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2

    def multi_layer_easing(self, t: float, distance: float) -> float:
        eased = 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2

        micro_variation = (random.random() - 0.5) * 0.02
        fractal_variation = self.perlin_noise(t * 5, distance * 0.01, self.motion_state["perlinSeed"]) * 0.015
        eased += micro_variation + fractal_variation

        tremor_phase = time.time() * 10 + t * math.pi * 8
        tremor = math.sin(tremor_phase) * 0.008 * self.motion_state["temporalCorrelation"]
        eased += tremor

        current_entropy = self.motion_state["entropyAccumulator"] % 1
        lapse_prob = (1 - self.config["attentionSpan"]) * (1 + current_entropy) * 0.1
        if random.random() < lapse_prob:
            eased += self.random_gaussian(0, 0.025)

        ID = math.log2(distance / 100 + 1)
        hesitation_prob = self.config.get('hesitationProbability') * (ID / 5)
        if distance > 500 and 0.35 < t < 0.65 and random.random() < hesitation_prob:
            eased *= 0.92

        if t > 0.8:
            eased += self.random_gaussian(0, 0.008 * self.config["fatigueMultiplier"])

        return self.clamp(eased, 0, 1)

    def get_bezier_point(self, t, p0, p1, p2, p3):
        omt = 1 - t
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        return {
            "x": p0["x"] * omt3 + 3 * p1["x"] * omt2 * t + 3 * p2["x"] * omt * t2 + p3["x"] * t3,
            "y": p0["y"] * omt3 + 3 * p1["y"] * omt2 * t + 3 * p2["y"] * omt * t2 + p3["y"] * t3,
        }

    def calculate_realistic_polling_delay(self, phase: float, velocity_factor: float = 1.0) -> float:
        correlation = self.motion_state["temporalCorrelation"]
        polling_phase = self.motion_state["pollingPhase"]

        correlated_random = random.random() * (1 - correlation) + polling_phase * correlation
        self.motion_state["pollingPhase"] = correlated_random

        if correlated_random < 0.65:
            base_delay = self.config["typicalPollingInterval"] + self.random_gaussian(0, 1.5)
        elif correlated_random < 0.82:
            base_delay = self.config["minPollingInterval"] + random.random() * 1.6
        else:
            base_delay = 11.8 + random.random() * 4.8

        if 0.3 < phase < 0.7:
            base_delay *= 0.88 * velocity_factor
        elif phase > 0.85:
            base_delay *= 1.25
        elif phase < 0.15:
            base_delay *= 0.95 + random.random() * 0.15

        # Locate in calculate_realistic_polling_delay:
        entropy_noise = self.perlin_noise(
            time.time() * 10,
            self.motion_state["entropyAccumulator"],
            self.motion_state["perlinSeed"]
        )
        base_delay += entropy_noise * (1.2 * self.config["entropyTarget"])

        # Feed back actual path behavior to adjust the accumulator
        if len(self.move_history) > 5:
            current_calculated_entropy = self.calculate_entropy(self.move_history[-5:])
            # Adjust accumulator based on distance from target
            entropy_error = self.config["entropyTarget"] - current_calculated_entropy
            self.motion_state["entropyAccumulator"] += entropy_error * 0.05
        else:
            self.motion_state["entropyAccumulator"] += 0.05

        return self.clamp(
            base_delay,
            self.config["minPollingInterval"],
            self.config["maxPollingInterval"]
        )

    def generate_velocity_profile(self, num_points: int, distance: float):
        profile = []
        peak_position = 0.40 + random.random() * 0.15

        for i in range(num_points):
            t = i / max(1, num_points)

            if t < peak_position:
                norm_t = t / peak_position if peak_position else 0
                velocity = math.exp(-(((norm_t - 1) * 2.2) ** 2))
            else:
                norm_t = (t - peak_position) / (1 - peak_position) if peak_position < 1 else 0
                velocity = math.exp(-((norm_t * 2.8) ** 2))

            noise_variation = self.perlin_noise(i * 0.1, 0, self.motion_state["perlinSeed"] + 100)
            velocity *= (1 + noise_variation * 0.15)
            velocity = max(0.1, velocity)
            profile.append(velocity)

        return profile

    def calculate_realistic_control_points(self, start_x, start_y, target_x, target_y, distance, options):
        dx = target_x - start_x
        dy = target_y - start_y

        base_deviation = distance * (0.10 + random.random() * 0.32)
        deviation = base_deviation * 0.35 if options.get("isApproach") else base_deviation

        length = math.sqrt(dx * dx + dy * dy) or 1
        perp_x = -dy / length
        perp_y = dx / length

        direction_bias = 1 if random.random() < 0.65 else -1

        c1_factor_base = 0.18 + random.random() * 0.24
        c2_factor_base = 0.54 + random.random() * 0.28
        asymmetry = (random.random() - 0.5) * 0.22

        c1_factor = self.clamp(c1_factor_base + asymmetry, 0.15, 0.48)
        c2_factor = self.clamp(c2_factor_base - asymmetry, 0.50, 0.88)

        c1_deviation = deviation * (0.5 + random.random() * 0.6)
        c2_deviation = deviation * (0.4 + random.random() * 0.7)

        fatigue_impact = self.config["fatigueMultiplier"]

        c1x = start_x + dx * c1_factor + direction_bias * c1_deviation * perp_x * fatigue_impact
        c1y = start_y + dy * c1_factor + direction_bias * c1_deviation * perp_y * fatigue_impact
        c2x = start_x + dx * c2_factor + direction_bias * c2_deviation * perp_x * fatigue_impact
        c2y = start_y + dy * c2_factor + direction_bias * c2_deviation * perp_y * fatigue_impact

        return {
            "p0": {"x": start_x, "y": start_y},
            "p1": {"x": c1x, "y": c1y},
            "p2": {"x": c2x, "y": c2y},
            "p3": {"x": target_x, "y": target_y},
        }

    def generate_realistic_correction_path(self, overshoot_x, overshoot_y, target_x, target_y, viewport, options):
        correction_d = self.calculate_distance(
            {"x": overshoot_x, "y": overshoot_y},
            {"x": target_x, "y": target_y}
        )

        correction_num_points = max(8, round(correction_d / 10))
        base_jitter = options.get("jitterStdDev", 1.5)
        jitter_std_dev = base_jitter * 0.6 * self.config["fatigueMultiplier"]

        dx = target_x - overshoot_x
        dy = target_y - overshoot_y
        length = math.sqrt(dx * dx + dy * dy) or 1

        correction_deviation = correction_d * (0.03 + random.random() * 0.09)
        perp_x = -dy / length
        perp_y = dx / length
        correction_sign = -1 if random.random() < 0.5 else 1

        c1x = overshoot_x + dx * 0.32 + correction_sign * correction_deviation * perp_x * random.random()
        c1y = overshoot_y + dy * 0.32 + correction_sign * correction_deviation * perp_y * random.random()
        c2x = overshoot_x + dx * 0.75 + correction_sign * correction_deviation * perp_x * random.random()
        c2y = overshoot_y + dy * 0.75 + correction_sign * correction_deviation * perp_y * random.random()

        p0 = {"x": overshoot_x, "y": overshoot_y}
        p1 = {"x": c1x, "y": c1y}
        p2 = {"x": c2x, "y": c2y}
        p3 = {"x": target_x, "y": target_y}

        correction_points = []

        for i in range(1, correction_num_points + 1):
            linear_t = i / correction_num_points
            eased_t = self.multi_layer_easing(linear_t, correction_d)
            point = self.get_bezier_point(eased_t, p0, p1, p2, p3)
            point["x"] += self.random_gaussian(0, jitter_std_dev)
            point["y"] += self.random_gaussian(0, jitter_std_dev)
            point["x"] = self.clamp(point["x"], 0, viewport["width"] - 1)
            point["y"] = self.clamp(point["y"], 0, viewport["height"] - 1)
            correction_points.append(point)

        return correction_points

    def handle_realistic_overshoot(self, start_x, start_y, target_x, target_y, box, viewport, points, options, distance, width):
        adjusted_overshoot_prob = options.get("overshootProb", 0.16) * self.config["fatigueMultiplier"]
        is_random_target = box is None

        should_overshoot = (
            (not is_random_target)
            and (not options.get("isApproach"))
            and distance > 120
            and random.random() < adjusted_overshoot_prob
            and self.config["attentionSpan"] < 0.92
        )

        if not should_overshoot:
            return {"points": points, "finalPos": {"x": target_x, "y": target_y}}

        dx = target_x - start_x
        dy = target_y - start_y
        length = math.sqrt(dx * dx + dy * dy) or 1
        dir_x = dx / length
        dir_y = dy / length

        overshoot_factor = (0.08 + random.random() * 0.20) * self.config["fatigueMultiplier"]
        overshoot_dist = overshoot_factor * width

        overshoot_x = target_x + dir_x * overshoot_dist
        overshoot_y = target_y + dir_y * overshoot_dist

        margin = 20
        if (
            overshoot_x < margin or overshoot_x >= viewport["width"] - margin or
            overshoot_y < margin or overshoot_y >= viewport["height"] - margin
        ):
            overshoot_dist *= 0.5
            overshoot_x = target_x + dir_x * overshoot_dist
            overshoot_y = target_y + dir_y * overshoot_dist

        overshoot_x = self.clamp(overshoot_x, margin, viewport["width"] - margin)
        overshoot_y = self.clamp(overshoot_y, margin, viewport["height"] - margin)

        overshoot_result = self.calculate_human_bezier_points(
            start_x, start_y, overshoot_x, overshoot_y, box, viewport,
            {**options, "overshootProb": 0}
        )

        correction_points = self.generate_realistic_correction_path(
            overshoot_x, overshoot_y, target_x, target_y, viewport, options
        )

        return {
            "points": overshoot_result["points"] + correction_points,
            "finalPos": {"x": target_x, "y": target_y},
        }

    # Change the method name to reflect its new generator nature
    def generate_human_bezier_points(self, start_x, start_y, target_x, target_y, box, viewport, options):
        distance = self.calculate_distance({"x": start_x, "y": start_y}, {"x": target_x, "y": target_y})
        width = min(box["width"], box["height"]) if box else options.get("defaultTargetWidth", 100)

        ID = math.log2(distance / width + 1)
        predicted_mt = (self.config["fittsA"] + self.config["fittsB"] * ID) * 1000
        adjusted_mt = predicted_mt * self.config["fatigueMultiplier"] * (0.95 + random.random() * 0.1)

        complexity_multiplier = 1.0
        if self.config["curveComplexity"] == "low":
            complexity_multiplier = 0.7
        elif self.config["curveComplexity"] == "high":
            complexity_multiplier = 1.3

        base_num_points = round(adjusted_mt / self.config["typicalPollingInterval"])
        base_num_points = max(15, round(base_num_points * complexity_multiplier))
        base_num_points = self.apply_fatigue(base_num_points)
        num_points = options.get("numPoints", base_num_points)

        controls = self.calculate_realistic_control_points(
            start_x, start_y, target_x, target_y, distance, options
        )

        # Note: target_drift metadata can be handled externally or stored in self.motion_state
        base_jitter = options.get("jitterStdDev", 1.5)
        jitter_std_dev = base_jitter * self.config["fatigueMultiplier"]

        velocity_profile = self.generate_velocity_profile(num_points, distance)

        # 🛠️ TRACK STATE LOCALLY: Instead of checking points[-1], keep a local reference
        prev_point = None
        generated_base_points = []  # Keep track only if your overshoot handler absolutely needs them

        for i in range(1, num_points + 1):
            linear_t = i / num_points
            eased_t = self.multi_layer_easing(linear_t, distance)

            point = self.get_bezier_point(
                eased_t,
                controls["p0"],
                controls["p1"],
                controls["p2"],
                controls["p3"],
            )

            if random.random() < self.config["microCorrectionFrequency"] and 0.2 < linear_t < 0.9:
                correction_angle = random.random() * math.pi * 2
                correction_magnitude = self.random_gaussian(0, 4 * self.config["fatigueMultiplier"])

                for depth in range(self.config["fractalDepth"]):
                    scale = 0.5 ** depth
                    fractal_noise = self.perlin_noise(
                        i * 0.1 * (depth + 1),
                        linear_t * 10 * (depth + 1),
                        self.motion_state["perlinSeed"] + depth
                    )
                    point["x"] += math.cos(correction_angle) * correction_magnitude * scale + fractal_noise * scale
                    point["y"] += math.sin(correction_angle) * correction_magnitude * scale + fractal_noise * scale

            progress_factor = 1 - eased_t
            distance_to_end = progress_factor * distance
            velocity_influence = velocity_profile[i - 1]
            adaptive_jitter = jitter_std_dev * min(1.5, distance_to_end / 70) * (0.8 + velocity_influence * 0.4)

            gaussian_noise = self.random_gaussian(0, adaptive_jitter)
            # Split into independent X and Y Gaussian noise components
            gaussian_noise_x = self.random_gaussian(0, adaptive_jitter)
            gaussian_noise_y = self.random_gaussian(0, adaptive_jitter)

            perlin_noise_x = self.perlin_noise(i * 0.15, 0, self.motion_state["perlinSeed"]) * adaptive_jitter * 0.3
            perlin_noise_y = self.perlin_noise(0, i * 0.15, self.motion_state["perlinSeed"] + 1) * adaptive_jitter * 0.3

            point["x"] += gaussian_noise_x + perlin_noise_x
            point["y"] += gaussian_noise_y + perlin_noise_y

            if self.config["attentionSpan"] < 0.95:
                if random.random() > self.config["attentionSpan"]:
                    error_magnitude = (1 - self.config["attentionSpan"]) * 18 * self.config["fatigueMultiplier"]
                    point["x"] += self.random_gaussian(0, error_magnitude * 0.25)
                    point["y"] += self.random_gaussian(0, error_magnitude * 0.25)

            if 0.3 < linear_t < 0.85 and random.random() < 0.12:
                # Split sub-movements into independent axis adjustments
                sub_movement_x = self.random_gaussian(0, 2.5 * self.config["fatigueMultiplier"])
                sub_movement_y = self.random_gaussian(0, 2.5 * self.config["fatigueMultiplier"])
                point["x"] += sub_movement_x
                point["y"] += sub_movement_y

            # 🛠️ FIXED: Replaced `len(points) > 0` and `points[-1]` with `prev_point`
            if prev_point is not None and random.random() < 0.2:
                angle = math.atan2(point["y"] - prev_point["y"], point["x"] - prev_point["x"])
                angle_variation = self.random_gaussian(0, 0.08)
                dist = self.calculate_distance(prev_point, point)
                point["x"] = prev_point["x"] + math.cos(angle + angle_variation) * dist
                point["y"] = prev_point["y"] + math.sin(angle + angle_variation) * dist

            point["x"] = self.clamp(point["x"], 0, viewport["width"] - 1)
            point["y"] = self.clamp(point["y"], 0, viewport["height"] - 1)

            # Yield the point out immediately to the runtime loop!
            yield point

            prev_point = point
            generated_base_points.append(point)

        # 🛠️ OVERSHOOT HANDLING:
        # If your overshoot method returns a dictionary containing a list of points:
        result = self.handle_realistic_overshoot(
            start_x, start_y, target_x, target_y, box, viewport, generated_base_points, options, distance, width
        )

        # Stream any compensation or overshoot corrections out seamlessly
        # Skip the ones we already yielded by slicing or checking if overshoot appended new items
        if len(result["points"]) > len(generated_base_points):
            for over_point in result["points"][len(generated_base_points):]:
                yield over_point

    # -----------------------------
    # Movement
    # -----------------------------

    async def micro_mouse_adjustment(self):
        if not self.last_pos:
            return

        micro_x = self.last_pos["x"] + self.random_gaussian(0, 2.5 * self.config["fatigueMultiplier"])
        micro_y = self.last_pos["y"] + self.random_gaussian(0, 2.5 * self.config["fatigueMultiplier"])

        viewport = await self.get_viewport()

        try:
            await self.page.mouse.move(
                self.clamp(micro_x, 0, viewport["width"] - 1),
                self.clamp(micro_y, 0, viewport["height"] - 1),
            )
        except Exception:
            pass

    # async def move_to_position(self, target_x: float, target_y: float, options: Optional[Dict[str, Any]] = None):
    #     options = options or {}
    #     viewport = await self.get_viewport()
    #     if not self.last_pos:
    #         self.initialize_position(viewport)
    #
    #     # 1. Initialize the new generator version of your path calculation
    #     point_generator = self.generate_human_bezier_points(
    #         self.last_pos["x"], self.last_pos["y"], target_x, target_y, None, viewport, options
    #     )
    #
    #     # Total count evaluation or tracking step progress
    #     # For simplicity, we can track indices locally to feed into phase calculations
    #     step_idx = 0
    #     points_buffer = []
    #
    #     for point in point_generator:
    #         points_buffer.append(point)
    #         step_idx += 1
    #
    #         # 2. Approximate phase progress (estimate or total expected length check)
    #         phase = step_idx / 50.0  # Or track a rolling estimation
    #
    #         # 3. Fetch delay from your calculation function
    #         delay_ms = self.calculate_realistic_polling_delay(phase=phase, velocity_factor=1.0)
    #
    #         # 4. Fire high-precision wait sequence
    #         await self._precise_async_sleep(delay_ms)
    #
    #         # Move hardware mouse via Playwright
    #         await self.page.mouse.move(point["x"], point["y"])
    #         self.last_pos = {"x": point["x"], "y": point["y"]}

    async def move_to_position(self, target_x: float, target_y: float,
                                        options: Optional[Dict[str, Any]] = None):
        options = options or {}
        viewport = await self.get_viewport()
        if not self.last_pos:
            self.initialize_position(viewport)

        point_generator = self.generate_human_bezier_points(
            self.last_pos["x"], self.last_pos["y"], target_x, target_y, None, viewport, options
        )

        for point in point_generator:
            # Fluctuate the delay slightly around the 4ms mark to break mechanical consistency
            dynamic_delay = random.uniform(2.5, 5.5)
            await self._precise_async_sleep(dynamic_delay)

            # Dispatch mouse position modification
            await self.page.mouse.move(point["x"], point["y"])
            self.last_pos = {"x": point["x"], "y": point["y"]}
    async def move(self, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        viewport = await self.get_viewport()

        if not self.last_pos:
            self.initialize_position(viewport)

        padding = 60
        target_x = padding + random.random() * (viewport["width"] - 2 * padding)
        target_y = padding + random.random() * (viewport["height"] - 2 * padding)

        await self.move_to_position(target_x, target_y, options)
        self.update_action_count()
    async def move_with_curve(self, x1: float, y1: float, x2: float, y2: float, num_points: int):
        # 1. Curve Setup
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = math.atan2(y2 - y1, x2 - x1)

        curve_offset = random.uniform(-0.15, 0.15) * distance
        ctrl_x = mid_x + curve_offset * math.cos(angle + math.pi / 2)
        ctrl_y = mid_y + curve_offset * math.sin(angle + math.pi / 2)

        # Correlated noise state (The "Drift")
        drift_x, drift_y = 0.0, 0.0

        # 2. Total Duration Calculation (Fitts's Law approximation)
        # Average human move takes 150ms-400ms based on distance
        target_duration = (math.log2(distance / 10 + 1) * 100) * random.uniform(0.8, 1.2)
        start_time = time.time()

        for i in range(1, num_points + 1):
            progress = i / num_points

            # 3. ASYMMETRIC EASING (Ballistic Phase)
            # Power easing (t^2 or t^3) creates a faster acceleration than Sine
            if progress < 0.7:
                t = progress ** 2 / (progress ** 2 + (1 - progress) ** 2)
            else:
                # Slower, more linear correction at the end
                t = progress

                # 4. Quadratic Bézier
            base_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x2
            base_y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y2

            # 5. CORRELATED TREMOR (Pink Noise)
            # Instead of random.uniform, we 'evolve' the noise
            drift_x = (drift_x * 0.4) + random.uniform(-0.6, 0.6)
            drift_y = (drift_y * 0.4) + random.uniform(-0.6, 0.6)

            current_x = float(base_x + drift_x)
            current_y = float(base_y + drift_y)

            await self.page.mouse.move(current_x, current_y)

            # 6. DYNAMIC THROTTLING
            # Calculate how much time we SHOULD have spent by now
            elapsed = (time.time() - start_time) * 1000
            expected = (progress * target_duration)

            if elapsed < expected:
                # Only sleep if we are ahead of the 'human' pace
                await asyncio.sleep((expected - elapsed) / 1000)
            else:
                # If Playwright/OS overhead is high, yield to the event loop
                await asyncio.sleep(0)

        self.last_pos = {"x": float(x2), "y": float(y2)}
    async def move_to_element_naturally(self, element, options=None):
        options = options or {}
        box = await self.get_element_bounding_box(element)
        if not box:
            return

        viewport = await self.get_viewport()

        # 1. Human Logic: Aim for a random spot within the element
        # Use your existing calculate_click_target logic
        target = self.calculate_click_target(box, options)

        # 2. Add an "Attention" delay (mimic eye movement)
        await self.random_delay(150, 350)

        # 3. Use your sophisticated movement logic
        # This calls your calculate_human_bezier_points and handles the loop
        await self.move_to_position(target["x"], target["y"], options)

        self.log(f"Naturally moved to element at {target}")

    async def move_to_tile_naturally(self, element, options: Optional[Dict[str, Any]] = None):
        """
        Specialized high-velocity movement engine for grid arrays.
        Uses approach-relative targeting to mimic human corner-cutting.
        """
        options = options or {}
        box = await self.get_element_bounding_box(element)
        if not box:
            return

        if not self.last_pos:
            self.last_pos = {"x": box["x"] + box["width"] * 0.5, "y": box["y"] + box["height"] * 0.5}

        current_x = self.last_pos["x"]
        current_y = self.last_pos["y"]

        # 1. 🎯 THE HUMAN CORNER-CUTTING FIX
        # Calculate the dead center of the target tile
        center_x = box["x"] + box["width"] * 0.5
        center_y = box["y"] + box["height"] * 0.5

        # Humans rarely move all the way to the center of a tile if they enter from the side.
        # We pull the destination 12% to 26% closer to where the mouse is currently resting.
        corner_cut_bias = random.uniform(0.12, 0.26)
        target_x = center_x + (current_x - center_x) * corner_cut_bias
        target_y = center_y + (current_y - center_y) * corner_cut_bias

        # 2. STRICT INNER SAFE-ZONE CLAMPING
        # Ensure that even with heavy bias, the coordinate remains cleanly inside the image frame
        pad_w = box["width"] * 0.18
        pad_h = box["height"] * 0.18
        target_x = self.clamp(target_x, box["x"] + pad_w, box["x"] + box["width"] - pad_w)
        target_y = self.clamp(target_y, box["y"] + pad_h, box["y"] + box["height"] - pad_h)

        # 3. ⚡ MUSCLE MEMORY SPEED INJECTION
        # Rapid grid clicks use highly aggressive kinematics compared to slow menu navigation.
        # We pass custom modifiers to your library's underlying Bezier generator.
        tile_options = {
            **options,
            "speedMultiplier": random.uniform(1.4, 1.8),  # Snap swiftly to the target
            "entropyTarget": self.config.get("entropyTarget", 1.0) * 0.75,  # Cleaner, direct paths
        }

        # Execute using your updated, ultra-smooth movement loop
        await self.move_to_position(target_x, target_y, options=tile_options)
    # -----------------------------
    # Scroll
    # -----------------------------

    async def pre_scroll_mouse_movement(self, viewport, options):
        if not self.last_pos:
            self.initialize_position(viewport)

        hover_target = {
            "x": viewport["width"] * (0.25 + random.random() * 0.5),
            "y": viewport["height"] * (0.15 + random.random() * 0.7),
        }

        distance = self.calculate_distance(self.last_pos, hover_target)
        if distance > 60:
            await self.move_to_position(
                hover_target["x"],
                hover_target["y"],
                {**options, "numPoints": max(6, round(distance / 60))}
            )

    async def execute_scroll_sequence(self, target_scroll, direction, num_steps, overshoot_amount, scroll_container, options):
        base_jitter_std_dev = options.get("scrollJitterStdDev", 18)
        jitter_std_dev = base_jitter_std_dev * self.config["fatigueMultiplier"]

        for i in range(1, num_steps + 1):
            try:
                current_scroll = await self.get_current_scroll_y()

                remaining_delta = abs(target_scroll - current_scroll)
                if remaining_delta < 8:
                    break

                progress = i / num_steps
                log_deceleration = 1 - math.log10(1 + 9 * progress)
                eased_progress = self.ease_in_out_cubic(progress)
                blended_progress = eased_progress * 0.6 + log_deceleration * 0.4

                step_delta = (remaining_delta * (1 - blended_progress) * 0.3) / self.config["fatigueMultiplier"]

                distance_based_jitter = min(jitter_std_dev, remaining_delta * 0.12)
                step_delta += self.random_gaussian(0, distance_based_jitter)
                step_delta = self.clamp(step_delta, 8, 180)

                if overshoot_amount > 0 and i > num_steps * 0.75:
                    overshoot_fraction = (i - num_steps * 0.75) / (num_steps * 0.25)
                    step_delta += overshoot_amount * overshoot_fraction * 0.4

                await self.page.mouse.wheel(0, direction * step_delta)

                base_delay = (18 + random.random() * 75) * self.config["fatigueMultiplier"]
                micro_pause = random.random() * 90 if random.random() < 0.12 else 0
                await self.random_delay(base_delay, base_delay + micro_pause)

                if random.random() < 0.18:
                    await self.micro_mouse_adjustment()

            except Exception as e:
                self.log("execute_scroll_sequence failed:", str(e))
                break

    async def execute_correction_scroll_logarithmic(self, target_scroll, direction, correction_steps, scroll_container, options):
        base_jitter_std_dev = options.get("scrollJitterStdDev", 18) / 2
        jitter_std_dev = base_jitter_std_dev * self.config["fatigueMultiplier"]

        for i in range(1, correction_steps + 1):
            try:
                current_scroll = await self.get_current_scroll_y()
                correction_delta = abs(target_scroll - current_scroll)
                if correction_delta < 8:
                    break

                progress = i / correction_steps
                log_factor = 1 - math.log10(1 + 9 * progress)

                step_delta = (correction_delta * log_factor * 0.4) / self.config["fatigueMultiplier"]
                step_delta += self.random_gaussian(0, jitter_std_dev)
                step_delta = self.clamp(step_delta, 8, 130)

                await self.page.mouse.wheel(0, -direction * step_delta)
                await self.random_delay(
                    12 * self.config["fatigueMultiplier"],
                    65 * self.config["fatigueMultiplier"]
                )
            except Exception as e:
                self.log("execute_correction_scroll_logarithmic failed:", str(e))
                break

    async def scroll_to_element(self, element, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        viewport = await self.get_viewport()

        if await self.is_element_in_viewport(element, options.get("visibilityBuffer", 50)):
            if random.random() < 0.25:
                micro_scroll = self.random_gaussian(0, 12)
                await self.page.mouse.wheel(0, micro_scroll)
                await self.random_delay(50, 150)
            return

        box = await self.get_element_bounding_box(element)
        if not box:
            raise RuntimeError("Element has no bounding box")

        scroll_container = await self.get_scroll_container(element)
        target_position = options.get("targetPosition", "center")

        current_scroll = viewport["scrollY"]

        if target_position == "top":
            target_scroll = box["y"] - options.get("offset", 100)
        elif target_position == "bottom":
            target_scroll = box["y"] + box["height"] - viewport["height"] + options.get("offset", 100)
        else:
            target_scroll = box["y"] + box["height"] / 2 - viewport["height"] / 2

        max_scroll = scroll_container["info"]["scrollHeight"] - viewport["height"]
        target_scroll = self.clamp(target_scroll, 0, max_scroll)

        await self.pre_scroll_mouse_movement(viewport, options)

        delta = abs(target_scroll - current_scroll)
        if delta < 10:
            return

        direction = 1 if target_scroll > current_scroll else -1
        scroll_id = math.log2(delta / 100 + 1)
        base_steps = max(5, round(8 * scroll_id))
        num_steps = self.apply_fatigue(base_steps)

        overshoot_prob = options.get("overshootProb", 0.18)
        should_overshoot = (
            delta > 250 and random.random() < overshoot_prob and self.config["attentionSpan"] < 0.94
        )

        overshoot_amount = 0
        if should_overshoot:
            overshoot_amount = self.random_gaussian(0.15, 0.07) * viewport["height"]
            overshoot_amount = self.clamp(overshoot_amount, 40, viewport["height"] * 0.35)

        await self.execute_scroll_sequence(
            target_scroll, direction, num_steps, overshoot_amount, scroll_container, options
        )

        if overshoot_amount > 0:
            await self.random_delay(120, 350)
            await self.execute_correction_scroll_logarithmic(
                target_scroll,
                direction,
                max(3, round(num_steps / 3)),
                scroll_container,
                options
            )

        await self.random_delay(80, 180)
        self.update_action_count()

    # -----------------------------
    # Click
    # -----------------------------

    def calculate_click_target(self, box, options):
        click_padding_factor = options.get("clickPadding", 0.68)
        fatigue_offset = (self.config["fatigueMultiplier"] - 1) * 0.15

        bias_x = -0.1 + fatigue_offset
        bias_y = -0.05 + fatigue_offset

        offset_x = (
            self.random_gaussian(bias_x, 0.25 * self.config["fatigueMultiplier"]) * box["width"]
        ) * click_padding_factor
        offset_y = (
            self.random_gaussian(bias_y, 0.25 * self.config["fatigueMultiplier"]) * box["height"]
        ) * click_padding_factor

        target_x = box["x"] + box["width"] / 2 + offset_x
        target_y = box["y"] + box["height"] / 2 + offset_y

        margin_x = min(8, box["width"] * 0.1)
        margin_y = min(8, box["height"] * 0.1)

        target_x = self.clamp(target_x, box["x"] + margin_x, box["x"] + box["width"] - margin_x)
        target_y = self.clamp(target_y, box["y"] + margin_y, box["y"] + box["height"] - margin_y)

        return {"x": target_x, "y": target_y}

    def calculate_natural_approach_target(self, click_target, box, viewport):
        if not self.last_pos:
            distance = 25 + random.random() * 35
            angle = random.random() * math.pi * 2
            x = click_target["x"] + math.cos(angle) * distance
            y = click_target["y"] + math.sin(angle) * distance
            return {
                "x": self.clamp(x, 0, viewport["width"] - 1),
                "y": self.clamp(y, 0, viewport["height"] - 1),
            }

        dx = click_target["x"] - self.last_pos["x"]
        dy = click_target["y"] - self.last_pos["y"]
        distance = math.sqrt(dx * dx + dy * dy) or 1

        dir_x = dx / distance
        dir_y = dy / distance
        approach_distance = 25 + random.random() * 35

        perpendicular_angle = math.atan2(dir_y, dir_x) + (random.random() - 0.5) * (math.pi / 6)
        jitter_magnitude = (random.random() - 0.5) * 20 * self.config["fatigueMultiplier"]

        x = click_target["x"] - dir_x * approach_distance + math.cos(perpendicular_angle) * jitter_magnitude
        y = click_target["y"] - dir_y * approach_distance + math.sin(perpendicular_angle) * jitter_magnitude

        return {
            "x": self.clamp(x, 0, viewport["width"] - 1),
            "y": self.clamp(y, 0, viewport["height"] - 1),
        }

    async def post_click_behavior(self, click_target, viewport, options):
        behavior = random.random()

        if behavior < 0.35:
            await self.random_delay(120, 550)
        elif behavior < 0.65:
            jitter_x = click_target["x"] + self.random_gaussian(0, 6 * self.config["fatigueMultiplier"])
            jitter_y = click_target["y"] + self.random_gaussian(0, 6 * self.config["fatigueMultiplier"])

            await self.move_to_position(
                self.clamp(jitter_x, 0, viewport["width"] - 1),
                self.clamp(jitter_y, 0, viewport["height"] - 1),
                {**options, "numPoints": 2},
            )
            await self.random_delay(60, 220)
        else:
            away_distance = 35 + random.random() * 80
            away_angle = random.random() * math.pi * 2
            away_x = click_target["x"] + math.cos(away_angle) * away_distance
            away_y = click_target["y"] + math.sin(away_angle) * away_distance

            await self.move_to_position(
                self.clamp(away_x, 0, viewport["width"] - 1),
                self.clamp(away_y, 0, viewport["height"] - 1),
                options,
            )
    async def hover_jitter(self, duration_ms, intensity=0.4):
        start = self._now_ms()
        while self._now_ms() - start < duration_ms:
            # Brownian-like motion
            jx = self.last_pos["x"] + random.uniform(-intensity, intensity)
            jy = self.last_pos["y"] + random.uniform(-intensity, intensity)
            await self.page.mouse.move(jx, jy)
            await asyncio.sleep(0.01)  # 100Hz polling

    def generate_smooth_path(start, end, num_steps=None):
        start_x, start_y = start
        end_x, end_y = end

        distance = math.hypot(end_x - start_x, end_y - start_y)
        if num_steps is None:
            num_steps = int(max(12, distance / 10))

        # 1. Generate Organic Bezier Control Points
        # Instead of perfectly symmetrical paths, push the control points out naturally
        control_scale = distance * random.uniform(0.1, 0.3)

        # Control point 1 (pushed out from start)
        cp1_x = start_x + (end_x - start_x) * 0.25 + random.uniform(-control_scale, control_scale)
        cp1_y = start_y + (end_y - start_y) * 0.25 + random.uniform(-control_scale, control_scale)

        # Control point 2 (pushed out from end)
        cp2_x = start_x + (end_x - start_x) * 0.75 + random.uniform(-control_scale, control_scale)
        cp2_y = start_y + (end_y - start_y) * 0.75 + random.uniform(-control_scale, control_scale)

        path_points = []
        tremor_x, tremor_y = 0.0, 0.0

        for i in range(num_steps):
            # Linear progress (0.0 to 1.0)
            linear_t = i / (num_steps - 1)

            # 2. Apply Easing Profile (In-Out Cubic Easing)
            if linear_t < 0.5:
                eased_t = 4 * linear_t * linear_t * linear_t
            else:
                eased_t = 1 - ((-2 * linear_t + 2) ** 3) / 2

            # 3. Cubic Bezier Interpolation Formula
            # B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            mt = 1 - eased_t
            x = (mt ** 3 * start_x) + (3 * mt ** 2 * eased_t * cp1_x) + (3 * mt * eased_t ** 2 * cp2_x) + (
                        eased_t ** 3 * end_x)
            y = (mt ** 3 * start_y) + (3 * mt ** 2 * eased_t * cp1_y) + (3 * mt * eased_t ** 2 * cp2_y) + (
                        eased_t ** 3 * end_y)

            # 4. Inject Smooth Coherent Noise (Micro-Tremors)
            # Keeps deviations correlated to prevent frantic micro-stuttering
            tremor_x = (tremor_x * 0.75) + (random.uniform(-0.4, 0.4) * 0.25)
            tremor_y = (tremor_y * 0.75) + (random.uniform(-0.4, 0.4) * 0.25)

            path_points.append({
                "x": x + tremor_x,
                "y": y + tremor_y
            })

        return path_points
    async def click(self, element=None, fast: bool = True):
        """
        Simulates physical hardware mouse switch actuation with discrete pixel coordinates,
        pre-click hover tremor, actuation slippage, and mechanical spring back.
        """
        if not self.last_pos:
            return

        # Ensure starting baseline is cleanly rounded
        origin_x = int(round(self.last_pos["x"]))
        origin_y = int(round(self.last_pos["y"]))

        # 1. THE HOVER TREMOR (Dwell & Deceleration)
        # Human hands micro-tremor slightly right before the downward click force begins
        tremor_steps = 2 if fast else random.randint(3, 5)
        for _ in range(tremor_steps):
            origin_x += random.choice([-1, 0, 1])
            origin_y += random.choice([-1, 0, 1])
            await self.page.mouse.move(origin_x, origin_y)
            await asyncio.sleep(random.uniform(0.005, 0.015))

        # 2. BUTTON DOWN ACTUATION
        await self.page.mouse.down()

        # 3. GAUSSIAN HOLD TIME
        # Physical switches require time to fully depress and release
        hold_time = random.gauss(0.065, 0.012) if fast else random.gauss(0.115, 0.025)
        hold_time = max(0.040, hold_time)  # Enforce physical human compression limit

        # 4. ACTUATION SLIPPAGE (Downward compression force)
        # Finger pressure slightly nudges the physical mouse chassis down/sideways
        slip_x = origin_x
        slip_y = origin_y
        if random.random() < 0.78:
            slip_x += random.choice([-1, 0, 1])
            slip_y += random.choice([0, 1])  # Biased downwards due to finger push
            await self.page.mouse.move(slip_x, slip_y)

        await asyncio.sleep(hold_time)

        # 5. BUTTON UP & MECHANICAL SPRING-BACK
        # As the finger lifts, the switch snaps back, often shifting the mouse 1 pixel back
        await self.page.mouse.up()

        if random.random() < 0.60:
            # Snap back close to original position or settle clean
            recovery_x = slip_x + (1 if origin_x > slip_x else -1 if origin_x < slip_x else 0)
            recovery_y = slip_y + (-1 if slip_y > origin_y else 0)
            await self.page.mouse.move(recovery_x, recovery_y)
            self.last_pos = {"x": float(recovery_x), "y": float(recovery_y)}
        else:
            self.last_pos = {"x": float(slip_x), "y": float(slip_y)}

        # 6. POST-CLICK COGNITIVE VALIDATION
        # The pause happens HERE (visual confirmation), not as a blind sleep before moving
        post_pause = random.uniform(0.02, 0.06) if fast else random.uniform(0.08, 0.18)
        await asyncio.sleep(post_pause)

    # -----------------------------
    # Reset
    # -----------------------------

    def reset(self):
        self.config["actionCount"] = 0
        self.config["attentionSpan"] = 0.88 + random.random() * 0.10
        self.config["fatigueMultiplier"] = 1.0
        self.move_history = []
        self.last_pos = None
        self.invalidate_viewport_cache()

        self.motion_state = {
            "lastVelocity": {"x": 0.0, "y": 0.0},
            "lastAcceleration": {"x": 0.0, "y": 0.0},
            "lastJerk": {"x": 0.0, "y": 0.0},
            "temporalCorrelation": 0.5,
            "entropyAccumulator": 0.0,
            "perlinSeed": random.random() * 10000,
            "pollingPhase": random.random(),
        }

        self.log("State reset complete")

    async def wander_randomly(self, locator, stop_event: asyncio.Event):
        """
        Simulates organic, unpredictable human cognitive idling and element scanning.
        Integrates high-precision sub-millisecond sleeps and real-time state tracking
        to prevent bot-detection flags.
        """
        box = await self.get_element_bounding_box(locator)
        if not box:
            return

        # Initialize local cursor coordinates to current global tracking state
        if not self.last_pos:
            self.last_pos = {"x": box["x"] + box["width"] * 0.5, "y": box["y"] + box["height"] * 0.5}

        current_x = self.last_pos["x"]
        current_y = self.last_pos["y"]

        while not stop_event.is_set():
            behavior = random.choices(
                ["fixation", "drift", "saccade"],
                weights=[0.45, 0.40, 0.15],
                k=1
            )[0]

            if behavior == "fixation":
                # 👀 FIXATION: Siting still while processing data. Micro-tremors only.
                duration = random.uniform(0.12, 0.35)
                steps = int(duration / 0.015)

                for _ in range(steps):
                    if stop_event.is_set():
                        break

                    tremor_x = current_x + random.uniform(-0.4, 0.4)
                    tremor_y = current_y + random.uniform(-0.4, 0.4)

                    try:
                        await self.page.mouse.move(tremor_x, tremor_y)
                        # 🛠️ FIX 1: Instantly sync state to avoid teleportation on exit
                        self.last_pos = {"x": tremor_x, "y": tremor_y}
                    except Exception:
                        pass

                    # 🛠️ FIX 2: High precision sleep to hit exactly ~15ms
                    await self._precise_async_sleep(15.0)

            elif behavior == "drift":
                # 🐌 DRIFT: Lazy, wandering glance across the layout container.
                pad_w, pad_h = box["width"] * 0.15, box["height"] * 0.15
                target_x = random.uniform(box["x"] + pad_w, box["x"] + box["width"] - pad_w)
                target_y = random.uniform(box["y"] + pad_h, box["y"] + box["height"] - pad_h)

                distance = math.hypot(target_x - current_x, target_y - current_y)
                steps = int(max(8, distance / 6))

                for i in range(steps):
                    if stop_event.is_set():
                        break
                    t = i / max(1, steps - 1)

                    # Imperfect bowing arc to the movement path
                    bow_factor = math.sin(t * math.pi) * random.uniform(-6, 6)
                    interp_x = current_x + (target_x - current_x) * t + bow_factor
                    interp_y = current_y + (target_y - current_y) * t + (bow_factor * 0.5)

                    try:
                        await self.page.mouse.move(interp_x, interp_y)
                        self.last_pos = {"x": interp_x, "y": interp_y}  # 🛠️ FIX 1
                    except Exception:
                        pass

                    await self._precise_async_sleep(random.uniform(20.0, 35.0))

                current_x = self.last_pos["x"]
                current_y = self.last_pos["y"]

            elif behavior == "saccade":
                # ⚡ SACCADE: Sudden fast visual focus jump to a new area.
                pad_w, pad_h = box["width"] * 0.1, box["height"] * 0.1
                target_x = random.uniform(box["x"] + pad_w, box["x"] + box["width"] - pad_w)
                target_y = random.uniform(box["y"] + pad_h, box["y"] + box["height"] - pad_h)

                steps = random.randint(4, 7)
                for i in range(steps):
                    if stop_event.is_set():
                        break
                    t = i / max(1, steps - 1)

                    # Cubic-Out Easing: Fast snap outwards, heavy deceleration at end
                    eased_t = 1 - ((1 - t) ** 3)
                    interp_x = current_x + (target_x - current_x) * eased_t
                    interp_y = current_y + (target_y - current_y) * eased_t

                    try:
                        await self.page.mouse.move(interp_x, interp_y)
                        self.last_pos = {"x": interp_x, "y": interp_y}  # 🛠️ FIX 1
                    except Exception:
                        pass

                    # 🛠️ FIX 2: Standard sleep will break a 10ms execution. Enforce precision.
                    await self._precise_async_sleep(10.0)

                current_x = self.last_pos["x"]
                current_y = self.last_pos["y"]

            # Small variation delay before picking the next focus state
            if not stop_event.is_set():
                await self._precise_async_sleep(random.uniform(50.0, 120.0))

        # Final guarantee that our state is completely aligned
        self.last_pos = {"x": current_x, "y": current_y}