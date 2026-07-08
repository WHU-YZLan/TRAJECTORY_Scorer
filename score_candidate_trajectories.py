#!/usr/bin/env python3
"""Score planner candidate trajectories with an AirScape-style interface.

This first version implements the planner-trajectory to motion-prompt bridge.
Later stages can plug the prompt into AirScape generation and scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data0/llj/codex-airscape/checkpoints/CogVideoX-5b-I2V")
DEFAULT_TRANSFORMER = Path("/data0/llj/codex-airscape/checkpoints/Airscape/phase1")
DEFAULT_OUTPUT_DIR = Path("/data0/llj/codex-airscape/outputs/candidate_trajectories")
DEFAULT_MAX_V = 10.0
DEFAULT_MAX_YAW_RATE = 1.57
DEFAULT_MAX_ACC = 5.0
DEFAULT_MIN_ALTITUDE = 1.0
DEFAULT_UNCERTAINTY_SAMPLES = 3


def trajectory_to_motion_prompt(
    traj: dict[str, Any],
    goal: dict[str, Any],
    tail_condition: str | None = None,
) -> str:
    """Convert a numeric planner trajectory into an AirScape motion prompt.

    The expected waypoint convention is body-frame relative motion:
    dx forward/backward, dy right/left, dz up/down, dyaw right/left rotation.
    """
    total_dx = sum(p["dx"] for p in traj["waypoints"])
    total_dy = sum(p["dy"] for p in traj["waypoints"])
    total_dz = sum(p["dz"] for p in traj["waypoints"])
    total_dyaw = sum(p["dyaw"] for p in traj["waypoints"])

    motion: list[str] = []

    if total_dx > 0:
        motion.append(f"fly forward {total_dx:.1f} meters")
    if total_dx < 0:
        motion.append(f"fly backward {abs(total_dx):.1f} meters")
    if total_dy > 0:
        motion.append(f"move right {total_dy:.1f} meters")
    if total_dy < 0:
        motion.append(f"move left {abs(total_dy):.1f} meters")
    if total_dz > 0:
        motion.append(f"ascend {total_dz:.1f} meters")
    if total_dz < 0:
        motion.append(f"descend {abs(total_dz):.1f} meters")
    if total_dyaw > 0:
        motion.append(f"turn right {abs(total_dyaw):.1f} radians")
    if total_dyaw < 0:
        motion.append(f"turn left {abs(total_dyaw):.1f} radians")

    prompt = "The drone will " + ", ".join(motion)

    if not motion:
        prompt = "The drone will hover in place"

    if goal and goal.get("instruction"):
        prompt += f", while following the instruction: {goal['instruction']}"

    if tail_condition:
        prompt += f". The scene condition is {tail_condition}."

    return prompt


def prompts_from_request(request: dict[str, Any]) -> dict[str, str]:
    """Return an AirScape motion prompt for every candidate trajectory."""
    goal = request.get("goal") or {}
    tail_condition = request.get("tail_condition")
    prompts: dict[str, str] = {}

    for traj in request.get("candidate_trajectories", []):
        traj_id = str(traj.get("traj_id"))
        prompts[traj_id] = trajectory_to_motion_prompt(traj, goal, tail_condition)

    return prompts


def _waypoint_dt(point: dict[str, Any]) -> float:
    dt = float(point.get("dt", 0.0))
    if dt <= 0:
        raise ValueError("waypoint dt must be positive")
    return dt


def check_dynamics(
    traj: dict[str, Any],
    max_v: float = DEFAULT_MAX_V,
    max_yaw_rate: float = DEFAULT_MAX_YAW_RATE,
    max_acc: float = DEFAULT_MAX_ACC,
) -> tuple[float, bool, dict[str, float]]:
    """Check simple kinematic constraints for a candidate trajectory."""
    waypoints = traj.get("waypoints", [])
    if not waypoints:
        return 0.0, True, {"max_velocity": 0.0, "max_yaw_rate": 0.0, "max_acceleration": 0.0}

    velocities: list[np.ndarray] = []
    yaw_rates: list[float] = []

    for point in waypoints:
        dt = _waypoint_dt(point)
        velocity = np.array(
            [
                float(point.get("dx", 0.0)) / dt,
                float(point.get("dy", 0.0)) / dt,
                float(point.get("dz", 0.0)) / dt,
            ],
            dtype=np.float32,
        )
        velocities.append(velocity)
        yaw_rates.append(abs(float(point.get("dyaw", 0.0))) / dt)

    max_velocity = max(float(np.linalg.norm(v)) for v in velocities)
    max_yaw = max(yaw_rates) if yaw_rates else 0.0

    accelerations = []
    for prev, cur, point in zip(velocities[:-1], velocities[1:], waypoints[1:]):
        dt = _waypoint_dt(point)
        accelerations.append(float(np.linalg.norm(cur - prev) / dt))
    max_acceleration = max(accelerations) if accelerations else 0.0

    ratios = [
        max_velocity / max_v if max_v > 0 else math.inf,
        max_yaw / max_yaw_rate if max_yaw_rate > 0 else math.inf,
        max_acceleration / max_acc if max_acc > 0 else math.inf,
    ]
    violation = any(ratio > 1.0 for ratio in ratios)
    # Smooth score: 1 when fully legal, decays with the worst normalized violation.
    dynamics_score = max(0.0, min(1.0, 1.0 / max(1.0, max(ratios))))

    details = {
        "max_velocity": max_velocity,
        "max_yaw_rate": max_yaw,
        "max_acceleration": max_acceleration,
    }
    return dynamics_score, violation, details


def extract_final_frame(video_path: str | Path, output_path: str | Path | None = None) -> Image.Image:
    """Read and optionally save the final RGB frame of a video."""
    frame = None
    for frame in iio.imiter(video_path):
        pass
    if frame is None:
        raise ValueError(f"no frames found in video: {video_path}")

    image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    return image


def read_video_frames(video_path: str | Path, max_frames: int | None = None) -> list[Image.Image]:
    """Read RGB frames from a video, optionally uniformly sub-sampling them."""
    raw_frames = [Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB") for frame in iio.imiter(video_path)]
    if max_frames is None or len(raw_frames) <= max_frames:
        return raw_frames
    indices = np.linspace(0, len(raw_frames) - 1, max_frames).round().astype(int)
    return [raw_frames[int(i)] for i in indices]


def _tokenize_text(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    stopwords = {
        "the",
        "a",
        "an",
        "to",
        "and",
        "or",
        "of",
        "in",
        "on",
        "with",
        "while",
        "will",
        "drone",
        "fly",
        "move",
        "turn",
    }
    return {token for token in cleaned.split() if len(token) > 2 and token not in stopwords}


def compute_safety_score(
    traj: dict[str, Any],
    uav_state: dict[str, Any] | None,
    dynamics_rejected: bool,
    min_altitude: float = DEFAULT_MIN_ALTITUDE,
) -> float:
    """MVP safety score using trajectory geometry.

    This is the document's priority-3 fallback before a depth/collision model is
    available. It penalizes dynamically invalid trajectories and trajectories
    that descend below a safe altitude.
    """
    if dynamics_rejected:
        return 0.0

    z = float((uav_state or {}).get("z", min_altitude + 1.0))
    min_z = z
    for point in traj.get("waypoints", []):
        z += float(point.get("dz", 0.0))
        min_z = min(min_z, z)

    if min_z < min_altitude:
        return 0.2
    return 0.9


def compute_semantic_score(pred_video: str | Path, instruction: str, motion_prompt: str) -> float:
    """MVP semantic score before a VLM scorer is connected.

    A real implementation should ask a VLM whether the generated video follows
    both the task instruction and motion prompt. This fallback only checks that
    the motion prompt carries task-relevant words, so it is intentionally
    conservative rather than pretending to understand the video.
    """
    del pred_video
    if not instruction:
        return 0.7

    instruction_tokens = _tokenize_text(instruction)
    prompt_tokens = _tokenize_text(motion_prompt)
    if not instruction_tokens:
        return 0.7

    overlap = len(instruction_tokens & prompt_tokens) / len(instruction_tokens)
    return max(0.2, min(1.0, 0.5 + 0.5 * overlap))


def compute_temporal_score(pred_video: str | Path, max_frames: int = 16) -> float:
    """MVP temporal continuity score from adjacent-frame changes."""
    frames = read_video_frames(pred_video, max_frames=max_frames)
    if len(frames) < 2:
        return 0.0

    resized = [_pil_to_pixel_tensor(frame, (160, 96)) for frame in frames]
    diffs = [float(np.mean(np.abs(b - a))) for a, b in zip(resized[:-1], resized[1:])]
    mean_diff = float(np.mean(diffs))
    jumpiness = float(np.std(diffs))

    # Adjacent-frame RGB changes around 0.05-0.10 are common for smooth motion.
    # Larger and inconsistent jumps are penalized.
    penalty = 3.0 * mean_diff + 2.0 * jumpiness
    return max(0.0, min(1.0, 1.0 - penalty))


def compute_uncertainty(goal_scores: list[float] | None = None) -> float:
    """Variance-based uncertainty. With one sample, uncertainty is zero."""
    if not goal_scores or len(goal_scores) < 2:
        return 0.0
    return max(0.0, min(1.0, float(np.var(goal_scores))))


def _pil_to_lpips_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = F.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    return tensor * 2.0 - 1.0


def _pil_to_pixel_tensor(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(size, Image.BICUBIC), dtype=np.float32) / 255.0


class GoalImageScorer:
    """Compute final-frame to goal-image similarity."""

    def __init__(self, metric: str = "auto", device: str | None = None) -> None:
        self.metric = metric
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lpips_model = None
        self.dreamsim_model = None
        self.dreamsim_preprocess = None

        if metric in {"auto", "lpips_dreamsim"}:
            self._try_load_lpips_dreamsim(required=metric == "lpips_dreamsim")

    def _try_load_lpips_dreamsim(self, required: bool) -> None:
        try:
            import lpips  # type: ignore

            self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        except Exception as exc:
            if required:
                raise RuntimeError("LPIPS is required but not available. Install the `lpips` package.") from exc

        try:
            from dreamsim import dreamsim  # type: ignore

            self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True, device=str(self.device))
            self.dreamsim_model.eval()
        except Exception as exc:
            if required:
                raise RuntimeError("DreamSim is required but not available. Install the `dreamsim` package.") from exc

        if self.lpips_model is None or self.dreamsim_model is None:
            self.metric = "pixel"

    def score(self, final_frame: Image.Image, goal_image: Image.Image) -> dict[str, float | str]:
        if self.metric == "pixel":
            return self._score_pixel(final_frame, goal_image)
        return self._score_lpips_dreamsim(final_frame, goal_image)

    def _score_pixel(self, final_frame: Image.Image, goal_image: Image.Image) -> dict[str, float | str]:
        size = (224, 224)
        a = _pil_to_pixel_tensor(final_frame, size)
        b = _pil_to_pixel_tensor(goal_image, size)
        mse = float(np.mean((a - b) ** 2))
        # MSE over [0,1] RGB is in [0,1]; convert to a bounded similarity.
        score = max(0.0, min(1.0, 1.0 - mse))
        return {
            "goal_score": score,
            "lpips_distance": None,
            "lpips_score": None,
            "dreamsim_distance": None,
            "dreamsim_score": None,
            "goal_metric": "pixel_mse_fallback",
        }

    def _score_lpips_dreamsim(self, final_frame: Image.Image, goal_image: Image.Image) -> dict[str, float | str]:
        if self.lpips_model is None or self.dreamsim_model is None or self.dreamsim_preprocess is None:
            raise RuntimeError("LPIPS/DreamSim models are not loaded")

        with torch.inference_mode():
            final_lpips = _pil_to_lpips_tensor(final_frame, self.device)
            goal_lpips = _pil_to_lpips_tensor(goal_image, self.device)
            lpips_distance = float(self.lpips_model(final_lpips, goal_lpips).item())
            lpips_score = max(0.0, min(1.0, 1.0 - lpips_distance))

            final_ds = self.dreamsim_preprocess(final_frame).to(self.device)
            goal_ds = self.dreamsim_preprocess(goal_image).to(self.device)
            dreamsim_distance = float(self.dreamsim_model(final_ds, goal_ds).item())
            dreamsim_score = max(0.0, min(1.0, 1.0 - dreamsim_distance))

        goal_score = 0.5 * lpips_score + 0.5 * dreamsim_score
        return {
            "goal_score": goal_score,
            "lpips_distance": lpips_distance,
            "lpips_score": lpips_score,
            "dreamsim_distance": dreamsim_distance,
            "dreamsim_score": dreamsim_score,
            "goal_metric": "lpips_dreamsim",
        }


def score_generated_videos(
    request: dict[str, Any],
    generated: dict[str, Any],
    output_dir: Path,
    goal_metric: str,
    score_device: str | None,
    max_v: float,
    max_yaw_rate: float,
    max_acc: float,
    min_altitude: float = DEFAULT_MIN_ALTITUDE,
    uncertainty_videos: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Score generated videos and return the best dynamically legal trajectory."""
    goal = request.get("goal") or {}
    goal_image_path = goal.get("goal_image")
    if not goal_image_path:
        raise ValueError("request['goal']['goal_image'] is required for goal-image scoring")

    goal_image = Image.open(goal_image_path).convert("RGB")
    scorer = GoalImageScorer(metric=goal_metric, device=score_device)
    scores: list[dict[str, Any]] = []

    traj_by_id = {str(traj.get("traj_id")): traj for traj in request.get("candidate_trajectories", [])}
    final_frame_dir = output_dir / "final_frames"

    for traj_id, video_path in generated["predicted_videos"].items():
        traj = traj_by_id[traj_id]
        dyn_score, dyn_reject, dyn_details = check_dynamics(traj, max_v, max_yaw_rate, max_acc)

        final_frame_path = final_frame_dir / f"traj_{traj_id}_final.jpg"
        final_frame = extract_final_frame(video_path, final_frame_path)
        goal_scores = scorer.score(final_frame, goal_image)

        goal_score = float(goal_scores["goal_score"])
        uncertainty_goal_scores = [goal_score]
        for sample_idx, sample_video_path in enumerate((uncertainty_videos or {}).get(traj_id, []), start=1):
            sample_final_frame_path = final_frame_dir / f"traj_{traj_id}_sample_{sample_idx}_final.jpg"
            sample_final_frame = extract_final_frame(sample_video_path, sample_final_frame_path)
            sample_goal_scores = scorer.score(sample_final_frame, goal_image)
            uncertainty_goal_scores.append(float(sample_goal_scores["goal_score"]))

        safety_score = compute_safety_score(
            traj=traj,
            uav_state=request.get("uav_state"),
            dynamics_rejected=dyn_reject,
            min_altitude=min_altitude,
        )
        semantic_score = compute_semantic_score(
            pred_video=video_path,
            instruction=str(goal.get("instruction", "")),
            motion_prompt=str(generated["motion_prompts"].get(traj_id, "")),
        )
        temporal_score = compute_temporal_score(video_path)
        uncertainty = compute_uncertainty(uncertainty_goal_scores)
        total_score = (
            0.30 * goal_score
            + 0.25 * safety_score
            + 0.15 * semantic_score
            + 0.10 * temporal_score
            + 0.10 * dyn_score
            - 0.10 * uncertainty
        )
        rejected = bool(dyn_reject or safety_score < 0.3 or semantic_score < 0.2 or uncertainty > 0.7)

        scores.append(
            {
                "traj_id": int(traj_id) if str(traj_id).isdigit() else traj_id,
                "total_score": total_score,
                "goal_score": goal_score,
                "safety_score": safety_score,
                "semantic_score": semantic_score,
                "temporal_score": temporal_score,
                "dynamics_score": dyn_score,
                "uncertainty": uncertainty,
                "rejected": rejected,
            }
        )

    valid_scores = [score for score in scores if not score["rejected"]]
    ranked = sorted(valid_scores, key=lambda item: item["total_score"], reverse=True)

    best_traj_id = ranked[0]["traj_id"] if ranked else None

    return {
        "best_traj_id": best_traj_id,
        "ranked_traj_ids": [item["traj_id"] for item in ranked],
        "scores": scores,
        "predicted_videos": generated["predicted_videos"],
    }


def score_trajectories(
    request: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = DEFAULT_MODEL,
    transformer_path: Path | None = DEFAULT_TRANSFORMER,
    steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    gpu: str | None = "0",
    sequential_offload: bool = True,
    python_bin: str | Path = sys.executable,
    goal_metric: str = "auto",
    score_device: str | None = None,
    max_v: float = DEFAULT_MAX_V,
    max_yaw_rate: float = DEFAULT_MAX_YAW_RATE,
    max_acc: float = DEFAULT_MAX_ACC,
    min_altitude: float = DEFAULT_MIN_ALTITUDE,
    uncertainty_samples: int = DEFAULT_UNCERTAINTY_SAMPLES,
) -> dict[str, Any]:
    """Planner-facing API: score candidate trajectories and return best traj id."""
    generated = generate_candidate_videos(
        request=request,
        output_dir=output_dir,
        model_path=model_path,
        transformer_path=transformer_path,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        gpu=gpu,
        sequential_offload=sequential_offload,
        python_bin=python_bin,
    )
    uncertainty_videos: dict[str, list[str]] = {}
    if uncertainty_samples > 1:
        goal = request.get("goal") or {}
        tail_condition = request.get("tail_condition")
        current_rgb = request["current_rgb"]
        sample_dir = output_dir / "uncertainty_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for traj in request.get("candidate_trajectories", []):
            traj_id = str(traj.get("traj_id"))
            prompt = generated["motion_prompts"][traj_id]
            uncertainty_videos[traj_id] = []
            for sample_idx in range(1, uncertainty_samples):
                sample_output = sample_dir / f"traj_{traj_id}_sample_{sample_idx}_pred.mp4"
                sample_video = airscape_generate_video(
                    current_rgb=current_rgb,
                    motion_prompt=prompt,
                    output_path=sample_output,
                    model_path=model_path,
                    transformer_path=transformer_path,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed + 1000 * sample_idx + (int(traj_id) if traj_id.isdigit() else 0),
                    gpu=gpu,
                    sequential_offload=sequential_offload,
                    python_bin=python_bin,
                )
                uncertainty_videos[traj_id].append(str(sample_video))

    return score_generated_videos(
        request=request,
        generated=generated,
        output_dir=output_dir,
        goal_metric=goal_metric,
        score_device=score_device,
        max_v=max_v,
        max_yaw_rate=max_yaw_rate,
        max_acc=max_acc,
        min_altitude=min_altitude,
        uncertainty_videos=uncertainty_videos,
    )


def airscape_generate_video(
    current_rgb: str | Path,
    motion_prompt: str,
    output_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    transformer_path: str | Path | None = DEFAULT_TRANSFORMER,
    steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    gpu: str | None = "0",
    sequential_offload: bool = True,
    python_bin: str | Path = sys.executable,
) -> Path:
    """Generate one future video with the existing AirScape diffusers runner."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_bin),
        str(ROOT / "tools" / "run_i2v_diffusers.py"),
        "--model",
        str(model_path),
        "--image",
        str(current_rgb),
        "--prompt",
        motion_prompt,
        "--output",
        str(output_path),
        "--steps",
        str(steps),
        "--guidance-scale",
        str(guidance_scale),
        "--seed",
        str(seed),
    ]

    if transformer_path:
        cmd.extend(["--transformer", str(transformer_path)])

    if sequential_offload:
        cmd.append("--sequential-offload")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return output_path


def generate_candidate_videos(
    request: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = DEFAULT_MODEL,
    transformer_path: Path | None = DEFAULT_TRANSFORMER,
    steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    gpu: str | None = "0",
    sequential_offload: bool = True,
    python_bin: str | Path = sys.executable,
) -> dict[str, Any]:
    """Generate one predicted future video for each candidate trajectory."""
    current_rgb = request["current_rgb"]
    goal = request.get("goal") or {}
    tail_condition = request.get("tail_condition")

    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_videos: dict[str, str] = {}
    motion_prompts: dict[str, str] = {}

    for traj in request.get("candidate_trajectories", []):
        traj_id = str(traj.get("traj_id"))
        prompt = trajectory_to_motion_prompt(traj, goal, tail_condition)
        motion_prompts[traj_id] = prompt

        output_path = output_dir / f"traj_{traj_id}_pred.mp4"
        video_path = airscape_generate_video(
            current_rgb=current_rgb,
            motion_prompt=prompt,
            output_path=output_path,
            model_path=model_path,
            transformer_path=transformer_path,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed + int(traj_id) if traj_id.isdigit() else seed,
            gpu=gpu,
            sequential_offload=sequential_offload,
            python_bin=python_bin,
        )
        predicted_videos[traj_id] = str(video_path)

    return {
        "motion_prompts": motion_prompts,
        "predicted_videos": predicted_videos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert candidate trajectories to prompts and optionally generate AirScape videos.")
    parser.add_argument("--request", type=Path, required=True, help="JSON request containing candidate_trajectories.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--generate", action="store_true", help="Run AirScape and generate one video per candidate trajectory.")
    parser.add_argument("--score", action="store_true", help="After generation, score each video against goal_image and select best traj_id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated candidate videos.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CogVideoX-I2V model directory.")
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER, help="AirScape transformer checkpoint directory.")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use 'none' to leave it unchanged.")
    parser.add_argument("--no-sequential-offload", action="store_true", help="Disable sequential CPU offload.")
    parser.add_argument("--goal-metric", choices=["auto", "lpips_dreamsim", "pixel"], default="auto")
    parser.add_argument("--score-device", default=None, help="Device for LPIPS/DreamSim scoring. Defaults to cuda if available.")
    parser.add_argument("--max-v", type=float, default=DEFAULT_MAX_V)
    parser.add_argument("--max-yaw-rate", type=float, default=DEFAULT_MAX_YAW_RATE)
    parser.add_argument("--max-acc", type=float, default=DEFAULT_MAX_ACC)
    parser.add_argument("--min-altitude", type=float, default=DEFAULT_MIN_ALTITUDE)
    parser.add_argument("--uncertainty-samples", type=int, default=DEFAULT_UNCERTAINTY_SAMPLES, help="Number of videos to sample per trajectory for uncertainty variance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))

    if args.score:
        payload = score_trajectories(
            request=request,
            output_dir=args.output_dir,
            model_path=args.model,
            transformer_path=args.transformer,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            gpu=None if args.gpu == "none" else args.gpu,
            sequential_offload=not args.no_sequential_offload,
            goal_metric=args.goal_metric,
            score_device=args.score_device,
            max_v=args.max_v,
            max_yaw_rate=args.max_yaw_rate,
            max_acc=args.max_acc,
            min_altitude=args.min_altitude,
            uncertainty_samples=args.uncertainty_samples,
        )
    elif args.generate:
        payload = generate_candidate_videos(
            request=request,
            output_dir=args.output_dir,
            model_path=args.model,
            transformer_path=args.transformer,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            gpu=None if args.gpu == "none" else args.gpu,
            sequential_offload=not args.no_sequential_offload,
        )
    else:
        payload = {"motion_prompts": prompts_from_request(request)}

    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
