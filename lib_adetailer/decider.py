"""Suggest detection models (and each model's slice of the prompt) from a generation prompt.

Two deciders, same output:

* keyword rules — ``auto_mapping.py`` clusters; instant, no dependencies.
* a decision-model server — anything that speaks the Jev / System One wire format
  (``POST /v1/systemone``: a ``state`` plus typed ``questions``, answers come back as
  probabilities), e.g. ``laya-serve`` from forge-neo-decider-lab, Kev, JevK5 or hosted Jev.
  It runs in its own process: those models need newer torch/transformers than Forge's venv.

Standard library only, and no imports from the rest of ``lib_adetailer`` at module level, so
the decider lab can load this file by path and drive it with an in-process model.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

# ---------------------------------------------------------------------------
#  Regions: what a detector finds, in booru terms, and the model files for it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    key: str
    description: str
    models: tuple[str, ...]  # preference order; the first one installed is used

    def pick_model(self, available: Iterable[str]) -> str | None:
        available = set(available)
        return next((m for m in self.models if m in available), None)


# Keys match the auto_mapping cluster names where both exist.
REGIONS: tuple[Region, ...] = (
    Region("face", "a character's face: eyes, mouth, expression", (
        "face_yolov8s.pt", "face_yolov8n.pt", "AnzhcFace.pt", "FacesV1.pt")),
    Region("eyes", "eyes in close detail: iris, pupils, eyelashes, eye color", (
        "Anzhceyes.pt", "PitEyeDetailer-v2-seg.pt")),
    Region("mouth", "mouth in close detail: lips, teeth, tongue, open mouth", (
        "adetailer2dMouth_v10.pt",)),
    Region("hair", "hair and hairstyle", ("HairOnly.pt",)),
    Region("hands", "hands and fingers, things held in hands, hand gestures", (
        "hand_yolov8s.pt", "animehand.pt", "hand_yolov8n.pt")),
    Region("feet", "feet, toes, soles, shoes, footwear", ("adetailerFootYolov8x_v20.pt",)),
    Region("person", "a whole character or person's body", (
        "person_yolov8s-seg.pt", "anime_person.pt", "yolo12l-person-seg-extended.pt")),
    Region("breasts", "breasts, chest, cleavage", ("breasts_seg.pt", "Anzhcbreasts.pt")),
    Region("nipples", "nipples, areola", (
        "nipples_v2_yolov11s-seg.pt", "ntd11_anime_nsfw_segm_v4_nipples.pt")),
    Region("belly", "belly, midriff, navel, stomach", (
        "belly_seg_v2.42_less_groin.pt", "navel_best.pt")),
    Region("armpit", "armpits, arms raised", ("armpit_seg_v1.1.pt",)),
    Region("butt", "buttocks, ass, hips from behind", ("assdetailer-seg.pt",)),
    Region("anus", "anus", ("anus_v4.pt", "ntd11_anime_nsfw_segm_v4_anus.pt")),
    Region("female_genitals", "vulva, pussy, female genitals", (
        "pussy_yolo11s_seg_best.pt", "pussy.pt", "ntd11_anime_nsfw_segm_v4_pussy.pt")),
    Region("pubic_hair", "female pubic hair", ("female_pubic_hair_detector_v1.0.pt",)),
    Region("male_genitals", "penis, testicles, male genitals", (
        "cockAndBallDetection2D_v20.pt", "penis.pt", "ntd11_anime_nsfw_segm_v4_penis.pt")),
    Region("bra", "bra, bikini top, strapless top", ("anime_bra_strapless-seg_v10.pt",)),
    Region("panties", "panties, underwear bottoms", ("womensUnderwear_pantiesSegV3b.pt",)),
    Region("top_clothes", "shirt, top, upper-body clothing", ("female_clothes_top_seg.pt",)),
    Region("clothing", "clothing and outfit in general", ("clothing1.pt",)),
)
REGION_BY_KEY = {r.key: r for r in REGIONS}


def region_for_model(model: str) -> str | None:
    return next((r.key for r in REGIONS if model in r.models), None)


# ---------------------------------------------------------------------------
#  Prompt -> tags
# ---------------------------------------------------------------------------

_LORA_RE = re.compile(r"<[^<>]*>")
_WEIGHT_RE = re.compile(r":\s*-?\d+(?:\.\d+)?\s*(?=[)\]])")
QUALITY_TAGS = {
    "masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic",
    "absurdres", "highres", "newest", "good quality", "normal quality",
}


def split_tags(prompt: str) -> list[str]:
    """Comma-separated tags with LoRA calls, emphasis weights, brackets and quality tags removed.

    ``(small penis, testicles:0.6)`` -> ``small penis``, ``testicles``.
    """
    text = _LORA_RE.sub(",", prompt or "")
    text = _WEIGHT_RE.sub("", text)
    text = re.sub(r"[()\[\]{}]", "", text)
    tags, seen = [], set()
    for raw in re.split(r"[,\n]", text):
        tag = " ".join(raw.split())
        key = tag.lower()
        if not tag or key in QUALITY_TAGS or key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
#  Deciders
# ---------------------------------------------------------------------------


@dataclass
class Suggestion:
    region: str
    model: str
    score: float
    tags: list[str] = field(default_factory=list)


@dataclass
class Suggestions:
    decider: str
    items: list[Suggestion]
    scores: dict[str, float] = field(default_factory=dict)  # every region asked, not only chosen
    notes: list[str] = field(default_factory=list)


def keyword_suggest(prompt: str, available: Iterable[str]) -> Suggestions:
    """hamdetailer's keyword clusters (auto_mapping.py)."""
    from lib_adetailer.auto_mapping import build_auto_prompts, load_auto_mapping

    available = set(available)
    items, notes = [], []
    for hit in build_auto_prompts(prompt, load_auto_mapping(None)):
        model = hit["model"]
        if model not in available:
            region = REGION_BY_KEY.get(hit["name"])
            model = region.pick_model(available) if region else None
            if not model:
                notes.append(f"{hit['name']}: no installed model")
                continue
        items.append(Suggestion(hit["name"], model, 1.0, split_tags(hit["prompt"])))
    return Suggestions("keyword rules", items, notes=notes)


class SystemOneClient(Protocol):
    label: str

    def run(self, states: list[Any], questions: dict[str, Any]) -> list[dict[str, Any]]: ...


class SystemOneHTTP:
    """POST /v1/systemone, one request per state."""

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 60):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.label = f"{base_url}" + (f" ({model})" if model else "")

    def run(self, states, questions):
        out = []
        for state in states:
            body: dict[str, Any] = {"state": state, "questions": questions}
            if self.model:
                body["model"] = self.model
            req = urllib.request.Request(
                self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out.append(json.load(resp))
        return out


class DecisionModelDecider:
    """Two typed-decision steps over a System One client.

    1. Regions: one yes/no question per installed region, in chunks so none is truncated.
    2. Tags: per prompt tag, one choice among the chosen regions plus "none".
    """

    def __init__(self, client: SystemOneClient, threshold: float = 0.5, chunk: int = 6):
        self.client = client
        self.threshold = threshold
        self.chunk = chunk

    def region_scores(self, tags: list[str], regions: list[Region]) -> tuple[dict[str, float], list[str]]:
        state = ", ".join(tags)
        scores, notes = {}, []
        for i in range(0, len(regions), self.chunk):
            group = regions[i:i + self.chunk]
            questions = {r.key: {
                "type": "noul",
                "instructions": "This is a text-to-image prompt (booru tags). Will the generated "
                                f"image clearly show {r.description}?",
            } for r in group}
            res = self.client.run([state], questions)[0]
            notes += [f"truncated: {q}" for q in res.get("usage", {}).get("truncated_questions", [])]
            scores.update({k: float(v["noul"]) for k, v in res["answers"].items()})
        return scores, notes

    def assign_tags(self, tags: list[str], chosen: list[Region]) -> dict[str, list[str]]:
        criteria = {r.key: r.description for r in chosen}
        criteria["none"] = "the whole image, character identity, pose, setting, style or quality"
        question = {"region": {
            "type": "choice",
            "instructions": "Which body part or item does this image tag describe?",
            "criteria": criteria,
        }}
        by_region: dict[str, list[str]] = {r.key: [] for r in chosen}
        for tag, res in zip(tags, self.client.run(tags, question)):
            pick = res["answers"]["region"]["choice"]
            if pick in by_region:
                by_region[pick].append(tag)
        return by_region

    def suggest(self, prompt: str, available: Iterable[str]) -> Suggestions:
        available = set(available)
        tags = split_tags(prompt)
        regions = [r for r in REGIONS if r.pick_model(available)]
        if not tags or not regions:
            return Suggestions(self.client.label, [])
        scores, notes = self.region_scores(tags, regions)
        chosen = sorted((r for r in regions if scores[r.key] >= self.threshold),
                        key=lambda r: -scores[r.key])
        by_region = self.assign_tags(tags, chosen) if chosen else {}
        items = [Suggestion(r.key, r.pick_model(available), scores[r.key], by_region[r.key])
                 for r in chosen]
        return Suggestions(self.client.label, items, scores, notes)


# ---------------------------------------------------------------------------
#  Passes -> prompts
# ---------------------------------------------------------------------------


def prompts_for_passes(passes: list[dict], labels: list[str], suggestions: list[dict]) -> list[str]:
    """Prompt per pass: the suggested tags of every region its detections came from.

    ``labels`` are the detection model names (detection_state["labels"]); ``suggestions``
    are Suggestion dicts. A pass whose models were not suggested gets "".
    """
    tags_by_model: dict[str, list[str]] = {}
    for s in suggestions:
        # a detector picked by hand for the same region still gets the region's tags
        region = REGION_BY_KEY.get(s["region"])
        for m in {s["model"], *(region.models if region else ())}:
            tags_by_model.setdefault(m, []).extend(s["tags"])
    out = []
    for p in passes:
        tags: list[str] = []
        for i in p.get("det_indices", []):
            for t in tags_by_model.get(labels[i], []) if i < len(labels) else []:
                if t not in tags:
                    tags.append(t)
        out.append(", ".join(tags))
    return out
