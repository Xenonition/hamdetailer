from __future__ import annotations

import json
import re
from typing import Any


def default_auto_mapping_json() -> str:
    return json.dumps(
        {
            "global_keywords": [
                "1girl",
                "1boy",
                "girl",
                "boy",
                "woman",
                "man",
                "female",
                "male",
                "person",
            ],
            "clusters": [
                {
                    "name": "face",
                    "model": "face_yolov8s.pt",
                    "keywords": [
                        "face",
                        "head",
                        "eyes",
                        "eye",
                        "eyebrow",
                        "eyelash",
                        "iris",
                        "pupil",
                        "gaze",
                        "mouth",
                        "lips",
                        "teeth",
                        "tongue",
                        "smile",
                        "frown",
                        "grin",
                        "smirk",
                        "open mouth",
                        "closed mouth",
                        "surprised",
                        "blush",
                        "angry",
                        "tearing up",
                        "happy",
                        "crying",
                        "tears",
                        "torogao",
                    ],
                    "include_global": True,
                },
                {
                    "name": "breasts",
                    "model": "breasts_seg.pt",
                    "keywords": [
                        "breasts",
                        "breast",
                        "chest",
                        "cleavage",
                        "bra",
                        "bikini",
                    ],
                    "include_global": False,
                },
                {
                    "name": "nipples",
                    "model": "nipples_v2_yolov11s-seg.pt",
                    "keywords": ["nipples", "nipple", "areola"],
                    "include_global": False,
                },
                {
                    "name": "belly",
                    "model": "belly_seg_v2.42_less_groin.pt",
                    "keywords": ["belly", "midriff", "navel", "linea alba"],
                    "include_global": False,
                },
                {
                    "name": "anus",
                    "model": "anus_v4.pt",
                    "keywords": ["anus", "asshole", "butthole"],
                    "include_global": False,
                },
                {
                    "name": "female_genitals",
                    "model": "pussy.pt",
                    "keywords": ["pussy", "cleft of venus", "labia", "clit", "vaginal"],
                    "include_global": False,
                },
                {
                    "name": "male_genitals",
                    "model": "cockAndBallDetection2D_v20.pt",
                    "keywords": [
                        "cock",
                        "penis",
                        "dick",
                        "balls",
                        "testicles",
                        "shaft",
                        "glans",
                    ],
                    "include_global": False,
                },
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def load_auto_mapping(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        raw = default_auto_mapping_json()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def split_prompt_phrases(prompt: str) -> list[str]:
    if not prompt:
        return []
    return [p.strip() for p in re.split(r"\s*,\s*", prompt) if p.strip()]


def normalize_phrase(text: str) -> str:
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text = re.sub(r":\s*-?\d+(?:\.\d+)?", "", text)
    return text.lower().strip()


def phrase_matches_keyword(phrase: str, keyword: str) -> bool:
    if not keyword:
        return False
    phrase_norm = normalize_phrase(phrase)
    keyword_norm = normalize_phrase(keyword)
    if not keyword_norm:
        return False
    if " " in keyword_norm:
        return keyword_norm in phrase_norm
    return re.search(rf"\b{re.escape(keyword_norm)}\b", phrase_norm) is not None


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_auto_prompts(prompt: str, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = mapping.get("clusters", [])
    if not isinstance(clusters, list):
        return []

    phrases = split_prompt_phrases(prompt)
    global_keywords = mapping.get("global_keywords", [])
    if not isinstance(global_keywords, list):
        global_keywords = []

    global_phrases = [
        p for p in phrases if any(phrase_matches_keyword(p, kw) for kw in global_keywords)
    ]

    results = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        model = cluster.get("model")
        keywords = cluster.get("keywords", [])
        include_global = bool(cluster.get("include_global", False))
        if not model or not isinstance(keywords, list):
            continue

        matched_phrases = [
            p for p in phrases if any(phrase_matches_keyword(p, kw) for kw in keywords)
        ]

        if include_global:
            matched_phrases = [*global_phrases, *matched_phrases]

        matched_phrases = unique_preserve_order(matched_phrases)
        if not matched_phrases:
            continue

        results.append(
            {
                "name": cluster.get("name", ""),
                "model": model,
                "prompt": ", ".join(matched_phrases),
            }
        )

    return results
