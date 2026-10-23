"""Allowlisted development evidence. No model imports, inference or training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from src.config.paths import REPO_ROOT
from src.symbolic.relations import Premise, Query
from src.symbolic.solver import SymbolicSolver
from src.training.answer_parser import strict_parse_answer

LABELS = ("Yes", "No", "Undetermined")
RUN = "artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    if not rows or any(r.get("is_synthetic_demo") is True for r in rows):
        raise ValueError("synthetic demonstrations excluded from empirical analytics")
    confusion = {a: {b: 0 for b in (*LABELS, "Invalid")} for a in LABELS}
    for row in rows:
        prediction = row["parsed_answer"]
        confusion[row["ground_truth_label"]][prediction if prediction in LABELS else "Invalid"] += 1
    recall, f1 = {}, {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[a][label] for a in LABELS)
        recall[label] = tp / support if support else 0
        f1[label] = 2 * tp / (support + predicted) if support + predicted else 0
    n = len(rows)
    u = [r for r in rows if r["ground_truth_label"] == "Undetermined"]
    sufficient = [r for r in rows if r["ground_truth_label"] != "Undetermined"]
    return {"n": n, "families": len({r["family_id"] for r in rows}),
            "balanced_accuracy": sum(recall.values()) / 3, "macro_f1": sum(f1.values()) / 3,
            "per_class_recall": recall, "confusion_matrix": confusion,
            "invalid_output_rate": sum(r["parsed_answer"] not in LABELS for r in rows) / n,
            "exact_compliance": sum(r["raw_completion"].strip() in LABELS for r in rows) / n,
            "unsupported_definite_rate": sum(r["parsed_answer"] in ("Yes", "No") for r in u) / len(u),
            "over_abstention_rate": sum(r["parsed_answer"] == "Undetermined" for r in sufficient) / len(sufficient)}


class ArtifactStore:
    def __init__(self, root=REPO_ROOT):
        self.root = Path(root).resolve()
        self.sources = {}
        def read(relative):
            path = self.root / relative
            self.sources[relative] = sha(path)
            return json.loads(path.read_text(encoding="utf-8"))
        ce = read("artifacts/training/phase_2i_a/phase_2i_a_report_v1.json")
        self.architecture = {"text_layers": ce["load_meta"]["text_num_layers"],
                             "hidden_size": ce["load_meta"]["text_hidden_size"],
                             "source": "artifacts/training/phase_2i_a/phase_2i_a_report_v1.json#load_meta",
                             "activation_data_available": False, "causal_circuit_established": False}
        baseline = read(RUN + "/baseline.json")
        pilot = read(RUN + "/evaluation_step_0005.json")
        self.identity = read(RUN + "/run.json")["identity"]
        self.conditions = []
        conditions = [
            ("frozen", "Frozen Qwen", 0, ce["baseline_standard"], "standard"),
            ("prompt", "Frozen + abstention prompt", 0, baseline["abstention_prompt"], "abstention"),
            ("ce5", "CE · step 5", 5, ce["val_step_5"], "standard"),
            ("es5", "CE + ES · step 5", 5, pilot["standard"], "standard"),
            ("ce50", "CE · step 50", 50, ce["val_final"], "standard"),
            ("es5prompt", "CE + ES · step 5 + abstention prompt", 5, pilot["abstention_prompt"], "abstention"),
        ]
        self.rows = {}
        reference = None
        for key, name, step, data, prompt in conditions:
            rows = data["raw"]
            ids = {r["sample_id"] for r in rows}
            if reference is None:
                reference = ids
            if ids != reference or len(rows) != 18 or len(ids) != 18:
                raise ValueError("incompatible development slice")
            for row in rows:
                if strict_parse_answer(row["raw_completion"])["parsed_label"] != row["parsed_answer"]:
                    raise ValueError("stored parser result mismatch")
            self.rows[key] = {r["sample_id"]: r for r in rows}
            self.conditions.append({"id": key, "name": name, "step": step, "prompt": prompt,
                                    "evidence": "OBSERVED BEHAVIOR", "metrics": metrics(rows)})
        # Prefer a portable package containing only these 18 development cases.
        packaged = self.root / "dashboard/assets/development_cases.json"
        if packaged.is_file():
            records = json.loads(packaged.read_text(encoding="utf-8"))
            self.sources["dashboard/assets/development_cases.json"] = sha(packaged)
        else:
            records = []
            with (self.root / self.identity["dataset"]).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["sample_id"] in reference:
                        records.append(row)
        self.cases = {}
        for row in records:
            if row["sample_id"] not in reference or row["split"] != "validation" or row.get("is_synthetic_demo") is not False:
                raise ValueError("only allowlisted real development evidence may enter dashboard")
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("prompt hash mismatch")
            for rows in self.rows.values():
                saved = rows[row["sample_id"]]
                if any(saved[k] != row[k] for k in ("family_id", "ground_truth_label", "prompt_sha256", "image_sha256")):
                    raise ValueError("case identity mismatch")
            solution = SymbolicSolver.solve([Premise(**p) for p in row["premises"]], Query(**row["query"]))
            if solution.epistemic_status.value != row["symbolic_status"] or not solution.algorithm_agreement:
                raise ValueError("symbolic oracle mismatch")
            self.cases[row["sample_id"]] = {**row, "solver": solution.to_dict()}
        if set(self.cases) != reference:
            raise ValueError("missing allowlisted development cases")
        self.training = [json.loads(line) for line in (self.root / RUN / "train_metrics_append.jsonl").read_text().splitlines()]
        self.sources[RUN + "/train_metrics_append.jsonl"] = sha(self.root / RUN / "train_metrics_append.jsonl")
        if [r["step"] for r in self.training] != list(range(1, 6)):
            raise ValueError("expected frozen five-step evidence")

    def image(self, sample_id):
        row = self.cases[sample_id]
        original = (self.root / row["image_file"]).resolve()
        if not original.is_relative_to(self.root):
            raise ValueError("image outside workspace")
        packaged = self.root / "dashboard/assets/images" / (sample_id.replace("::", "__") + ".png")
        path = original if original.is_file() else packaged
        if sha(path) != row["image_sha256"]:
            raise ValueError("image hash mismatch")
        return path

    def workspace(self):
        cases = []
        for sample_id, row in sorted(self.cases.items()):
            cases.append({**row, "predictions": {k: v[sample_id] for k, v in self.rows.items()}})
        return {"schema": "crossmodallab_dashboard_v1", "is_synthetic_demo": False,
                "split": "development-validation", "holdout_used": False,
                "conditions": self.conditions, "cases": cases,
                "model": {"id": self.identity["model_id"], "revision": self.identity["revision"],
                          "quantization": self.identity["quantization"], "lora": self.identity["lora"]},
                "architecture": self.architecture,
                "training": self.training, "run_id": Path(RUN).name, "completed_step": 5,
                "sources": [{"path": p, "sha256": h} for p, h in sorted(self.sources.items())],
                "runtime": self.identity["runtime"], "optimizer": self.identity["optimizer"],
                "scoring": {"candidates": [" Yes", " No", " Undetermined"], "token_counts": [1, 1, 2],
                            "normalization": "summed, not length-normalized", "coefficient": .25, "margin": 1},
                "limitations": ["Repeatedly inspected development data: 18 cases, 6 families; no generalization claim.",
                                "Historical parser accepts label prefixes with malformed suffixes; exact compliance is separate.",
                                "Undetermined spans two tokens; summed candidate scores have token-length effects.",
                                "No verified relation-specific causal circuit. No per-case attention or neuron data in this viewer.",
                                "Architecture and symbolic graphs are explanatory schematics, not measured neural reasoning traces."]}

    def provenance(self):
        def git(*args):
            try:
                return subprocess.check_output(["git", *args], cwd=self.root, text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                return "UNAVAILABLE"
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "working_tree_dirty": bool(git("status", "--porcelain")),
                "mode": "read-only stored evidence; no GPU or live inference", "sources": self.sources}
