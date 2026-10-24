"""Reader-facing descriptions derived from allowlisted evidence, never new results."""
STATUS = {"entailed": "Supported by the evidence", "contradicted": "Contradicted by the evidence",
          "underdetermined": "Not enough information"}


def name(value):
    return value.replace("_", " ")


def explain_case(case):
    facts = [f'{name(p["subject"])} is left of {name(p["object"])}' for p in case["premises"]]
    prefix = "We know " + "; and ".join(facts) + ". "
    status = case["symbolic_status"]
    if status == "underdetermined":
        return prefix + "Both orders of the queried objects fit these facts. Neither Yes nor No is justified: answer Undetermined."
    if status == "contradicted":
        return prefix + "The evidence chain runs opposite to the question. Strict left-of cannot hold both ways: answer No."
    return prefix + "Following the two evidence arrows establishes the queried relation by transitivity: answer Yes."


def comparison_takeaway(case):
    count = sum(case["predictions"][key]["correct"] for key in ("frozen", "ce5", "ce50", "es5"))
    return ("All four checkpoints answered this case incorrectly." if count == 0 else
            "All four checkpoints gave the correct extracted answer." if count == 4 else
            f"{count} of four checkpoints gave the correct extracted answer.")
