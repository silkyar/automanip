"""Finite, configuration-only follow-up search. Never changes task or evaluator code."""
import copy


BASES = {
    "anchor090": dict(gamma=.9, ent_coef=0.0),
    "gamma0875": dict(gamma=.875, ent_coef=0.0),
    "gamma0925": dict(gamma=.925, ent_coef=0.0),
    "entropy001": dict(gamma=.9, ent_coef=.001),
    "entropy003": dict(gamma=.9, ent_coef=.003),
}


def recipe(base, phase="tune", variant="control"):
    if base not in BASES or phase not in ("tune", "short", "verify"):
        raise ValueError("unregistered recipe")
    if variant not in ("control", "lr2", "epochs4") or (phase=="tune" and variant!="control"):
        raise ValueError("unregistered variant")
    return dict(id=f"{phase}-{base}-{variant}", **BASES[base],
                learning_rate=.0006 if variant=="lr2" else .0003,
                update_epochs=4 if variant=="epochs4" else 8)


def validate_recipe(candidate):
    if type(candidate) is not dict or type(candidate.get("id")) is not str:
        raise ValueError("recipe object required")
    parts=candidate["id"].split("-")
    if len(parts)!=3 or candidate!=recipe(parts[1],parts[0],parts[2]):
        raise ValueError("recipe differs from preregistration")
    if any(type(candidate[k]) is not float for k in ("gamma","ent_coef","learning_rate")):
        raise ValueError("floating point recipe parameters required")
    if type(candidate["update_epochs"]) is not int:raise ValueError("integer epoch count required")
    return candidate


def campaign(phase, base="anchor090", variant="control"):
    if phase=="tune":
        candidates=[recipe(b) for b in BASES if b!="anchor090"]
        hypothesis="Refine discount near 0.9 and test small entropy bonuses, one change at a time."
    elif phase=="short":
        candidates=[recipe(base,phase,v) for v in ("control","lr2","epochs4")]
        hypothesis="At half the transition budget, test faster learning and fewer PPO optimization epochs against a paired control."
    elif phase=="verify":
        candidates=[recipe(base,phase,variant)]
        hypothesis="Repeat the promising half-budget efficiency change at the original full transition budget."
    else:raise ValueError("unknown search phase")
    return dict(campaign_id="stackcube-overnight-20261027-"+phase,hypothesis=hypothesis,
                requested_transitions=12_500_000 if phase=="short" else 25_000_000,
                training_seeds=[1,2,3],candidates=candidates)


def validate_overnight(value):
    phase=value["campaign_id"].removeprefix("stackcube-overnight-20261027-")
    if phase not in ("tune","short","verify"):raise ValueError("unknown overnight campaign")
    candidates=value["candidates"]
    if type(candidates) is not list or not candidates:raise ValueError("candidates required")
    for candidate in candidates:validate_recipe(candidate)
    _,base,variant=candidates[0]["id"].split("-")
    expected=campaign(phase,base,variant)
    if candidates!=expected["candidates"] or value["requested_transitions"]!=expected["requested_transitions"]:
        raise ValueError("unregistered phase/budget/candidate combination")
    return value


def candidate_stats(summary):
    groups={}
    for run in summary["runs"]:
        groups.setdefault(run["candidate"]["id"],[]).append(run)
    result=[]
    for name,runs in groups.items():
        runs=sorted(runs,key=lambda r:r["seed"])
        if [r["seed"] for r in runs]!=[1,2,3] or any(r["status"]!="complete" for r in runs):
            raise ValueError("three completed paired training seeds required")
        rates=[r["final"]["report"]["success_at_end_rate"] for r in runs]
        result.append(dict(candidate=copy.deepcopy(runs[0]["candidate"]),rates=rates,mean=sum(rates)/3,
                           mean_training_seconds=sum(r["trainer_process_seconds"] for r in runs)/3))
    return result


def choose_tuning(summary, anchor):
    choices=[anchor]
    for item in candidate_stats(summary):
        if item["mean"]>=anchor["mean"]+.02 and min(a-b for a,b in zip(item["rates"],anchor["rates"]))>=-.05:
            choices.append(item)
    return max(choices,key=lambda item:item["mean"])


def choose_efficiency(summary):
    items=candidate_stats(summary)
    control=next(item for item in items if item["candidate"]["id"].endswith("-control"))
    promising=[]
    for item in items:
        if item is control:continue
        score_gain=item["mean"]>=control["mean"]+.03
        faster=item["mean"]>=control["mean"]-.03 and item["mean_training_seconds"]<=.9*control["mean_training_seconds"]
        if (score_gain or faster) and min(a-b for a,b in zip(item["rates"],control["rates"]))>=-.10:
            promising.append(item)
    return max(promising,key=lambda item:(item["mean"],-item["mean_training_seconds"])) if promising else None
