
"""Biologically directional interpretation rules for STX-LBD hypotheses.

The rules use entity types and hypothesis classes rather than the display/query
order. They describe associations conservatively and do not claim causality or
experimental confirmation.
"""

from __future__ import annotations

import re
from typing import Iterable


TYPE_ALIASES = {
    "GENE": "SXT_GENE",
    "SXT_GENE": "SXT_GENE",
    "SXT GENE": "SXT_GENE",
    "DINO_TAXON": "DINO_TAXON",
    "DINOPHYCEAE": "DINO_TAXON",
    "DINO TAXON": "DINO_TAXON",
    "CYANO_TAXON": "CYANO_TAXON",
    "CYANO TAXON": "CYANO_TAXON",
    "TAXON": "TAXON",
    "TOXIN": "TOXIN",
    "ENV_FACTOR": "ENV_FACTOR",
    "ENV FACTOR": "ENV_FACTOR",
    "ENVIRONMENT": "ENV_FACTOR",
    "ENVIRONMENTAL FACTOR": "ENV_FACTOR",
    "BIOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
    "BIOLOGICAL PROCESS": "BIOLOGICAL_PROCESS",
    "PROCESS": "BIOLOGICAL_PROCESS",
    "DETECTION_METHOD": "DETECTION_METHOD",
    "DETECTION METHOD": "DETECTION_METHOD",
    "METHOD": "DETECTION_METHOD",
    "EVOLUTIONARY_TERM": "EVOLUTIONARY_TERM",
    "EVOLUTION": "EVOLUTIONARY_TERM",
    "UNKNOWN": "UNKNOWN",
    "": "UNKNOWN",
}


def _clean_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalise_type(value: object) -> str:
    text = _clean_label(value).upper()
    return TYPE_ALIASES.get(text, text or "UNKNOWN")


def _normalise_class(value: object) -> str:
    text = _clean_label(value).casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _bridge_list(value: object, limit: int = 5) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        raw_items: Iterable[object] = value
    else:
        raw_items = re.split(r"[;,|]", str(value))

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        label = _clean_label(item)
        key = label.casefold()

        if not label or label.casefold() in {"nan", "none", "not available", "-"}:
            continue

        if key not in seen:
            seen.add(key)
            cleaned.append(label)

        if len(cleaned) >= limit:
            break

    return cleaned


def _bridge_sentence(bridge_nodes: object) -> str:
    bridges = _bridge_list(bridge_nodes)

    if not bridges:
        return ""

    if len(bridges) == 1:
        formatted = bridges[0]
    elif len(bridges) == 2:
        formatted = f"{bridges[0]} and {bridges[1]}"
    else:
        formatted = ", ".join(bridges[:-1]) + f", and {bridges[-1]}"

    return (
        " In the historical knowledge graph, this association is connected "
        f"through bridge concepts including {formatted}."
    )


def _pick(
    entity_a: str,
    type_a: str,
    entity_b: str,
    type_b: str,
    wanted_type: str,
) -> tuple[str | None, str | None]:
    """Return (wanted entity, other entity), independent of input order."""
    if type_a == wanted_type:
        return entity_a, entity_b
    if type_b == wanted_type:
        return entity_b, entity_a
    return None, None


def _pick_taxon(
    entity_a: str,
    type_a: str,
    entity_b: str,
    type_b: str,
) -> tuple[str | None, str | None]:
    taxon_types = {"DINO_TAXON", "CYANO_TAXON", "TAXON"}

    if type_a in taxon_types:
        return entity_a, entity_b
    if type_b in taxon_types:
        return entity_b, entity_a
    return None, None


def generate_biological_interpretation(
    entity_a: object,
    type_a: object,
    entity_b: object,
    type_b: object,
    hypothesis_class: object = "",
    bridge_nodes: object = "",
) -> str:
    """Generate a conservative, biologically directional interpretation.

    Parameters are order-independent. For example, sxtA–salinity and
    salinity–sxtA both produce an environmental-to-gene interpretation.
    """
    a = _clean_label(entity_a)
    b = _clean_label(entity_b)
    ta = _normalise_type(type_a)
    tb = _normalise_type(type_b)
    hclass = _normalise_class(hypothesis_class)
    bridge = _bridge_sentence(bridge_nodes)

    if not a or not b:
        return "A biological interpretation could not be generated because an entity label is missing."

    # --------------------------------------------------------
    # Hypothesis-class rules: these take priority over row order.
    # --------------------------------------------------------
    if hclass in {
        "environment_gene_regulation",
        "environment_gene_association",
        "environmental_gene_regulation",
    }:
        env, _ = _pick(a, ta, b, tb, "ENV_FACTOR")
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        if env and gene:
            return (
                f"{env.capitalize()} may influence the expression, regulation, or "
                f"activity of {gene}, potentially affecting saxitoxin biosynthesis."
                + bridge
            )

    if hclass in {
        "taxon_gene_association",
        "species_gene_association",
    }:
        taxon, _ = _pick_taxon(a, ta, b, tb)
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        if taxon and gene:
            return (
                f"The association between {taxon} and {gene} suggests that this taxon "
                "may possess, express, or regulate the gene as part of its saxitoxin-related "
                "genetic capacity; functional involvement requires experimental confirmation."
                + bridge
            )

    if hclass in {
        "gene_process_association",
        "gene_biological_process_association",
    }:
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        if gene and process:
            return (
                f"{gene} is associated with {process}, suggesting that the gene may "
                "participate in, or be regulated during, this biological process in the "
                "context of saxitoxin biology."
                + bridge
            )

    if hclass in {
        "process_toxin_association",
        "biological_process_toxin_association",
    }:
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        if process and toxin:
            return (
                f"{process.capitalize()} is associated with {toxin}, suggesting a possible "
                "role in its biosynthesis, transformation, accumulation, or biological effects."
                + bridge
            )

    if hclass in {
        "taxon_toxin_association",
        "species_toxin_association",
    }:
        taxon, _ = _pick_taxon(a, ta, b, tb)
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        if taxon and toxin:
            return (
                f"The occurrence or production of {toxin} is associated with {taxon}, "
                "which may indicate taxon-specific toxin-producing capacity or toxin composition."
                + bridge
            )

    if hclass in {
        "environment_toxin_association",
        "environmental_toxin_association",
    }:
        env, _ = _pick(a, ta, b, tb, "ENV_FACTOR")
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        if env and toxin:
            return (
                f"{env.capitalize()} may influence the production, accumulation, composition, "
                f"or environmental dynamics of {toxin}."
                + bridge
            )

    if hclass in {
        "cross_taxa_association",
        "cross_taxa_transfer_signal",
        "cyano_gene_transfer_signal",
        "cyanobacteria_to_dinoflagellate_transfer",
    }:
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        taxon, _ = _pick_taxon(a, ta, b, tb)

        if gene and taxon:
            core = (
                f"The association between {taxon} and {gene} represents a cross-taxa "
                "knowledge signal that may reflect conserved or independently recurring "
                "saxitoxin-related biology across cyanobacteria and dinoflagellates."
            )
        else:
            core = (
                f"The relationship between {a} and {b} represents a cross-taxa knowledge "
                "signal shared across cyanobacterial and dinoflagellate literature."
            )

        return (
            core
            + " This semantic transfer signal should not, by itself, be interpreted as "
              "evidence of horizontal gene transfer."
            + bridge
        )

    # --------------------------------------------------------
    # Entity-type rules: used when a class is absent or broad.
    # --------------------------------------------------------
    pair = {ta, tb}

    if pair == {"ENV_FACTOR", "SXT_GENE"}:
        env, _ = _pick(a, ta, b, tb, "ENV_FACTOR")
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        return (
            f"{env.capitalize()} may influence the expression, regulation, or activity "
            f"of {gene}, potentially affecting saxitoxin biosynthesis."
            + bridge
        )

    if "SXT_GENE" in pair and pair & {"DINO_TAXON", "CYANO_TAXON", "TAXON"}:
        taxon, _ = _pick_taxon(a, ta, b, tb)
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        return (
            f"The association between {taxon} and {gene} suggests taxon-specific "
            "presence, expression, or regulation of this saxitoxin-related gene."
            + bridge
        )

    if pair == {"SXT_GENE", "TOXIN"}:
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        return (
            f"{gene} may contribute to the biosynthesis, modification, or composition "
            f"of {toxin}; the predicted association does not establish a direct enzymatic role."
            + bridge
        )

    if pair == {"BIOLOGICAL_PROCESS", "SXT_GENE"}:
        gene, _ = _pick(a, ta, b, tb, "SXT_GENE")
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        return (
            f"{gene} is associated with {process}, suggesting possible participation "
            "in, or regulation during, this biological process."
            + bridge
        )

    if pair == {"ENV_FACTOR", "TOXIN"}:
        env, _ = _pick(a, ta, b, tb, "ENV_FACTOR")
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        return (
            f"{env.capitalize()} may influence the production, accumulation, composition, "
            f"or environmental occurrence of {toxin}."
            + bridge
        )

    if "TOXIN" in pair and pair & {"DINO_TAXON", "CYANO_TAXON", "TAXON"}:
        taxon, _ = _pick_taxon(a, ta, b, tb)
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        return (
            f"{toxin} is associated with {taxon}, suggesting possible toxin production, "
            "accumulation, or taxon-specific toxin composition."
            + bridge
        )

    if pair == {"BIOLOGICAL_PROCESS", "TOXIN"}:
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        toxin, _ = _pick(a, ta, b, tb, "TOXIN")
        return (
            f"{process.capitalize()} is associated with {toxin}, suggesting a possible "
            "connection to toxin biosynthesis, transformation, accumulation, or effects."
            + bridge
        )

    if "DETECTION_METHOD" in pair:
        method, other = _pick(a, ta, b, tb, "DETECTION_METHOD")
        return (
            f"{method} may be used to detect, quantify, or characterize {other} "
            "within saxitoxin research."
            + bridge
        )

    if pair == {"ENV_FACTOR", "BIOLOGICAL_PROCESS"}:
        env, _ = _pick(a, ta, b, tb, "ENV_FACTOR")
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        return (
            f"{env.capitalize()} may influence {process} in saxitoxin-producing organisms "
            "or harmful algal bloom systems."
            + bridge
        )

    if "BIOLOGICAL_PROCESS" in pair and pair & {"DINO_TAXON", "CYANO_TAXON", "TAXON"}:
        taxon, _ = _pick_taxon(a, ta, b, tb)
        process, _ = _pick(a, ta, b, tb, "BIOLOGICAL_PROCESS")
        return (
            f"{process.capitalize()} is associated with {taxon}, suggesting a "
            "taxon-specific physiological, ecological, or molecular role."
            + bridge
        )

    if pair == {"TOXIN"}:
        return (
            f"{a} and {b} are associated toxin entities, which may reflect shared "
            "biosynthetic origins, interconversion, co-occurrence, or related toxicological profiles."
            + bridge
        )

    if pair == {"SXT_GENE"}:
        return (
            f"{a} and {b} are associated saxitoxin-related genes, suggesting coordinated, "
            "complementary, or evolutionarily linked roles in the biosynthetic system."
            + bridge
        )

    # Safe fallback: explicitly nondirectional.
    return (
        f"{a} and {b} are predicted to be biologically associated in the STX-LBD "
        "knowledge graph. The available semantic evidence supports further investigation "
        "but does not establish directionality or causality."
        + bridge
    )
