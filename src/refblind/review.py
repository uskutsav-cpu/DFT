"""Apply a human's explicit reference review; do not automatically certify labels."""

from __future__ import annotations

from dataclasses import replace

from .datasets import Dataset
from .provenance import digest
from .schema import ReferenceQuality


def apply_reference_reviews(dataset: Dataset, reviews: list[dict]) -> Dataset:
    mapping = {o.id: o for o in dataset.observables}
    seen = set()
    for review in reviews:
        oid = review["observable_id"]
        if oid in seen or oid not in mapping:
            raise ValueError("review IDs must be unique and belong to the dataset")
        seen.add(oid)
        if not review.get("reviewer") or not review.get("note"):
            raise ValueError("reviewer and substantive note are required")
        observable = mapping[oid]
        if observable.reference is None:
            raise ValueError("cannot review a missing reference")
        quality = ReferenceQuality(review["quality"])
        if quality == ReferenceQuality.SYNTHETIC and not dataset.metadata.get("synthetic"):
            raise ValueError("do not relabel real data synthetic through reference review")
        kwargs = {"quality": quality, "review_note": f"{review['reviewer']}: {review['note']}"}
        if "uncertainty" in review:
            # Same unit as this reference, NOT implicitly kcal/mol.
            kwargs["uncertainty"] = review["uncertainty"]
        mapping[oid] = replace(observable, reference=replace(observable.reference, **kwargs))
    digest(reviews)
    return Dataset(
        dataset.name,
        dataset.structures,
        [mapping[o.id] for o in dataset.observables],
        {
            **dataset.metadata,
            "reference_reviews": dataset.metadata.get("reference_reviews", []) + reviews,
            "review_parent_hash": dataset.fingerprint,
        },
    )
