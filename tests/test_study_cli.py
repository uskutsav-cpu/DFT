from dataclasses import replace

import pytest

from refblind.assembly import assemble_table
from refblind.attachments import FeatureAttachment, attach_features
from refblind.cli import main
from refblind.configuration import load_config
from refblind.gmtkn import import_subset, parse_res
from refblind.pathways import audit_transition_state, interpolate_endpoints
from refblind.provenance import freeze_protocol, read_json, write_json
from refblind.review import apply_reference_reviews
from refblind.study import StudyConfig, eligible_rows, evaluate_study, fit_study, run_study
from refblind.synthetic import synthetic_table
from refblind.tables import StudyTable


@pytest.fixture
def small_table():
    return synthetic_table(groups=20, rows_per_group=3, seed=9)


def test_full_study_reproducible(tmp_path, small_table):
    config = StudyConfig(bootstrap_repetitions=20, thresholds=(0.25, 0.75))
    a = run_study(small_table, config, tmp_path / "a")
    b = run_study(small_table, config, tmp_path / "b")
    assert a == b and a["synthetic"] is True
    assert a["test_groups"] >= 2 and a["reference_blind_count"] <= a["test_rows"]
    artifact = read_json(tmp_path / "a-frozen.json")
    assert artifact["test_used_for_selection"] is False
    split = artifact["split"]["assignments"]
    for model in artifact["models"].values():
        assert all(split[i] == "train" for i in model["training_ids"])
        assert all(split[i] == "calibration" for i in model["calibration_ids"])
    assert (tmp_path / "a/REPORT.md").is_file()
    assert "synthetic" in (tmp_path / "a/REPORT.md").read_text().lower()
    assert (tmp_path / "a/policy-decisions.csv").is_file()
    with pytest.raises(FileExistsError):
        run_study(small_table, config, tmp_path / "a")


def test_artifact_rejects_tampering_and_relabelled_data(tmp_path, small_table):
    artifact = fit_study(
        small_table,
        StudyConfig(thresholds=(0.5,), bootstrap_repetitions=10),
        tmp_path / "frozen.json",
    )
    bad = dict(artifact)
    bad["test_used_for_selection"] = True
    with pytest.raises(ValueError):
        evaluate_study(small_table, bad, tmp_path / "no")
    changed = StudyTable(
        [replace(small_table.rows[0], reference=100.0)] + small_table.rows[1:], small_table.metadata
    )
    with pytest.raises(ValueError):
        evaluate_study(changed, artifact, tmp_path / "no")


def test_exclusions_explicit(small_table):
    row = small_table.rows[0]
    bad = replace(row, id="failure", mlip=None)
    dubious = replace(row, id="questionable", reference_quality="questionable")
    table = StudyTable(small_table.rows + [bad, dubious], small_table.metadata)
    rows, excluded = eligible_rows(table, StudyConfig())
    assert len(excluded) == 2 and len(rows) <= len(small_table.rows)


def test_assembly_keeps_published_reference_separate(dataset, calculations):
    mlip = [replace(c, method_id="mlip", fidelity="mlip", cost=0.01) for c in calculations]
    table = assemble_table(dataset, calculations + mlip, mlip_methods=["mlip"], dft_method="dft-a")
    assert table.rows[0].reference == 1.0
    assert table.rows[0].high_level_value is None and not table.rows[0].high_level_supported
    assert table.rows[0].mlip_features["ensemble_std"] is None
    assert table.rows[0].dft_features["fod"] is None


def test_auxiliary_feature_costs_and_overwrite(row):
    clean = replace(row, dft_features={"fod": None}, costs={**row.costs, "diagnostics": 0.0})
    table = StudyTable([clean], {"synthetic": True})
    a = FeatureAttachment(
        row.id,
        "diagnostics",
        {"fod": 1.5},
        {"fod-job": 8.0},
        {"method": "FOD/TPSS", "source_sha256": "0" * 64},
    )
    attached = attach_features(table, [a])
    assert (
        attached.rows[0].dft_features["fod"] == 1.5 and attached.rows[0].costs["diagnostics"] == 8.0
    )
    with pytest.raises(ValueError):
        attach_features(attached, [a])
    with pytest.raises(ValueError):
        FeatureAttachment(
            row.id,
            "screen",
            {"reference_error": 3.0},
            {"a": 1.0},
            {"method": "bad", "source_sha256": "0" * 64},
        )


def test_review_is_explicit(dataset):
    reviews = [
        {
            "observable_id": "barrier1",
            "quality": "reviewed",
            "reviewer": "Test reviewer",
            "note": "Test fixture only, not an experimental review",
            "uncertainty": 0.1,
        }
    ]
    new = apply_reference_reviews(dataset, reviews)
    assert new.observables[0].reference.quality == "reviewed"
    assert new.observables[0].reference.uncertainty == 0.1
    assert dataset.observables[0].reference.quality == "synthetic"
    with pytest.raises(ValueError):
        apply_reference_reviews(dataset, [{**reviews[0], "note": ""}])


@pytest.mark.parametrize(
    "text", ["a: 1\na: 2", "a: .nan", "token: secret", "nested:\n  password: abc", "a: ["]
)
def test_config_rejects_unsafe_or_ambiguous(tmp_path, text):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_config(p)


def test_gmtkn_data_grammar_no_execution(tmp_path):
    text = "$tmer r/$f ts/$f x -1 1 $w 2.0 # synthetic fixture\n"
    assert parse_res("touch /THIS_MUST_NEVER_RUN\n" + text)[0]["reference"] == 2.0
    with pytest.raises(ValueError):
        parse_res("$tmer ../bad/$f ts/$f x -1 1 $w 2.0")
    with pytest.raises(ValueError):
        parse_res("$tmer r/$f ts/$f x -1 1 $w $(rm -rf x)")
    subset = tmp_path / "BH76"
    subset.mkdir()
    (subset / ".res").write_text(text + text)
    for name, length in [("r", 0.74), ("ts", 1.2)]:
        p = subset / name
        p.mkdir()
        (p / "struc.xyz").write_text(f"2\nsynthetic fixture\nH 0 0 0\nH 0 0 {length}\n")
    data = import_subset(tmp_path)
    assert (
        len(data.observables) == 2 and data.observables[0].group_id == data.observables[1].group_id
    )
    assert data.observables[1].metadata["duplicate_of"] == data.observables[0].id
    assert data.observables[0].reference.quality == "unreviewed"
    with pytest.raises(ValueError):
        import_subset(tmp_path, expected_commit="0" * 40)


def test_pathway_guards(molecules):
    a, b = molecules.values()
    with pytest.raises(ValueError):
        interpolate_endpoints(a, b)
    path = interpolate_endpoints(a, b, atom_mapping_verified=True)
    assert (
        len(path) == 7
        and path[0].geometry_hash == a.geometry_hash
        and path[-1].geometry_hash == b.geometry_hash
    )
    assert not audit_transition_state(
        [-200, 100, 200], irc_endpoints_verified=False
    ).verified_transition_state
    assert audit_transition_state(
        [-200, 100, 200], irc_endpoints_verified=True
    ).verified_transition_state
    assert not audit_transition_state(
        [-200, -100, 200], irc_endpoints_verified=True
    ).verified_transition_state
    with pytest.raises(ValueError):
        audit_transition_state([-200, 100], irc_endpoints_verified="false")


def test_freeze_preserves_artifacts(tmp_path):
    f = tmp_path / "data"
    f.write_text("data")
    target = tmp_path / "frozen.json"
    freeze_protocol({"threshold": 1}, [f], target)
    assert target.is_file()
    with pytest.raises(FileExistsError):
        freeze_protocol({"threshold": 1}, [f], target)


def test_cli_doctor_legacy_and_error(capsys, tmp_path):
    assert main(["doctor"]) == 0
    assert "optional" in capsys.readouterr().out
    assert main(["--mlip", "1", "--dft", "2", "--high-level", "4"]) == 0
    assert "surrogate_error" in capsys.readouterr().out
    assert main(["validate", str(tmp_path / "missing")]) == 2


def test_cli_demo(tmp_path, capsys):
    assert (
        main(
            ["demo", "--groups", "12", "--rows-per-group", "2", "--output", str(tmp_path / "demo")]
        )
        == 0
    )
    assert (tmp_path / "demo/REPORT.md").is_file()
    assert "synthetic" in capsys.readouterr().out.lower()


def test_cli_dataset_plan_review(tmp_path, dataset, capsys):
    d = tmp_path / "d.json"
    dataset.save(d)
    c = tmp_path / "c.yaml"
    c.write_text("backend: orca\nmethod_id: test\n")
    plan = tmp_path / "p.json"
    assert main(["validate", str(d)]) == 0
    assert main(["plan", "--dataset", str(d), "--config", str(c), "--output", str(plan)]) == 0
    assert main(["run", "--plan", str(plan), "--output", str(tmp_path / "no")]) == 0
    reviews = tmp_path / "reviews.json"
    write_json(
        reviews,
        [
            {
                "observable_id": "barrier1",
                "quality": "reviewed",
                "reviewer": "Test",
                "note": "Synthetic test only",
            }
        ],
    )
    assert (
        main(
            [
                "review-references",
                "--dataset",
                str(d),
                "--reviews",
                str(reviews),
                "--output",
                str(tmp_path / "reviewed.json"),
            ]
        )
        == 0
    )
