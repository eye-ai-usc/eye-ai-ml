"""Unit tests for EyeAI's pure/static helper functions.

These functions contain the genuine eye-ai domain logic (latest-observation
selection, 24-2 HVF priority, ICD10 condition mapping, closest-date
cross-modality matching, ROC computation). They operate on pandas DataFrames
and need no catalog connection, so they are tested here with synthetic
fixtures. Catalog/bag methods are covered by the dev integration tests in
``test_eye_ai.py``.
"""

import pandas as pd
import pytest

from eye_ai.eye_ai import EyeAI


class _FakeBag:
    """Minimal stand-in for a DatasetBag: serves table rows from a dict."""

    def __init__(self, tables: dict[str, list[dict]], dataset_rid: str = "1-TEST"):
        self._tables = tables
        self.dataset_rid = dataset_rid

    def get_table_as_dict(self, table: str):
        return iter(self._tables.get(table, []))


class TestFilterAngle2:
    def test_keeps_only_image_angle_2(self):
        bag = _FakeBag(
            {
                "Image": [
                    {"RID": "I1", "Image_Angle": "2"},
                    {"RID": "I2", "Image_Angle": "1"},
                    {"RID": "I3", "Image_Angle": "2"},
                ]
            }
        )
        result = EyeAI.filter_angle_2(bag)
        assert set(result["RID"]) == {"I1", "I3"}
        assert (result["Image_Angle"] == "2").all()


class TestFindLatestObservation:
    def test_keeps_only_latest_encounter_per_subject(self):
        df = pd.DataFrame(
            {
                "Subject_RID": ["S1", "S1", "S2"],
                "date_of_encounter": ["2020-01-01", "2021-06-01", "2019-03-03"],
            }
        )
        result = EyeAI._find_latest_observation(df.copy())
        # S1 keeps only its 2021 row; S2 keeps its single row.
        assert len(result) == 2
        s1 = result[result["Subject_RID"] == "S1"]
        assert s1["date_of_encounter"].tolist() == ["2021-06-01"]

    def test_single_observation_unchanged(self):
        df = pd.DataFrame({"Subject_RID": ["S1"], "date_of_encounter": ["2020-01-01"]})
        result = EyeAI._find_latest_observation(df.copy())
        assert len(result) == 1


class TestComputeDiagnosis:
    def test_aggregates_per_image_and_rounds_cdr(self):
        df = pd.DataFrame(
            {
                "Image_RID": ["I1", "I1", "I2"],
                "Cup_Disk_Ratio": ["0.51234", "0.61234", "0.7"],
                "Diagnosis_Image": ["No Glaucoma", "Suspected Glaucoma", "No Glaucoma"],
                "Image_Quality": ["Good", "Good", "Bad"],
            }
        )
        # Simple aggregators: mean CDR, first diagnosis, first quality.
        result = EyeAI.compute_diagnosis(
            df,
            diag_func=lambda s: s.iloc[0],
            cdr_func="mean",
            image_quality_func=lambda s: s.iloc[0],
        )
        assert isinstance(result, list)
        by_image = {r["Image_RID"]: r for r in result}
        # CDR rounded to 4 decimals: mean(0.51234, 0.61234) = 0.56234
        assert by_image["I1"]["Cup_Disk_Ratio"] == pytest.approx(0.5623, abs=1e-4)
        assert by_image["I2"]["Cup_Disk_Ratio"] == pytest.approx(0.7, abs=1e-4)


class TestReshapeTable:
    def test_long_and_wide_shapes(self):
        frame = pd.DataFrame(
            {
                "Image_RID": ["I1", "I1"],
                "Image_Side": ["Left", "Left"],
                "Subject_RID": ["S1", "S1"],
                "Full_Name": ["GraderA", "GraderB"],
                "Image_Quality": ["Good", "Good"],
                "Diagnosis_Image": ["No Glaucoma", "Suspected Glaucoma"],
                "Cup_Disk_Ratio": [0.4, 0.6],
            }
        )
        long, wide = EyeAI.reshape_table([frame], compare_value="Cup_Disk_Ratio")
        assert len(long) == 2
        # wide pivots Full_Name into columns
        assert "GraderA" in wide.columns and "GraderB" in wide.columns
        assert wide.loc[("I1", "Left", "S1"), "GraderA"] == 0.4
        assert wide.loc[("I1", "Left", "S1"), "GraderB"] == 0.6


class TestSelect242:
    def test_prefers_24_2_over_other_field_sizes(self):
        hvf = pd.DataFrame(
            {
                "OCR_HVF.RID": ["H1", "H2"],
                "Observation.RID": ["O1", "O1"],
                "OCR_HVF.Image_Side": ["Left", "Left"],
                "OCR_HVF.Field_Size": ["30-2", "24-2"],
            }
        )
        result = EyeAI._select_24_2(hvf)
        # Only one row per (Observation, Side); 24-2 wins.
        assert len(result) == 1
        assert result.iloc[0]["OCR_HVF.Field_Size"] == "24-2"

    def test_passthrough_when_no_ocr_rid_column(self):
        df = pd.DataFrame({"foo": [1, 2]})
        result = EyeAI._select_24_2(df)
        assert result.equals(df)


class TestComputeConditionLabel:
    def test_icd10_mapping_and_priority(self):
        icd = pd.DataFrame(
            {
                "RID": ["R1", "R2", "R3"],
                "Clinical_Records": ["C1", "C1", "C2"],
                "ICD10_Eye": ["H40.11", "H40.00", "H35.0"],
            }
        )
        # compute_condition_label is an instance method but uses no self;
        # call it via the class with a throwaway first arg.
        out = EyeAI.compute_condition_label(object(), icd.copy())
        by_record = dict(zip(out["Clinical_Records"], out["Condition_Label"]))
        # C1 has POAG (H40.11) and GS (H40.00); POAG has higher priority (2<3).
        assert by_record["C1"] == "POAG"
        # H35.0 maps to Other.
        assert by_record["C2"] == "Other"


class TestClosestToFundus:
    def test_matches_closest_date_per_subject_and_side(self):
        report = pd.DataFrame(
            {
                "Subject.RID": ["S1", "S1"],
                "OCR_HVF.Image_Side": ["Left", "Left"],
                "Observation.Date_of_Encounter": ["2020-01-01", "2020-12-01"],
                "OCR_HVF.MD": [-2.0, -5.0],
            }
        )
        fundus = pd.DataFrame(
            {
                "Subject.RID": ["S1"],
                "Observation.Date_of_Encounter": ["2020-11-15"],
            }
        )
        matched = EyeAI.closest_to_fundus(report, fundus, side_col="OCR_HVF.Image_Side")
        # 2020-12-01 is closer to 2020-11-15 than 2020-01-01.
        assert len(matched) == 1
        assert matched.iloc[0]["OCR_HVF.MD"] == -5.0


class TestPlotRoc:
    def test_writes_roc_csv(self, tmp_path):
        # Minimal prediction CSV.
        pred = pd.DataFrame(
            {
                "True Label": [0, 0, 1, 1],
                "Probability Score": [0.1, 0.4, 0.35, 0.8],
            }
        )
        data_csv = tmp_path / "pred.csv"
        pred.to_csv(data_csv, index=False)

        roc_csv = tmp_path / "roc_plot.csv"

        class FakeExecution:
            def asset_file_path(self, asset_name, file_name):
                return roc_csv

        # plot_roc uses self only for nothing catalog-related; pass a bare object.
        import matplotlib
        matplotlib.use("Agg")  # headless

        EyeAI.plot_roc(object(), FakeExecution(), str(data_csv))
        assert roc_csv.exists()
        out = pd.read_csv(roc_csv)
        assert "False Positive Rate" in out.columns
        assert "True Positive Rate" in out.columns


class TestInsertConditionLabel:
    """Unit-test the prep logic + that it inserts via _domain_path() (the
    post-migration accessor), without touching a live catalog."""

    def test_renames_and_inserts_via_domain_path(self):
        from unittest.mock import MagicMock

        condition_label = pd.DataFrame(
            {"Clinical_Records": ["C1", "C2"], "Condition_Label": ["POAG", "GS"]}
        )

        ai = EyeAI.__new__(EyeAI)  # bypass __init__/connection
        mock_path = MagicMock()
        ai._domain_path = MagicMock(return_value=mock_path)

        EyeAI.insert_condition_label(ai, condition_label)

        # Used the migrated _domain_path() accessor (not the old property).
        ai._domain_path.assert_called_once()
        # Inserted records with 'Clinical_Records' renamed to 'RID'.
        inserted = mock_path.Clinical_Records.insert.call_args[0][0]
        assert all("RID" in row and "Clinical_Records" not in row for row in inserted)
        assert {row["RID"] for row in inserted} == {"C1", "C2"}
