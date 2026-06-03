from pathlib import Path
from typing import List, Callable
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

import numpy as np
import logging

from deriva_ml import DerivaML, DerivaMLException
from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import ML_SCHEMA
from deriva_ml.dataset import DatasetBag
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution import Execution

class EyeAIException(DerivaMLException):
    def __init__(self, msg=""):
        super().__init__(msg=msg)


class EyeAI(DerivaML):
    """Domain subclass of :class:`deriva_ml.DerivaML` for EYE-AI catalogs.

    EyeAI adds eye-ai-specific analytics over fundus images, HVF/RNFL reports,
    and clinical records. It is a faithful pass-through over ``DerivaML``: the
    constructor only supplies the eye-ai default ``hostname`` and ``catalog_id``.
    Catalog connection, path building, and the five DerivaML abstractions
    (Dataset, Workflow, Execution, Feature, Asset) are inherited unchanged.

    General data-prep (image cropping, RETFound directory layout, SVG
    bounding-box parsing) has been moved out of this class into the
    ``data-curation`` repository — it is not eye-ai domain logic. Wide-table
    construction delegates to deriva-ml's denormalization routines
    (``DatasetBag.get_denormalized_as_dataframe``).

    Method groups:
    - Tall/wide image analytics: ``image_tall``, ``reshape_table``,
      ``compute_diagnosis``, ``filter_angle_2``.
    - Multimodal assembly: ``extract_modality``, ``multimodal_wide``,
      ``severity_analysis``, ``multimodal_wide_from_subject_bag``,
      ``add_multimodal_measurements``.
    - Clinical labeling: ``compute_condition_label``, ``insert_condition_label``.
    - Evaluation: ``plot_roc``.
    - Domain matching helpers (static/private): ``_find_latest_observation``,
      ``_select_24_2``, ``closest_to_fundus``.
    """

    def __init__(self, hostname: str = 'www.eye-ai.org',
                 catalog_id: str = 'eye-ai',
                 domain_schemas: str | set[str] | None = None,
                 default_schema: str | None = None,
                 project_name: str | None = None,
                 cache_dir: str | Path | None = None,
                 working_dir: str | Path | None = None,
                 hydra_runtime_output_dir: str | Path | None = None,
                 ml_schema: str = ML_SCHEMA,
                 logging_level=logging.WARNING,
                 deriva_logging_level=logging.WARNING,
                 credential: dict | None = None,
                 s3_bucket: str | None = None,
                 use_minid: bool | None = None,
                 clean_execution_dir: bool = True,
                 mode: ConnectionMode | str = ConnectionMode.online):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_id (str): The catalog number or name.
        """

        super().__init__(
            hostname=hostname,
            catalog_id=catalog_id,
            domain_schemas=domain_schemas,
            default_schema=default_schema,
            ml_schema=ml_schema,
            project_name=project_name,
            cache_dir=cache_dir,
            working_dir=working_dir,
            hydra_runtime_output_dir=hydra_runtime_output_dir,
            logging_level=logging_level,
            deriva_logging_level=deriva_logging_level,
            credential=credential,
            s3_bucket=s3_bucket,
            use_minid=use_minid,
            clean_execution_dir=clean_execution_dir,
            mode=mode
        )

    @staticmethod
    def _find_latest_observation(df: pd.DataFrame):
        """
        Filter a DataFrame to retain only the rows representing the latest encounters for each subject.

        Args:
        - df (pd.DataFrame): Input DataFrame containing columns 'Subject_RID' and 'Date_of_Encounter'.

        Returns:
        - pd.DataFrame: DataFrame filtered to keep only the rows corresponding to the latest encounters
          for each subject.
        """
        latest_encounters = {}
        for index, row in df.iterrows():
            subject_rid = row['Subject_RID']
            date_of_encounter = row['date_of_encounter']
            if subject_rid not in latest_encounters or date_of_encounter > latest_encounters[subject_rid]:
                latest_encounters[subject_rid] = date_of_encounter
        for index, row in df.iterrows():
            if row['date_of_encounter'] != latest_encounters[row['Subject_RID']]:
                df.drop(index, inplace=True)
        return df

    def image_tall(self, ds_bag: DatasetBag, diagnosis_tag: str) -> pd.DataFrame:
        """
        Retrieve tall-format image data based on provided dataset and diagnosis tag filters.

        Args:
        - dataset_rid (str): RID of the dataset to filter images.
        - diagnosis_tag (str): Name of the diagnosis tag used for further filtering.

        Returns:
        - pd.DataFrame: DataFrame containing tall-format image data from fist observation of the subject,
          based on the provided filters.
        """

        # Denormalize via deriva-ml (replaces the former manual Subject->Observation
        # ->Image->Image_Diagnosis pd.merge chain). system_columns=["RCB"] retains
        # the diagnosis row's creating-user id, needed for the grader -> user_list
        # merge on grading tags (RCB is dropped by the denormalizer otherwise).
        frame = ds_bag.get_denormalized_as_dataframe(
            ["Subject", "Observation", "Image", "Image_Diagnosis"],
            system_columns=["RCB"],
        )

        # Map the Table.column denormalized labels back to the flat names the
        # rest of this method (and _find_latest_observation) expects.
        image_frame = frame.rename(columns={
            "Subject.RID": "Subject_RID",
            "Image.RID": "Image_RID",
            "Image_Diagnosis.RID": "Diagnosis_RID",
            "Image.Image_Angle": "Image_Angle",
            "Image.Image_Side": "Image_Side",
            "Image_Diagnosis.Diagnosis_Tag": "Diagnosis_Tag",
            "Image_Diagnosis.Diagnosis_Image": "Diagnosis_Image",
            "Image_Diagnosis.Cup_Disk_Ratio": "Cup_Disk_Ratio",
            "Image_Diagnosis.Image_Quality": "Image_Quality",
            "Image_Diagnosis.RCB": "RCB",
            "Observation.Date_of_Encounter": "date_of_encounter",
        })

        image_frame = image_frame[image_frame['Image_Angle'] == '2']
        image_frame = image_frame[image_frame['Diagnosis_Tag'] == diagnosis_tag]

        # Select only the first observation which is included in the grading app.
        image_frame = self._find_latest_observation(image_frame)

        grading_tags = ["GlaucomaSuspect", "AI_glaucomasuspect_test",
                        "GlaucomaSuspect-Training", "GlaucomaSuspect-Validation"]
        if diagnosis_tag in grading_tags:
            image_frame = pd.merge(image_frame, pd.DataFrame(self.user_list()), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(Full_Name=diagnosis_tag)

        return image_frame[
            ['Subject_RID', 'Image_RID', 'Diagnosis_RID', 'Full_Name', 'Image_Side',
             'Diagnosis_Image', 'Cup_Disk_Ratio', 'Image_Quality']]

    @staticmethod
    def reshape_table(frames: List[pd.DataFrame], compare_value: str):
        """
        Reshape a list of dataframes to long and wide format containing the pre-specified compare value.

        Args:
        - frames (List): A list of dataframes with tall-format image data from fist observation of the subject
        - compare_value (str): Column name of the compared value, choose from ["Diagnosis", "Image_Quality", "Cup_Disk_Ratio"]

        Returns:
        - pd.DataFrame: long and wide formatted dataframe with compare values from all graders and initial diagnosis.
        """
        long = pd.concat(frames).reset_index()
        # change data type for control vocab table
        cols = ['Image_Quality', 'Image_Side', 'Full_Name', 'Diagnosis_Image']
        for c in cols:
            long[c] = long[c].astype('category')
        wide = pd.pivot(long, index=['Image_RID', 'Image_Side', 'Subject_RID'], columns='Full_Name',
                        values=compare_value)  # Reshape from long to wide
        return long, wide

    @staticmethod
    def compute_diagnosis(df: pd.DataFrame,
                          diag_func: Callable,
                          cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]:
        """
        Compute a new diagnosis based on provided functions.

        Args:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - diag_func (Callable): Function to compute Diagnosis.
        - cdr_func (Callable): Function to compute Cup_Disk Ratio.
        - image_quality_func (Callable): Function to compute Image Quality.

        Returns:
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis.
          The Cup_Disk_Ratio is always round to 4 decimal places.
        """
    
        # Empty strings -> NaN. Assign back rather than chained inplace, which is a
        # silent no-op under pandas Copy-on-Write (pandas >= 3.0). pd.to_numeric with
        # errors="coerce" would handle "" too, but keep the explicit replace for clarity.
        df["Cup_Disk_Ratio"] = df["Cup_Disk_Ratio"].replace("", np.nan)
        df["Cup_Disk_Ratio"] = pd.to_numeric(df["Cup_Disk_Ratio"], errors="coerce")
        result = df.groupby("Image_RID").agg({"Cup_Disk_Ratio": cdr_func,
                                              "Diagnosis_Image": diag_func,
                                              "Image_Quality": image_quality_func})
        result = result.round({'Cup_Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image_RID', inplace=True)

        return result.to_dict(orient='records')


    @staticmethod
    def filter_angle_2(ds_bag: DatasetBag) -> pd.DataFrame:
        """
        Filters images for just Field_2 and saves the filtered data to a CSV file.

        Parameters:
        - ds_bag (str): DatasetBag of EyeAI dataset.

        Returns:
        - str: Path to the generated CSV file containing filtered images.
        """
        full_set = pd.DataFrame(list(ds_bag.get_table_as_dict('Image')))
        dataset_field_2 = full_set[full_set['Image_Angle'] == "2"]
        return dataset_field_2

    def plot_roc(self, execution: Execution, data: pd.DataFrame) -> Path:
        """
        Plot Receiver Operating Characteristic (ROC) curve based on prediction results. Save the plot values into a csv file.

        Parameters:
        - execution (Execution): The current execution context for asset tracking.
        - data (pd.DataFrame): DataFrame containing prediction results with columns 'True Label' and
        'Probability Score'.
        Returns:
            Path: Path to the saved csv file of ROC plot values .

        """
        roc_csv_path = execution.asset_file_path(asset_name="ROC", file_name="roc_plot.csv")
        pred_result = pd.read_csv(data)
        y_true = pred_result['True Label']
        scores = pred_result['Probability Score']
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
        roc_df.to_csv(roc_csv_path, index=False)
        # show plot in notebook
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        return roc_csv_path

    def compute_condition_label(self, icd10_asso: pd.DataFrame) -> pd.DataFrame:
        icd_mapping = {
            'H40.00*': 'GS',
            'H40.01*': 'GS',
            'H40.02*': 'GS',
            'H40.03*': 'GS',
            'H40.04*': 'GS',
            'H40.05*': 'GS',
            'H40.06*': 'GS',
            'H40.10*': 'POAG',
            'H40.11*': 'POAG',
            'H40.12*': 'POAG',
            'H40.13*': 'POAG',
            'H40.14*': 'POAG',
            'H40.15*': 'POAG',
            'H40.2*': 'PACG'
        }

        def map_icd_to_category(icd_code):
            for key, value in icd_mapping.items():
                if icd_code.startswith(key[:-1]):
                    return value
            return 'Other'

        # Apply the mapping
        icd10_asso['Condition_Label'] = icd10_asso['ICD10_Eye'].apply(map_icd_to_category)
        # Select severity
        priority = {'PACG': 1, 'POAG': 2, 'GS': 3, 'Other': 4}
        icd10_asso['Priority'] = icd10_asso['Condition_Label'].map(priority)
        icd10_asso = icd10_asso.sort_values(by=['Clinical_Records', 'Priority'])
        combined_prior = icd10_asso.drop_duplicates(subset=['Clinical_Records'], keep='first')
        combined_prior = combined_prior.drop(columns=['RID', 'ICD10_Eye', 'Priority'])
        return combined_prior

    def insert_condition_label(self, condition_label: pd.DataFrame):
        condition_label.rename(columns={'Clinical_Records': 'RID'}, inplace=True)
        entities = condition_label.to_dict(orient='records')
        self._domain_path().Clinical_Records.insert(entities)

    @staticmethod
    def _select_24_2(hvf: pd.DataFrame) -> pd.DataFrame:
        if 'OCR_HVF.RID' not in hvf.columns:
            return hvf
        hvf_clean = hvf.dropna(subset=['OCR_HVF.RID'])
        priority = {'24-2': 1, '10-2': 2, '30-2': 3}
        hvf_clean.loc[:, 'priority'] = hvf_clean['OCR_HVF.Field_Size'].map(priority)
        hvf_sorted = hvf_clean.sort_values(by=['Observation.RID', 'priority'])
        result = hvf_sorted.groupby(['Observation.RID', 'OCR_HVF.Image_Side']).first().reset_index()
        result = result.drop(columns=['priority'])
        return result


    @staticmethod
    def closest_to_fundus(report, fundus, side_col='OCR_HVF.Image_Side'):
        report = report.copy()
        fundus = fundus.copy()
        report['Observation.Date_of_Encounter'] = pd.to_datetime(report['Observation.Date_of_Encounter']).dt.tz_localize(None)
        fundus['Observation.Date_of_Encounter'] = pd.to_datetime(fundus['Observation.Date_of_Encounter']).dt.tz_localize(None)
        report_match = pd.DataFrame(columns=report.columns)

        def find_closest_date(target_date, dates):
            return min(dates, key=lambda d: abs(d - target_date))

        for idx, row in fundus.iterrows():
            rid = row['Subject.RID']
            target_date = row['Observation.Date_of_Encounter']

            for side in ['Left', 'Right']:
                filtered_data = report[(report['Subject.RID'] == rid) & (report[side_col] == side)]
                if not filtered_data.empty:
                    # Find the closest date entry
                    if sum(filtered_data['Observation.Date_of_Encounter'].isna()) > 0:
                        report_match = pd.concat([report_match, filtered_data.iloc[[0]]])
                    else:
                        closest_date = find_closest_date(target_date, filtered_data['Observation.Date_of_Encounter'])
                        closest_entries = filtered_data[filtered_data['Observation.Date_of_Encounter'] == closest_date]
                        report_match = pd.concat([report_match, closest_entries])
        return report_match

    def extract_modality(self, ds_bag: DatasetBag) -> dict[str, pd.DataFrame]:
        # Image
        image = ds_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Image"])
        fundus = image[['Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity',
                        'Observation.RID', 'Observation.Observation_ID', 'Observation.Date_of_Encounter']].drop_duplicates()

        # Report_HVF
        hvf_frame = ds_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_HVF", "OCR_HVF"])
        hvf = self._select_24_2(hvf_frame)
        hvf_match = self.closest_to_fundus(hvf, fundus, side_col='OCR_HVF.Image_Side')

        # Report_RNFL
        rnfl = ds_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_RNFL", "OCR_RNFL"])
        def highest_signal_strength(rnfl):
            rnfl_clean = rnfl.dropna(subset=['OCR_RNFL.RID', 'OCR_RNFL.Signal_Strength'])
            idx = rnfl_clean.groupby(['Observation.RID', 'OCR_RNFL.Image_Side'])['OCR_RNFL.Signal_Strength'].idxmax()
            result = rnfl_clean.loc[idx]
            return result
        rnfl = highest_signal_strength(rnfl)
        rnfl_match = self.closest_to_fundus(rnfl, fundus, side_col='OCR_RNFL.Image_Side')

        # select clinic records by the date of encounter (on the fundus date of encounter)
        clinic = ds_bag.get_denormalized_as_dataframe(include_tables=['Subject', 'Observation', 'Clinical_Records_Observation', 'Clinical_Records'])
        clinic_match = fundus.merge(clinic, on=['Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity',
                                                'Observation.RID', 'Observation.Observation_ID', 'Observation.Date_of_Encounter'], how='left')
        clinic_match = clinic_match[['Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity',
                                    'Observation.RID', 'Observation.Observation_ID', 'Observation.Date_of_Encounter',
                                    'Clinical_Records.RID', 'Clinical_Records.Date_of_Encounter', 'Clinical_Records.LogMAR_VA',
                                    'Clinical_Records.Visual_Acuity_Numerator', 'Clinical_Records.IOP',
                                    'Clinical_Records.Refractive_Error', 'Clinical_Records.CCT',
                                    'Clinical_Records.CDR', 'Clinical_Records.Gonioscopy','Clinical_Records.Condition_Display',
                                    'Clinical_Records.Provider', 'Clinical_Records.Clinical_ID', 'Clinical_Records.ICD_Condition_Label',
                                    'Clinical_Records.Powerform_Laterality'
                                    ]]
        return {"Clinic": clinic_match, "HVF": hvf_match, "RNFL": rnfl_match, "Fundus": fundus}

    def multimodal_wide(self, ds_bag: DatasetBag):
        # Todo add fundus image paths
        modality_df = self.extract_modality(ds_bag)
        clinic = modality_df['Clinic'].rename(columns={'Clinical_Records.Powerform_Laterality': 'Image_Side'})
        rnfl = modality_df['RNFL'].rename(columns={'OCR_RNFL.Image_Side': 'Image_Side'})
        fundus = modality_df['Fundus'] #.rename(columns={'Observation.date_of_encounter': 'date_of_encounter_Fundus'})
        hvf = modality_df['HVF'].rename(columns={'OCR_HVF.Image_Side': 'Image_Side'})
        
        rid_subjects = pd.concat([
            clinic['Subject.RID'],
            rnfl['Subject.RID'],
            fundus['Subject.RID'],
            hvf['Subject.RID']
        ]).drop_duplicates().reset_index(drop=True)
        sides = pd.DataFrame({'Image_Side': ['Right', 'Left']})
        expanded_subjects = rid_subjects.to_frame().merge(sides, how='cross')
        
        clinic.drop(columns=['Observation.RID', 'Observation.Observation_ID', 'Observation.Date_of_Encounter'], inplace=True)
        rnfl.drop(columns=[c for c in rnfl.columns if c.startswith('Observation.')], inplace=True)
        hvf.drop(columns=[c for c in hvf.columns if c.startswith('Observation.')], inplace=True)
        fundus.drop(columns=['Observation.RID', 'Observation.Observation_ID'], inplace=True)
        multimodal_wide = pd.merge(expanded_subjects, fundus, how='left', on=['Subject.RID'])
        multimodal_wide = pd.merge(multimodal_wide, clinic, how='left',
                                   on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        multimodal_wide = pd.merge(multimodal_wide, hvf, how='left',
                                   on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        multimodal_wide = pd.merge(multimodal_wide, rnfl, how='left',
                                   on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        return multimodal_wide

    def severity_analysis(self, ds_bag: DatasetBag):
        wide = self.multimodal_wide(ds_bag)

        def fill_severe_side(df, value_col, new_col, smaller=True):
            df[new_col] = None  # object dtype to hold string labels
            for _, group in df.groupby('Subject.RID'):
                if len(group) != 2:
                    continue
                left = group[group['Image_Side'] == 'Left']
                right = group[group['Image_Side'] == 'Right']
                if left.empty or right.empty:
                    continue
                left_val = left[value_col].values[0]
                right_val = right[value_col].values[0]
                if pd.isna(left_val) or pd.isna(right_val):
                    continue
                if smaller:
                    severe = 'Left' if left_val < right_val else ('Right' if right_val < left_val else 'Left/Right')
                else:
                    severe = 'Left' if left_val > right_val else ('Right' if right_val > left_val else 'Left/Right')
                df.loc[group.index, new_col] = severe

        fill_severe_side(wide, 'OCR_RNFL.Average_RNFL_Thickness(μm)', 'RNFL_severe', smaller=True)
        fill_severe_side(wide, 'OCR_HVF.MD', 'HVF_severe', smaller=True)
        fill_severe_side(wide, 'Clinical_Records.CDR', 'CDR_severe', smaller=False)

        def check_severity(row):
            severities = [row['RNFL_severe'], row['HVF_severe'], row['CDR_severe']]
            try:
                return not (all(["Left" in l for l in severities]) or all(["Right" in l for l in severities]))
            except Exception:
                return True

        wide['Severity_Mismatch'] = wide.apply(check_severity, axis=1)
        return wide

    def multimodal_wide_from_subject_bag(
        self,
        image_bag: DatasetBag,
        subject_bag: DatasetBag,
    ) -> pd.DataFrame:
        """Compute the multimodal wide table directly from an image bag and a subject bag,
        without saving any records back to the catalog.

        Use this as a workaround when the enriched image dataset cannot be downloaded due
        to the deep FK traversal timeout bug in deriva_ml. Switch to multimodal_wide once
        the dataset has been enriched via add_multimodal_measurements and the bug is fixed.

        Both bags should be downloaded within an execution context before calling this method.

        Args:
            image_bag: DatasetBag for the image dataset (provides fundus anchor).
            subject_bag: DatasetBag from a subject dataset covering the same subjects.

        Returns:
            Wide DataFrame with one row per subject per side, joining fundus, HVF, RNFL,
            and Clinical Records.
        """
        image_frame = image_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Image"])
        fundus = image_frame[['Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender',
                               'Subject.Subject_Ethnicity', 'Observation.RID',
                               'Observation.Observation_ID', 'Observation.Date_of_Encounter']].drop_duplicates()

        hvf_frame = subject_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_HVF", "OCR_HVF"])
        hvf = self._select_24_2(hvf_frame)
        hvf_matched = self.closest_to_fundus(hvf, fundus, side_col='OCR_HVF.Image_Side')

        rnfl_frame = subject_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_RNFL", "OCR_RNFL"])
        def highest_signal_strength(rnfl):
            rnfl_clean = rnfl.dropna(subset=['OCR_RNFL.RID', 'OCR_RNFL.Signal_Strength'])
            idx = rnfl_clean.groupby(['Observation.RID', 'OCR_RNFL.Image_Side'])['OCR_RNFL.Signal_Strength'].idxmax()
            return rnfl_clean.loc[idx]
        rnfl_matched = self.closest_to_fundus(highest_signal_strength(rnfl_frame), fundus, side_col='OCR_RNFL.Image_Side')

        clinic_frame = subject_bag.get_denormalized_as_dataframe(
            include_tables=['Subject', 'Observation', 'Clinical_Records_Observation', 'Clinical_Records']
        )
        clinic_matched = self.closest_to_fundus(clinic_frame, fundus, side_col='Clinical_Records.Powerform_Laterality')

        clinic = clinic_matched.rename(columns={'Clinical_Records.Powerform_Laterality': 'Image_Side'})
        rnfl = rnfl_matched.rename(columns={'OCR_RNFL.Image_Side': 'Image_Side'})
        hvf = hvf_matched.rename(columns={'OCR_HVF.Image_Side': 'Image_Side'})

        rid_subjects = pd.concat([
            clinic['Subject.RID'],
            rnfl['Subject.RID'],
            fundus['Subject.RID'],
            hvf['Subject.RID']
        ]).drop_duplicates().reset_index(drop=True)
        sides = pd.DataFrame({'Image_Side': ['Right', 'Left']})
        expanded_subjects = rid_subjects.to_frame().merge(sides, how='cross')

        clinic.drop(columns=['Observation.RID', 'Observation.Observation_ID', 'Observation.Date_of_Encounter'], inplace=True)
        rnfl.drop(columns=[c for c in rnfl.columns if c.startswith('Observation.')], inplace=True)
        hvf.drop(columns=[c for c in hvf.columns if c.startswith('Observation.')], inplace=True)
        fundus.drop(columns=['Observation.RID', 'Observation.Observation_ID'], inplace=True)

        result = pd.merge(expanded_subjects, fundus, how='left', on=['Subject.RID'])
        result = pd.merge(result, clinic, how='left',
                          on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        result = pd.merge(result, hvf, how='left',
                          on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        result = pd.merge(result, rnfl, how='left',
                          on=['Image_Side', 'Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender', 'Subject.Subject_Ethnicity'])
        return result

    def add_multimodal_measurements(
        self,
        image_dataset: DatasetSpec,
        subject_bag: DatasetBag,
        execution_rid: str | None = None,
    ) -> dict[str, int]:
        """Add matched HVF, RNFL, and Clinical Records from a subject dataset bag as
        members of an image dataset, making the image dataset self-contained for
        multimodal analysis.

        The selection logic mirrors extract_modality: 24-2 priority for HVF, highest
        signal strength for RNFL, and Clinical Records all matched to the closest fundus
        date per subject and side.

        Report_HVF, Report_RNFL, and Clinical_Records are registered as dataset element
        types if not already registered (one-time catalog schema change).

        Args:
            image_dataset: DatasetSpec for the image dataset to add members to.
            subject_bag: DatasetBag from a subject dataset covering the same subjects.
            execution_rid: Optional execution RID for provenance tracking.

        Returns:
            dict mapping table name to number of member RIDs added.
        """
        # Get fundus frame from image dataset bag
        image_bag = self.download_dataset_bag(image_dataset)
        image_frame = image_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Image"])
        fundus = image_frame[['Subject.RID', 'Subject.Subject_ID', 'Subject.Subject_Gender',
                               'Subject.Subject_Ethnicity', 'Observation.RID',
                               'Observation.Observation_ID', 'Observation.Date_of_Encounter']].drop_duplicates()

        # Match HVF from subject bag
        hvf_frame = subject_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_HVF", "OCR_HVF"])
        hvf = self._select_24_2(hvf_frame)
        hvf_matched = self.closest_to_fundus(hvf, fundus, side_col='OCR_HVF.Image_Side')

        # Match RNFL from subject bag
        rnfl_frame = subject_bag.get_denormalized_as_dataframe(["Subject", "Observation", "Report_RNFL", "OCR_RNFL"])
        def highest_signal_strength(rnfl):
            rnfl_clean = rnfl.dropna(subset=['OCR_RNFL.RID', 'OCR_RNFL.Signal_Strength'])
            idx = rnfl_clean.groupby(['Observation.RID', 'OCR_RNFL.Image_Side'])['OCR_RNFL.Signal_Strength'].idxmax()
            return rnfl_clean.loc[idx]
        rnfl_matched = self.closest_to_fundus(highest_signal_strength(rnfl_frame), fundus, side_col='OCR_RNFL.Image_Side')

        # Match Clinical Records from subject bag: exact observation match first,
        # falling back to closest date per subject per side
        clinic_frame = subject_bag.get_denormalized_as_dataframe(
            include_tables=['Subject', 'Observation', 'Clinical_Records_Observation', 'Clinical_Records']
        )
        clinic_matched = self.closest_to_fundus(clinic_frame, fundus, side_col='Clinical_Records.Powerform_Laterality')

        # Collect selected RIDs per table
        members: dict[str, list] = {}
        hvf_rids = hvf_matched['Report_HVF.RID'].dropna().unique().tolist() if 'Report_HVF.RID' in hvf_matched.columns else []
        rnfl_rids = rnfl_matched['Report_RNFL.RID'].dropna().unique().tolist() if 'Report_RNFL.RID' in rnfl_matched.columns else []
        clinic_rids = clinic_matched['Clinical_Records.RID'].dropna().unique().tolist() if 'Clinical_Records.RID' in clinic_matched.columns else []

        if hvf_rids:
            members['Report_HVF'] = hvf_rids
        if rnfl_rids:
            members['Report_RNFL'] = rnfl_rids
        if clinic_rids:
            members['Clinical_Records'] = clinic_rids

        if not members:
            return {}

        # Register tables as element types (idempotent)
        for table in members:
            self.add_dataset_element_type(table)

        # Add members to the live dataset
        dataset = self.lookup_dataset(image_dataset)
        dataset.add_dataset_members(members, execution_rid=execution_rid)

        return {k: len(v) for k, v in members.items()}
