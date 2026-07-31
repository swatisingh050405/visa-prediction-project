import numpy as np


class ShapExplainer:

    def __init__(self, preprocessor, explainer):

        self.preprocessor = preprocessor
        self.explainer = explainer

    def get_feature_importance(self, transformed_data):

        shap_values = self.explainer.shap_values(transformed_data)

        # Binary Classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = self.preprocessor.get_feature_names_out()

        explanations = []

        for feature, impact in zip(feature_names, shap_values[0]):

            explanations.append(
                {
                    "feature": self.clean_feature_name(feature),
                    "impact": float(impact)
                }
            )

        explanations.sort(
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        # Remove duplicate feature names
        unique = {}

        for item in explanations:

            if item["feature"] not in unique:
                unique[item["feature"]] = item

        explanations = list(unique.values())

        return explanations[:5]

    def clean_feature_name(self, feature_name):

        feature_name = feature_name.split("__")[-1]

        if feature_name.startswith("continent_"):
            return "Continent"

        if feature_name.startswith("education_of_employee_"):
            return "Education"

        if feature_name.startswith("has_job_experience_"):
            return "Job Experience"

        if feature_name.startswith("requires_job_training_"):
            return "Job Training"

        if feature_name.startswith("region_of_employment_"):
            return "Region"

        if feature_name.startswith("unit_of_wage_"):
            return "Wage Unit"

        if feature_name.startswith("full_time_position_"):
            return "Full Time Position"

        feature_name = feature_name.replace("_", " ")

        return feature_name.title()