import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Columbia Spine Registry Validation Tool",
    layout="wide"
)

st.title("Columbia Spine Registry Data Validation Tool")
st.caption("Automated rule-based and statistical validation for clinical registry data")

st.write("Upload a CSV file to detect statistically anomalous records.")

#Option to select validation mode
mode = st.radio(
    "Select Validation Mode",
    ["Statistical Anomaly Detection", "Rule-Based Validation"]
)

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    df["row_id"] = np.arange(len(df))

    # Optional Identifier Selection
    identifier_column = st.selectbox(
        "Select Identifier Column (Optional)",
        ["None"] + df.columns.tolist()
    ) 
    
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    feature_cols = [c for c in numeric_cols if c != "row_id"]

    # Remove identifier column from modeling if selected
    if identifier_column != "None":
        feature_cols = [c for c in feature_cols if c != identifier_column]

    # Remove fully missing columns
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]

    if len(feature_cols) == 0:
        st.error("No usable numeric columns found.")
    else:
        if mode == "Statistical Anomaly Detection":

            percent_value = st.slider(
                "Select percentage of records to flag for review (%)",
                min_value=1,
                max_value=50,   # limited at 50 because of isolation forest requirements and to preserve the assumption that anomalies represent a minority of records
                value=5,
                step=1
            )

            review_percent = percent_value / 100
            st.caption(f"{percent_value}% of rows will be flagged for review.")

            st.write("Running anomaly detection...")

            # Impute Missing Values
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(df[feature_cols])

            # Scaling 
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Tuned Isolation Forest
            iso = IsolationForest(
                n_estimators=500,                          # More stable
                contamination=review_percent,              # Match slider
                max_samples=min(256, len(df)),             # Stable subsampling
                max_features=0.8,                          # Feature subsampling
                random_state=42,
                n_jobs=-1
            )

            iso.fit(X_scaled)

            # anomaly_score: higher means more anomalous
            df["anomaly_score"] = -iso.decision_function(X_scaled)

            # Rank descending (most anomalous first)
            df_ranked = df.sort_values("anomaly_score", ascending=False)
            review_n = int(len(df) * review_percent)
            df_review = df_ranked.head(review_n)

            # Stability Check with Multiple Seeds
            seeds = [42, 99, 123]
            flag_sets = []

            for seed in seeds:
                iso_temp = IsolationForest(
                    n_estimators=500,
                    contamination=review_percent,
                    max_samples=min(256, len(df)),
                    max_features=0.8,
                    random_state=seed,
                    n_jobs=-1
                )

                iso_temp.fit(X_scaled)
                scores_temp = -iso_temp.decision_function(X_scaled)

                temp_df = df.copy()
                temp_df["temp_score"] = scores_temp
                temp_ranked = temp_df.sort_values("temp_score", ascending=False)
                temp_top = temp_ranked.head(review_n)

                flag_sets.append(set(temp_top["row_id"]))

            # Compute pairwise overlaps
            overlap_1_2 = len(flag_sets[0].intersection(flag_sets[1])) / review_n
            overlap_1_3 = len(flag_sets[0].intersection(flag_sets[2])) / review_n
            overlap_2_3 = len(flag_sets[1].intersection(flag_sets[2])) / review_n

            st.subheader("Model Stability Check")
            st.write("Overlap between seed 42 & 99:", round(overlap_1_2 * 100, 1), "%")
            st.write("Overlap between seed 42 & 123:", round(overlap_1_3 * 100, 1), "%")
            st.write("Overlap between seed 99 & 123:", round(overlap_2_3 * 100, 1), "%")
            # Explainability Layer
            
            # Calculate robust statistics
            medians = df[feature_cols].median()
            iqr = df[feature_cols].quantile(0.75) - df[feature_cols].quantile(0.25)

            def explain_row(row):
                deviations = {}
                for col in feature_cols:
                    if iqr[col] != 0:
                        deviations[col] = abs(row[col] - medians[col]) / iqr[col]
                    else:
                        deviations[col] = 0

                # Get top 3 drivers
                top_features = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:3]

                # Format explanation text
                explanation = ", ".join(
                    [f"{feature} (high deviation)" for feature, _ in top_features]
                )

                return explanation

            # Apply explanation to flagged rows only
            df_review["top_anomaly_drivers"] = df_review.apply(explain_row, axis=1)

            st.success("Anomaly detection complete.")

            st.write("Top flagged records:")
            st.dataframe(df_review)

            # Export to Excel
            output_path = "anomaly_results.xlsx"
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Full Dataset", index=False)
                df_review.to_excel(writer, sheet_name="Review List", index=False)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Excel Report",
                    data=f,
                    file_name="anomaly_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        if mode == "Rule-Based Validation":

            st.subheader("Create Numeric Validation Rules")

            if "rules" not in st.session_state:
                st.session_state.rules = []

            # Rule Type Selector
            rule_type = st.radio(
                "Rule Type",
                ["Compare to Value", "Compare to Another Column"]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                selected_column = st.selectbox("Select Column", feature_cols)

            with col2:
                operator = st.selectbox("Select Operator", [">", "<", ">=", "<=", "==", "!="])

            # Dynamic third input
            if rule_type == "Compare to Value":
                with col3:
                    comparison_value = st.number_input("Enter Value")

            if rule_type == "Compare to Another Column":
                with col3:
                    comparison_column = st.selectbox("Select Comparison Column", feature_cols)

            # Add Rule Button
            if st.button("Add Rule"):

                if rule_type == "Compare to Value":
                    st.session_state.rules.append(
                        {
                            "type": "value",
                            "column": selected_column,
                            "operator": operator,
                            "value": comparison_value
                        }
                    )

                if rule_type == "Compare to Another Column":
                    st.session_state.rules.append(
                        {
                            "type": "column",
                            "column": selected_column,
                            "operator": operator,
                            "comparison_column": comparison_column
                        }
                    )

            # Display Active Rules
            if st.session_state.rules:
                st.write("Active Rules:")

                for i, rule in enumerate(st.session_state.rules):
                    colA, colB = st.columns([4, 1])

                    with colA:
                        if rule["type"] == "value":
                            st.write(f"{rule['column']} {rule['operator']} {rule['value']}")
                        else:
                            st.write(f"{rule['column']} {rule['operator']} {rule['comparison_column']}")

                    with colB:
                        if st.button("❌", key=f"delete_{i}"):
                            st.session_state.rules.pop(i)
                            st.rerun()

                if st.button("Clear All Rules"):
                    st.session_state.rules = []
                    st.rerun()

            # Run Validation
            if st.button("Run Rule Validation"):

                if len(st.session_state.rules) == 0:
                    st.warning("No rules defined. Please add at least one rule.")
                    st.stop()

                violation_mask = pd.Series(False, index=df.index)

                for rule in st.session_state.rules:

                    col = rule["column"]
                    op = rule["operator"]

                    # Compare to constant value
                    if rule["type"] == "value":
                        val = rule["value"]

                        if op == ">":
                            current_mask = ~(df[col] > val)
                        elif op == "<":
                            current_mask = ~(df[col] < val)
                        elif op == ">=":
                            current_mask = ~(df[col] >= val)
                        elif op == "<=":
                            current_mask = ~(df[col] <= val)
                        elif op == "==":
                            current_mask = ~(df[col] == val)
                        elif op == "!=":
                            current_mask = ~(df[col] != val)

                    # Compare to another column
                    if rule["type"] == "column":
                        comp_col = rule["comparison_column"]

                        if op == ">":
                            current_mask = ~(df[col] > df[comp_col])
                        elif op == "<":
                            current_mask = ~(df[col] < df[comp_col])
                        elif op == ">=":
                            current_mask = ~(df[col] >= df[comp_col])
                        elif op == "<=":
                            current_mask = ~(df[col] <= df[comp_col])
                        elif op == "==":
                            current_mask = ~(df[col] == df[comp_col])
                        elif op == "!=":
                            current_mask = ~(df[col] != df[comp_col])

                    violation_mask |= current_mask

                df["rule_violation"] = violation_mask
                df_violations = df[df["rule_violation"]]

                st.success("Rule validation complete.")
                st.write("Flagged records:")
                st.dataframe(df_violations)

                # Export
                output_path = "rule_validation_results.xlsx"
                with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Full Dataset", index=False)
                    df_violations.to_excel(writer, sheet_name="Rule Violations", index=False)

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Rule Validation Report",
                        data=f,
                        file_name="rule_validation_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    )
