import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from datetime import timedelta

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="Churn Prediction",
    page_icon="📊",
)

# =========================
# CONSTANTS
# =========================
PASSWORD = "churn2026"
MODEL_PATH = "churn_model_lr.pkl"
SCALER_PATH = "churn_scaler.pkl"
MASTER_CSV = "master_transactions.csv"
PREV_SCORES_CSV = "previous_scores.csv"
TRACKING_CSV = "monthly_tracking.csv"

FEATURE_ORDER = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_items",
    "unique_products",
    "avg_days_between",
    "tenure_days",
    "num_countries",
]

NON_PRODUCTS = [
    "POST", "D", "M", "AMAZON FEE", "CRUK COMMISSION",
    "BANK CHARGES", "DOTCOM POSTAGE", "NEXT DAY CARRIAGE",
]

RISK_COLORS = {
    "High Risk": "#D85A30",
    "Medium Risk": "#F0C040",
    "Low Risk": "#1D9E75",
}

# =========================
# CACHED RESOURCES (model & scaler)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please place it in the app folder.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file '{SCALER_PATH}' not found. Please place it in the app folder.")
        st.stop()
    return joblib.load(SCALER_PATH)

# =========================
# HELPER FUNCTIONS
# =========================
def strip_currency(value):
    if isinstance(value, str):
        for char in ["£", "$", "€", ","]:
            value = value.replace(char, "")
    return value

def create_month_list(start_year=2022, end_year=2026):
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            months.append(f"{y}-{m:02d}")
    return months

def convert_df(df):
    """Convert dataframe to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")

# =========================
# MAIN APP
# =========================
def main():
    st.title("Customer Churn Prediction System")
    st.caption("Powered by CJ John Paulin")

    # ----- SIDEBAR -----
    with st.sidebar:
        st.header("Access")
        password = st.text_input("Password", type="password")
        if password != PASSWORD:
            st.warning("Enter password to access the app")
            st.stop()

        st.markdown("---")
        st.header("Upload")
        uploaded_file = st.file_uploader(
            "Upload monthly transaction file (.xlsx)",
            type=["xlsx"],
        )
        month_label = st.selectbox(
            "Month/Year of this upload",
            options=create_month_list(),
            index=len(create_month_list()) - 1,
        )

    # ----- FILE UPLOAD & STATE RESET -----
    if uploaded_file is None:
        # No file – clear everything
        st.session_state.confirm_clean = False
        st.session_state.data_appended = False
        st.session_state.processed = False
        for key in ["df_clean", "features_df", "previous_scores_df", "last_file"]:
            if key in st.session_state:
                del st.session_state[key]
        st.info("Please upload a .xlsx file to begin.")
        st.stop()

    # Reset state if a new file is uploaded (file name change)
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state.confirm_clean = False
        st.session_state.data_appended = False
        st.session_state.processed = False
        for key in ["df_clean", "features_df", "previous_scores_df"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_file = uploaded_file.name

    # =====================
    # STEP 1 – DATA CLEANING
    # =====================
    # Only run cleaning if not yet confirmed
    if not st.session_state.get("confirm_clean"):
        # Read uploaded file
        try:
            df = pd.read_excel(
                uploaded_file,
                sheet_name="Monthly Transactions",
                header=0,
                skiprows=[1],
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        cleaning_log = []
        total_rows_start = len(df)

        # 1. Strip currency symbols from UnitPrice
        df["UnitPrice"] = df["UnitPrice"].astype(str).apply(strip_currency)
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

        # 2. Convert InvoiceDate, drop invalid dates
        before = len(df)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed rows with invalid InvoiceDate", before - after))

        # 3. CustomerID: numeric, drop null / ≤0
        before = len(df)
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
        df = df.dropna(subset=["CustomerID"])
        df = df[df["CustomerID"] > 0]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed rows with missing or invalid CustomerID (≤0)", before - after))

        # 4. Blank Description
        before = len(df)
        df = df[df["Description"].notna() & (df["Description"].str.strip() != "")]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed rows with blank Description", before - after))

        # 5. Quantity ≤ 0
        before = len(df)
        df = df[df["Quantity"] > 0]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed rows with Quantity ≤ 0", before - after))

        # 6. UnitPrice ≤ 0
        before = len(df)
        df = df[df["UnitPrice"] > 0]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed rows with UnitPrice ≤ 0", before - after))

        # 7. Cancellation invoices
        before = len(df)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed cancellation invoices (starts with 'C')", before - after))

        # 8. Non-product StockCodes
        before = len(df)
        df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCTS)]
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed non-product StockCodes", before - after))

        # 9. True duplicates (InvoiceNo+StockCode+CustomerID+InvoiceDate)
        before = len(df)
        df = df.drop_duplicates(subset=["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate"])
        after = len(df)
        if before - after > 0:
            cleaning_log.append(("Removed true duplicates", before - after))

        # 10. Recalculate Revenue
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        # 11. Derive YearMonth (store as string to allow CSV saving)
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

        total_rows_end = len(df)

        # Show cleaning report
        st.subheader("Data Cleaning Report")
        if cleaning_log:
            log_df = pd.DataFrame(cleaning_log, columns=["Action", "Rows Removed"])
            st.table(log_df)
        else:
            st.write("No rows were removed during cleaning.")
        st.write(f"**Rows before cleaning:** {total_rows_start}")
        st.write(f"**Rows after cleaning:** {total_rows_end}")
        st.write(f"**Total rows removed:** {total_rows_start - total_rows_end}")

        # Confirm button
        confirm = st.button("✅ Confirm & Proceed to Dashboard")
        if confirm:
            st.session_state.confirm_clean = True
            st.session_state.df_clean = df.copy()
            st.session_state.month_label = month_label  # Save for dashboard
            st.session_state.data_appended = False
            st.session_state.processed = False
            st.rerun()  # rerun to move past cleaning
        else:
            st.info("Please confirm the cleaning results to continue.")
            st.stop()

    # =====================
    # PROCESSING (only once after confirmation)
    # =====================
    if st.session_state.get("confirm_clean") and not st.session_state.get("processed"):
        df_clean = st.session_state.df_clean

        # Load previous scores (last month) BEFORE overwriting them
        if os.path.exists(PREV_SCORES_CSV):
            st.session_state.previous_scores_df = pd.read_csv(PREV_SCORES_CSV)
        else:
            st.session_state.previous_scores_df = None

        # ---------- Append to master_transactions.csv ----------
        if not st.session_state.data_appended:
            if os.path.exists(MASTER_CSV):
                master_old = pd.read_csv(MASTER_CSV, parse_dates=["InvoiceDate"])
                master_new = pd.concat([master_old, df_clean], ignore_index=True)
                master_new.to_csv(MASTER_CSV, index=False)
            else:
                df_clean.to_csv(MASTER_CSV, index=False)
            st.session_state.data_appended = True

        # ---------- Feature Engineering on FULL master ----------
        master = pd.read_csv(MASTER_CSV, parse_dates=["InvoiceDate"])
        # Drop any remaining null CustomerIDs (shouldn't exist)
        master = master.dropna(subset=["CustomerID"])
        master["CustomerID"] = master["CustomerID"].astype(int)
        # Recreate YearMonth as Period for grouping
        master["YearMonth"] = pd.to_datetime(master["InvoiceDate"]).dt.to_period("M")

        reference_date = master["InvoiceDate"].max() + timedelta(days=1)

        features = master.groupby("CustomerID").agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Revenue", "sum"),
            avg_order_value=("Revenue", "mean"),
            total_items=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            avg_days_between=(
                "InvoiceDate",
                lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else 0,
            ),
            tenure_days=("InvoiceDate", lambda x: (x.max() - x.min()).days),
            num_countries=("Country", "nunique"),
        ).reset_index()

        # Fill any NaN in avg_days_between
        features["avg_days_between"] = features["avg_days_between"].fillna(0)

        # ---------- Model Scoring ----------
        lr = load_model()
        scaler = load_scaler()
        X = features[FEATURE_ORDER].astype(float)
        X_scaled = scaler.transform(X)
        features["churn_probability"] = lr.predict_proba(X_scaled)[:, 1]

        # ---------- Risk Tiers ----------
        def assign_tier(prob):
            if prob >= 0.70:
                return "High Risk"
            elif prob >= 0.40:
                return "Medium Risk"
            else:
                return "Low Risk"
        features["risk_tier"] = features["churn_probability"].apply(assign_tier)

        # ---------- Recommended Action ----------
        def get_action(row):
            if row["risk_tier"] == "High Risk" and row["monetary"] >= 500:
                return "Personal call this week"
            elif row["risk_tier"] == "High Risk" and row["monetary"] < 500:
                return "One email only — low priority"
            elif row["risk_tier"] == "Medium Risk":
                return "Send reactivation offer — 10-15% discount"
            else:
                return "Add to loyalty program"
        features["recommended_action"] = features.apply(get_action, axis=1)

        # ---------- Save new scores ----------
        features[["CustomerID", "churn_probability", "risk_tier"]].to_csv(
            PREV_SCORES_CSV, index=False
        )

        # ---------- Monthly tracking (avg churn) ----------
        current_month_str = st.session_state.get("month_label")
        avg_churn = features["churn_probability"].mean()
        if os.path.exists(TRACKING_CSV):
            tracking_df = pd.read_csv(TRACKING_CSV)
            if current_month_str not in tracking_df["Month"].values:
                new_row = pd.DataFrame({"Month": [current_month_str], "Avg_Churn_Probability": [avg_churn]})
                tracking_df = pd.concat([tracking_df, new_row], ignore_index=True)
                tracking_df.to_csv(TRACKING_CSV, index=False)
        else:
            pd.DataFrame({"Month": [current_month_str], "Avg_Churn_Probability": [avg_churn]}).to_csv(
                TRACKING_CSV, index=False
            )

        # Store everything needed for dashboard in session_state
        st.session_state.features_df = features
        st.session_state.master_df = master
        st.session_state.processed = True

    # =====================
    # DASHBOARD (only if processed)
    # =====================
    if not st.session_state.get("processed"):
        st.stop()

    # Retrieve data from session_state
    features = st.session_state.features_df
    df_clean = st.session_state.df_clean
    master = st.session_state.master_df

    # This month's data from uploaded file (filtered by YearMonth)
    upload_max_date = df_clean["InvoiceDate"].max()
    upload_month = upload_max_date.to_period("M")  # Period object
    # Ensure df_clean YearMonth is also Period for comparison
    df_clean["YearMonth"] = pd.to_datetime(df_clean["InvoiceDate"]).dt.to_period("M")
    df_this_month = df_clean[df_clean["YearMonth"] == upload_month]

    # ----- Metric calculations -----
    this_month_revenue = df_this_month["Revenue"].sum()
    this_month_customers = df_this_month["CustomerID"].nunique()
    this_month_transactions = df_this_month["InvoiceNo"].nunique()
    this_month_avg_order = this_month_revenue / this_month_transactions if this_month_transactions else 0

    # New vs returning
    previous_months = master[master["YearMonth"] != upload_month]
    prev_cust_set = set(previous_months["CustomerID"].unique())
    curr_cust_set = set(df_this_month["CustomerID"].unique())
    new_cust_count = len(curr_cust_set - prev_cust_set)
    returning_cust_count = len(curr_cust_set & prev_cust_set)

    # Product insights
    if not df_this_month.empty:
        prod_qty = df_this_month.groupby("Description")["Quantity"].sum()
        most_sold = prod_qty.idxmax() if not prod_qty.empty else "N/A"
        prod_rev = df_this_month.groupby("Description")["Revenue"].sum()
        top_rev = prod_rev.idxmax() if not prod_rev.empty else "N/A"
        unique_prods = df_this_month["StockCode"].nunique()
    else:
        most_sold = top_rev = "N/A"
        unique_prods = 0

    # Customer behavior
    avg_recency = features["recency"].mean()
    orders_per_cust = df_this_month.groupby("CustomerID")["InvoiceNo"].nunique()
    avg_orders = orders_per_cust.mean() if not orders_per_cust.empty else 0
    repeat_rate = (orders_per_cust > 1).mean() if not orders_per_cust.empty else 0

    # Risk summary
    risk_counts = features["risk_tier"].value_counts()
    total_cust_all = len(features)
    high_count = risk_counts.get("High Risk", 0)
    med_count = risk_counts.get("Medium Risk", 0)
    low_count = risk_counts.get("Low Risk", 0)
    high_rev = features[features["risk_tier"] == "High Risk"]["monetary"].sum()
    med_rev = features[features["risk_tier"] == "Medium Risk"]["monetary"].sum()
    low_rev = features[features["risk_tier"] == "Low Risk"]["monetary"].sum()

    st.markdown("---")
    st.header(f"📊 Dashboard — {st.session_state.get('month_label', upload_month)}")

    # =====================
    # SECTION 1 – Business Overview
    # =====================
    st.subheader("1. This Month's Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"£{this_month_revenue:,.2f}")
    with col2:
        st.metric("Total Customers", this_month_customers)
    with col3:
        st.metric("New Customers", new_cust_count)
    with col4:
        st.metric("Returning Customers", returning_cust_count)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Most Sold Product", most_sold)
    with col2:
        st.metric("Top Revenue Product", top_rev)
    with col3:
        st.metric("Unique Products Sold", unique_prods)
    with col4:
        st.metric("Avg Order Value", f"£{this_month_avg_order:,.2f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Days Since Last Purchase", f"{avg_recency:.1f} days")
    with col2:
        st.metric("Avg Orders per Customer", f"{avg_orders:.2f}")
    with col3:
        st.metric("Repeat Purchase Rate", f"{repeat_rate:.1%}")
    with col4:
        st.metric("Total Transactions", this_month_transactions)

    # =====================
    # SECTION 2 – Churn Risk Summary
    # =====================
    st.subheader("2. Churn Risk Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<h3 style='color:{RISK_COLORS['High Risk']}'>High Risk</h3>", unsafe_allow_html=True)
        st.metric("Customers", high_count, delta=f"{high_count/total_cust_all:.1%} of total")
        st.metric("Total Revenue", f"£{high_rev:,.2f}")
    with col2:
        st.markdown(f"<h3 style='color:{RISK_COLORS['Medium Risk']}'>Medium Risk</h3>", unsafe_allow_html=True)
        st.metric("Customers", med_count, delta=f"{med_count/total_cust_all:.1%} of total")
        st.metric("Total Revenue", f"£{med_rev:,.2f}")
    with col3:
        st.markdown(f"<h3 style='color:{RISK_COLORS['Low Risk']}'>Low Risk</h3>", unsafe_allow_html=True)
        st.metric("Customers", low_count, delta=f"{low_count/total_cust_all:.1%} of total")
        st.metric("Total Revenue", f"£{low_rev:,.2f}")

    # Revenue by risk tier bar chart
    rev_by_tier = features.groupby("risk_tier")["monetary"].sum().reset_index()
    fig_bar = px.bar(
        rev_by_tier,
        x="risk_tier",
        y="monetary",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        title="Revenue by Risk Tier",
        labels={"monetary": "Total Revenue", "risk_tier": "Risk Tier"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Delta with previous month if available
    prev_scores = st.session_state.get("previous_scores_df")
    if prev_scores is not None and not prev_scores.empty:
        old_high_rev = features[features["CustomerID"].isin(
            prev_scores[prev_scores["risk_tier"] == "High Risk"]["CustomerID"]
        )]["monetary"].sum()
        delta_rev = high_rev - old_high_rev
        st.metric(
            "High Risk Revenue vs Last Month",
            f"£{high_rev:,.2f}",
            delta=f"£{delta_rev:,.2f}",
            delta_color="inverse",  # increase = worse
        )

    # =====================
    # SECTION 3 – Month-over-Month Comparison
    # =====================
    st.subheader("3. Month Over Month Comparison")
    if master["YearMonth"].nunique() > 1 and prev_scores is not None and not prev_scores.empty:
        # Churn trend from tracking CSV
        if os.path.exists(TRACKING_CSV):
            tracking_df = pd.read_csv(TRACKING_CSV)
            if not tracking_df.empty:
                # Force categorical x-axis for proper monthly labels
                fig_trend = px.line(
                    tracking_df,
                    x="Month",
                    y="Avg_Churn_Probability",
                    title="Average Churn Probability Over Time",
                    markers=True,
                    category_orders={"Month": sorted(tracking_df["Month"].tolist())}
                )
                fig_trend.update_xaxes(type="category")
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Tracking data not yet available.")

        # Tier movements
        old = prev_scores.set_index("CustomerID")
        new = features.set_index("CustomerID")
        merged = old.join(new, lsuffix="_old", rsuffix="_new", how="inner")

        def movement(row):
            t_old = row["risk_tier_old"]
            t_new = row["risk_tier_new"]
            if t_old == "Low Risk" and t_new == "Medium Risk":
                return "Low → Medium (worse)"
            elif t_old == "Medium Risk" and t_new == "Low Risk":
                return "Medium → Low (improving)"
            elif t_old == "Medium Risk" and t_new == "High Risk":
                return "Medium → High (urgent)"
            elif t_old == "High Risk" and t_new == "Medium Risk":
                return "High → Medium (recovering)"
            elif t_old == t_new:
                return "No change"
            else:
                return "Other"
        merged["movement"] = merged.apply(movement, axis=1)
        move_counts = merged["movement"].value_counts().reset_index()
        move_counts.columns = ["Movement", "Count"]
        st.write("**Tier Movements vs Last Month**")
        st.dataframe(move_counts, use_container_width=True)

        # Retention of new customers from last month
        last_month = upload_month - 1
        first_purchase = master.groupby("CustomerID")["InvoiceDate"].min()
        last_month_new = first_purchase[first_purchase.dt.to_period("M") == last_month].index
        retained_set = set(last_month_new) & curr_cust_set
        if len(last_month_new) > 0:
            retention_rate = len(retained_set) / len(last_month_new)
            st.metric("New Customer Retention (last → this month)", f"{retention_rate:.1%}")
        else:
            st.metric("New Customer Retention", "N/A (no new customers last month)")
    else:
        st.info("Month over month comparison will be available from next month onwards.")

    # =====================
    # SECTION 4 – Action List
    # =====================
    st.subheader("4. Action List")
    # Top 5 high-risk high-revenue
    top5 = features[(features["risk_tier"] == "High Risk") & (features["monetary"] >= 500)].nlargest(5, "monetary")
    if not top5.empty:
        st.markdown("**🔴 Top 5 High-Risk Customers by Revenue (Priority)**")
        # Color only the risk_tier column
        def color_tier(val):
            return f"background-color: {RISK_COLORS.get(val, 'white')}; color: white"
        styled_top5 = top5.style.map(color_tier, subset=["risk_tier"])
        st.dataframe(styled_top5, use_container_width=True)

    # Full ranked list
    features["priority_score"] = features["churn_probability"] * features["monetary"]
    ranked = features.sort_values("priority_score", ascending=False)
    display_cols = [
        "CustomerID", "churn_probability", "risk_tier",
        "monetary", "recency", "recommended_action"
    ]
    # Color only the risk_tier cell
    styled_ranked = ranked[display_cols].style.map(
        lambda val: f"background-color: {RISK_COLORS.get(val, 'white')}; color: white",
        subset=["risk_tier"]
    )
    st.dataframe(styled_ranked, use_container_width=True, height=600)

    # =====================
    # SECTION 5 – Revenue at Risk & Campaign ROI
    # =====================
    st.subheader("5. Revenue at Risk & Campaign ROI")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Revenue at Risk (High)", f"£{high_rev:,.2f}")
        st.metric("Revenue at Risk (Medium)", f"£{med_rev:,.2f}")
    projected_med = med_rev * 0.25
    high_high_value_rev = features[(features["risk_tier"] == "High Risk") & (features["monetary"] >= 500)]["monetary"].sum()
    projected_high = high_high_value_rev * 0.15
    with col2:
        st.metric("Projected Medium Recovery (25%)", f"£{projected_med:,.2f}")
        st.metric("Projected High Recovery (15% of high-value)", f"£{projected_high:,.2f}")

    if prev_scores is not None and not prev_scores.empty:
        st.write("**Campaign ROI Tracker**")
        old_med_ids = prev_scores[prev_scores["risk_tier"] == "Medium Risk"]["CustomerID"]
        reactivated = set(old_med_ids) & curr_cust_set
        rev_reactivated = df_this_month[df_this_month["CustomerID"].isin(reactivated)]["Revenue"].sum()
        st.metric("Reactivated Customers (last month Medium → this month active)", len(reactivated))
        st.metric("Revenue Recovered from Reactivated", f"£{rev_reactivated:,.2f}")

    # =====================
    # SECTION 6 – Downloads
    # =====================
    # =====================

    st.subheader("6. Downloads")

    st.caption("All customers scored with churn probability, risk tier, and recommended action.")
    full_csv = convert_df(features.drop(columns=["priority_score"], errors="ignore"))
    st.download_button("📥 Download Full Scored Customer List (CSV)", full_csv,
                       file_name="full_scored_customers.csv", mime="text/csv")

    st.caption("High Risk customers with revenue ≥ £500 only — your Monday morning action list.")
    priority_df = features[(features["risk_tier"] == "High Risk") & (features["monetary"] >= 500)]
    priority_csv = convert_df(priority_df)
    st.download_button("📥 Download Priority Action List (CSV)", priority_csv,
                       file_name="priority_action_list.csv", mime="text/csv")

    st.caption("Complete raw transaction history across all monthly uploads — use this every 6 months to retrain the model.")
    master_csv = convert_df(master)
    st.download_button("📥 Download Master Transactions Database (CSV)", master_csv,
                       file_name="master_transactions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
