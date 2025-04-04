import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import re

# Indian currency formatting (manual)
def format_inr(amount):
    x = f"{int(round(amount))}"
    last_three = x[-3:]
    other = x[:-3]
    if other != '':
        other = re.sub(r"(\d)(?=(\d{2})+(?!\d))", r"\1,", other)
        return f"₹ {other},{last_three}"
    else:
        return f"₹ {last_three}"

st.set_page_config(page_title="Home Loan EMI Calculator", layout="wide")
st.title("🏠 Home Loan EMI Calculator (India)")

# Inputs inside the form
with st.form("loan_form"):
    st.subheader("Enter Loan Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, value=5000000, step=100000)
    with col2:
        interest_rate = st.number_input("Interest Rate (% p.a.)", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
    with col3:
        loan_tenure = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=20)

    col4, col5 = st.columns(2)
    with col4:
        loan_start_date = st.date_input("Loan Start Date", datetime.now())
    with col5:
        actual_emi = st.number_input("Monthly EMI You Plan to Pay (₹)", min_value=0.0, value=0.0, step=1000.0)

    part_payment_effect = st.radio(
        "Prepayment Strategy",
        options=["reduce_tenure", "reduce_emi"],
        format_func=lambda x: "Reduce Tenure" if x == "reduce_tenure" else "Reduce EMI",
        horizontal=True,
        key="strategy"
    )

    submitted = st.form_submit_button("📈 Calculate")

# Part payment section outside the form
st.markdown("---")
part_payment_enabled = st.checkbox("Do you want to make part payments?")
additional_payment = 0.0
yearly_payment = 0.0
prepayment_df = pd.DataFrame(columns=["Month", "Amount (₹)"])
payment_type = None

if part_payment_enabled:
    payment_type = st.radio("Select Part Payment Type", ["Monthly Fixed", "Yearly Fixed", "Custom Months"], key="ptype")

    if payment_type == "Monthly Fixed":
        additional_payment = st.number_input("Fixed Extra Prepayment per Month (₹)", min_value=0.0, value=0.0, step=1000.0, key="monthly")

    elif payment_type == "Yearly Fixed":
        yearly_payment = st.number_input("Fixed Extra Prepayment per Year (₹)", min_value=0.0, value=0.0, step=1000.0, key="yearly")

    elif payment_type == "Custom Months":
        st.markdown("### Enter One-time Prepayment Schedule")
        prepayment_df = st.data_editor(
            pd.DataFrame({"Month": ["Apr 2026"], "Amount (₹)": [100000]}),
            num_rows="dynamic",
            use_container_width=True,
            key="prepay_table"
        )

# Run calculation after form is submitted
if submitted:
    st.markdown("---")
    st.subheader("📊 Calculating Amortization Schedule...")

    def calculate_emi(principal, rate, tenure_years):
        rate_monthly = rate / (12 * 100)
        tenure_months = tenure_years * 12
        emi = principal * rate_monthly * (1 + rate_monthly) ** tenure_months / ((1 + rate_monthly) ** tenure_months - 1)
        return emi

    def create_amortization_schedule(principal, interest_rate, tenure_years, start_date, actual_emi=None, 
                                    additional_payment=0, part_payment_effect='reduce_tenure', prepayment_table=None,
                                    yearly_payment=0):
        monthly_rate = interest_rate / (12 * 100)
        total_months = tenure_years * 12

        standard_emi = calculate_emi(principal, interest_rate, tenure_years)
        if actual_emi is None or actual_emi < standard_emi:
            actual_emi = standard_emi

        prepayment_lookup = {}
        if prepayment_table is not None and not prepayment_table.empty:
            for _, row in prepayment_table.iterrows():
                prepayment_lookup[row['Month']] = row['Amount (₹)']

        remaining_principal = principal
        schedule = []
        month_count = 0
        current_date = start_date
        total_interest_paid = 0
        original_tenure_months = total_months
        next_recalculation = False
        current_emi = actual_emi

        while remaining_principal > 0 and month_count < 1000:
            month_count += 1
            interest_payment = remaining_principal * monthly_rate

            if next_recalculation and part_payment_effect == 'reduce_emi':
                remaining_months = original_tenure_months - month_count + 1
                if remaining_months > 0:
                    current_emi = calculate_emi(remaining_principal, interest_rate, remaining_months / 12)
                    next_recalculation = False

            extra_emi = max(0, actual_emi - current_emi) if part_payment_effect == 'reduce_tenure' else 0
            one_time_prepay = prepayment_lookup.get(current_date.strftime('%b %Y'), 0.0)

            is_yearly_month = current_date.month == start_date.month
            yearly_extra = yearly_payment if is_yearly_month else 0.0

            total_extra_payment = extra_emi + additional_payment + one_time_prepay + yearly_extra

            standard_principal_payment = min(current_emi - interest_payment, remaining_principal)
            total_principal_payment = min(standard_principal_payment + total_extra_payment, remaining_principal)
            remaining_principal -= total_principal_payment

            if (additional_payment > 0 or extra_emi > 0 or one_time_prepay > 0 or yearly_extra > 0) and part_payment_effect == 'reduce_emi':
                next_recalculation = True

            total_payment = interest_payment + total_principal_payment
            total_interest_paid += interest_payment

            schedule.append({
                'Month': month_count,
                'Date': current_date.strftime('%b %Y'),
                'EMI Payment': round(current_emi if month_count > 1 and part_payment_effect == 'reduce_emi' else actual_emi, 2),
                'Additional Payment': round(total_extra_payment, 2),
                'Total Payment': round(total_payment, 2),
                'Interest': round(interest_payment, 2),
                'Principal': round(total_principal_payment, 2),
                'Extra Principal': round(total_extra_payment, 2),
                'Remaining Principal': round(remaining_principal, 2)
            })

            current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            if remaining_principal < 1:
                remaining_principal = 0

        df = pd.DataFrame(schedule)

        if part_payment_effect == 'reduce_tenure':
            months_saved = original_tenure_months - month_count
            if actual_emi > standard_emi or additional_payment > 0:
                interest_saved = (original_tenure_months * standard_emi) - (total_interest_paid + principal)
            else:
                interest_saved = 0
        else:
            months_saved = 0
            original_interest = principal * monthly_rate * original_tenure_months - principal
            interest_saved = original_interest - total_interest_paid

        summary = {
            'Original EMI': round(standard_emi, 2),
            'Initial EMI': round(actual_emi, 2),
            'Final EMI': round(df['EMI Payment'].iloc[-1], 2),
            'Original Loan Tenure': original_tenure_months,
            'Actual Loan Tenure': month_count,
            'Total Principal': principal,
            'Total Interest Paid': round(total_interest_paid, 2),
            'Months Saved': months_saved,
            'Interest Saved': round(interest_saved, 2) if interest_saved > 0 else 0,
            'Part Payment Effect': 'Reducing Tenure' if part_payment_effect == 'reduce_tenure' else 'Reducing EMI'
        }

        return df, summary

    df, summary = create_amortization_schedule(
        loan_amount,
        interest_rate,
        loan_tenure,
        loan_start_date,
        actual_emi=actual_emi,
        additional_payment=additional_payment if part_payment_enabled and payment_type == 'Monthly Fixed' else 0.0,
        yearly_payment=yearly_payment if part_payment_enabled and payment_type == 'Yearly Fixed' else 0.0,
        part_payment_effect=part_payment_effect,
        prepayment_table=prepayment_df if part_payment_enabled and payment_type == 'Custom Months' else None
    )

    st.success("Schedule and summary generated successfully.")

    # Summary section
    col1, col2, col3 = st.columns(3)
    col1.metric("Original EMI", format_inr(summary['Original EMI']))
    col2.metric("Total Interest Paid", format_inr(summary['Total Interest Paid']))
    col3.metric("Actual Tenure", f"{summary['Actual Loan Tenure']} months")

    # Pie chart of payment distribution
    pie_df = pd.DataFrame({
        'Component': ['Principal', 'Interest', 'Prepayments'],
        'Amount': [summary['Total Principal'], summary['Total Interest Paid'], df['Extra Principal'].sum()]
    })
    pie_chart = px.pie(pie_df, values='Amount', names='Component', title='Loan Payment Breakdown')
    st.plotly_chart(pie_chart, use_container_width=True)

    # Line chart for principal and interest trend
    trend_df = df[['Date', 'Principal', 'Interest']].melt(id_vars='Date', var_name='Type', value_name='Amount')
    trend_chart = px.line(trend_df, x='Date', y='Amount', color='Type', title='Monthly Principal & Interest Trend')
    st.plotly_chart(trend_chart, use_container_width=True)

    # Year-wise breakdown table
    df['Year'] = df['Date'].apply(lambda x: x.split()[1])
    yearly = df.groupby('Year').agg({
        'Principal': 'sum',
        'Interest': 'sum',
        'Extra Principal': 'sum'
    }).reset_index()
    yearly.columns = ['Year', 'Principal Paid (₹)', 'Interest Paid (₹)', 'Prepayments (₹)']
    yearly = yearly.round(0).astype({'Principal Paid (₹)': 'int', 'Interest Paid (₹)': 'int', 'Prepayments (₹)': 'int'})
    st.markdown("### Year-wise Payment Summary")
    st.dataframe(yearly, use_container_width=True)

    # File export
    st.markdown("### 📁 Export Schedule")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Amortization Schedule as CSV", csv, "amortization_schedule.csv", "text/csv")
