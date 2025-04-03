import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Home Loan EMI Calculator", layout="wide")

# Title
st.title("Home Loan EMI Calculator with Part Payment Options")

# Function to calculate EMI
def calculate_emi(principal, rate, tenure_years):
    rate_monthly = rate / (12 * 100)  # Convert annual rate to monthly and percentage to decimal
    tenure_months = tenure_years * 12
    emi = principal * rate_monthly * (1 + rate_monthly) ** tenure_months / ((1 + rate_monthly) ** tenure_months - 1)
    return emi

# Create amortization schedule
def create_amortization_schedule(principal, interest_rate, tenure_years, start_date, actual_emi=None, 
                                additional_payment=0, part_payment_effect='reduce_tenure'):
    monthly_rate = interest_rate / (12 * 100)
    total_months = tenure_years * 12
    
    # Calculate the standard EMI
    standard_emi = calculate_emi(principal, interest_rate, tenure_years)
    
    # If actual EMI is not provided or less than standard EMI, use standard EMI
    if actual_emi is None or actual_emi < standard_emi:
        actual_emi = standard_emi
    
    # Initialize variables
    remaining_principal = principal
    schedule = []
    month_count = 0
    current_date = start_date
    
    total_interest_paid = 0
    original_tenure_months = total_months
    
    # For reduce EMI mode
    next_recalculation = False
    current_emi = actual_emi
    
    # Generate the schedule
    while remaining_principal > 0 and month_count < 1000:  # 1000 as a safety limit
        month_count += 1
        
        # Calculate interest for this month
        interest_payment = remaining_principal * monthly_rate
        
        # If we need to recalculate EMI (reduce EMI mode)
        if next_recalculation and part_payment_effect == 'reduce_emi':
            # Recalculate EMI based on remaining principal and remaining tenure
            remaining_months = original_tenure_months - month_count + 1
            if remaining_months > 0:
                current_emi = calculate_emi(remaining_principal, interest_rate, remaining_months/12)
                next_recalculation = False
        
        # Extra payment towards EMI (if any)
        extra_emi = max(0, actual_emi - current_emi) if part_payment_effect == 'reduce_tenure' else 0
        
        # Total extra payment (extra EMI + additional principal payment)
        total_extra_payment = extra_emi + additional_payment
        
        # Standard principal component of EMI
        standard_principal_payment = min(current_emi - interest_payment, remaining_principal)
        
        # Calculate total principal payment (including extra payments)
        total_principal_payment = min(standard_principal_payment + total_extra_payment, remaining_principal)
        
        # Update remaining principal
        remaining_principal -= total_principal_payment
        
        # If additional payment was made, flag for EMI recalculation next month
        if (additional_payment > 0 or extra_emi > 0) and part_payment_effect == 'reduce_emi':
            next_recalculation = True
        
        # Total payment this month
        total_payment = interest_payment + total_principal_payment
        
        # Update running total of interest paid
        total_interest_paid += interest_payment
        
        # Add row to schedule
        schedule.append({
            'Month': month_count,
            'Date': current_date.strftime('%b %Y'),
            'EMI Payment': round(current_emi if month_count > 1 and part_payment_effect == 'reduce_emi' else actual_emi, 2),
            'Additional Payment': round(additional_payment, 2),
            'Total Payment': round(total_payment, 2),
            'Interest': round(interest_payment, 2),
            'Principal': round(total_principal_payment, 2),
            'Extra Principal': round(total_extra_payment, 2),
            'Remaining Principal': round(remaining_principal, 2)
        })
        
        # Increment date to next month
        current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        
        # If remaining principal is very small, just close it out
        if remaining_principal < 1:
            remaining_principal = 0
    
    # Create a DataFrame
    df = pd.DataFrame(schedule)
    
    # Calculate summary
    if part_payment_effect == 'reduce_tenure':
        months_saved = original_tenure_months - month_count
        if actual_emi > standard_emi or additional_payment > 0:
            interest_saved = (original_tenure_months * standard_emi) - (total_interest_paid + principal)
        else:
            interest_saved = 0
    else:  # reduce_emi
        months_saved = 0  # Tenure remains the same
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

# Create sidebar for inputs
with st.sidebar:
    st.header("Loan Details")
    
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, value=5000000, step=100000)
    interest_rate = st.number_input("Interest Rate (% per annum)", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
    loan_tenure = st.number_input("Loan Tenure (years)", min_value=1, max_value=30, value=20)
    loan_start_date = st.date_input("Loan Start Date", datetime.now())
    
    st.header("Payment Options")
    
    # Calculate standard EMI based on inputs
    standard_emi = calculate_emi(loan_amount, interest_rate, loan_tenure)
    st.write(f"Standard EMI: ₹ {standard_emi:.2f}")
    
    actual_emi = st.number_input("Actual EMI Payment (₹)", min_value=float(standard_emi), value=float(standard_emi), step=1000.0)
    additional_payment = st.number_input("Additional Principal Payment (₹/month)", min_value=0.0, value=0.0, step=1000.0)
    
    part_payment_effect = st.radio(
        "Part Payment Effect",
        options=["reduce_tenure", "reduce_emi"],
        format_func=lambda x: "Reduce Tenure" if x == "reduce_tenure" else "Reduce EMI"
    )
    
    calculate_btn = st.button("Calculate", type="primary", use_container_width=True)

# Main area
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

if calculate_btn:
    st.session_state.calculated = True
    st.session_state.df, st.session_state.summary = create_amortization_schedule(
        loan_amount, interest_rate, loan_tenure, loan_start_date, 
        actual_emi, additional_payment, part_payment_effect
    )

if st.session_state.calculated:
    df = st.session_state.df
    summary = st.session_state.summary
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Amortization Table", "Charts"])
    
    with tab1:
        # Display summary in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loan Information")
            st.write(f"Original Loan Amount: ₹ {summary['Total Principal']:,.2f}")
            st.write(f"Interest Rate: {interest_rate}% per annum")
            st.write(f"Original EMI: ₹ {summary['Original EMI']:,.2f}")
            st.write(f"Initial EMI Payment: ₹ {summary['Initial EMI']:,.2f}")
            st.write(f"Final EMI Payment: ₹ {summary['Final EMI']:,.2f}")
            st.write(f"Part Payment Effect: {summary['Part Payment Effect']}")
            
        with col2:
            st.subheader("Loan Term & Savings")
            st.write(f"Original Loan Term: {summary['Original Loan Tenure']} months ({loan_tenure} years)")
            st.write(f"Actual Loan Term: {summary['Actual Loan Tenure']} months ({summary['Actual Loan Tenure']/12:.1f} years)")
            
            if summary['Part Payment Effect'] == 'Reducing Tenure':
                st.write(f"Months Saved: {summary['Months Saved']} months ({summary['Months Saved']/12:.1f} years)")
            else:
                st.write(f"EMI Reduction: ₹ {summary['Original EMI'] - summary['Final EMI']:,.2f}")
                
            st.write(f"Total Interest Paid: ₹ {summary['Total Interest Paid']:,.2f}")
            st.write(f"Interest Saved: ₹ {summary['Interest Saved']:,.2f}", 
                    help="Compared to original loan schedule")
            
            # Calculate total payments
            total_payments = summary['Total Principal'] + summary['Total Interest Paid']
            st.write(f"Total Amount Paid: ₹ {total_payments:,.2f}")
        
        # Add a progress meter for loan completion
        st.subheader("Loan Progress")
        loan_progress = min(100, 100 * (df['Month'].max() / summary['Original Loan Tenure']))
        st.progress(loan_progress / 100)
        st.write(f"Loan completion: {loan_progress:.1f}%")
    
    with tab2:
        # Display full amortization schedule
        st.dataframe(df, use_container_width=True)
    
    with tab3:
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Principal vs Interest chart
            fig1 = px.bar(
                df, 
                x='Month', 
                y=['Interest', 'Principal', 'Extra Principal'],
                title='Monthly Payment Breakdown',
                labels={'value': 'Amount (₹)', 'variable': 'Component'},
                color_discrete_map={
                    'Interest': '#FF6B6B', 
                    'Principal': '#4ECDC4',
                    'Extra Principal': '#1A535C'
                },
                barmode='stack'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Outstanding balance chart
            fig2 = px.line(
                df, 
                x='Month', 
                y='Remaining Principal',
                title='Outstanding Loan Balance',
                labels={'Remaining Principal': 'Balance (₹)', 'Month': 'Month Number'},
                color_discrete_sequence=['#4ECDC4']
            )
            fig2.update_traces(line=dict(width=3))
            st.plotly_chart(fig2, use_container_width=True)
        
        # Monthly EMI Chart (especially useful for reduce_emi mode)
        fig3 = px.line(
            df, 
            x='Month', 
            y='EMI Payment',
            title='Monthly EMI Payments',
            labels={'EMI Payment': 'EMI Amount (₹)', 'Month': 'Month Number'},
            color_discrete_sequence=['#FF9F1C']
        )
        fig3.update_traces(line=dict(width=3))
        st.plotly_chart(fig3, use_container_width=True)

else:
    # Show instructions when first loading the page
    st.info("👈 Fill in your loan details in the sidebar and click Calculate to see the amortization schedule.")
    
    # Add some explanations
    st.markdown("""
    ### How to use this calculator:
    
    1. **Enter your loan details** on the left sidebar
    2. **Choose your part payment strategy**:
       - **Reduce Tenure**: Your EMI stays the same, but your loan gets paid off faster
       - **Reduce EMI**: Your loan term stays the same, but your monthly payments decrease
    3. **Click Calculate** to see your loan schedule and savings
    
    ### About part payments:
    
    - **Actual EMI Payment**: If you pay more than the required EMI (e.g., required is ₹37,000 but you pay ₹40,000), the extra ₹3,000 goes toward reducing the principal
    - **Additional Principal Payment**: Any extra amount you pay specifically toward principal reduction
    
    Both types of part payments can either reduce your loan tenure or reduce your EMI payments, depending on which option you select.
    """)

# Add footer
st.markdown("""
---
Made with Streamlit | Data is calculated locally and not stored anywhere
""")