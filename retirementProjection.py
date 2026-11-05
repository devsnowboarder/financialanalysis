import json
import os
import random


def load_parameters(filename="retirement_config.json"):
    default_config = {
        "client_name": "John Doe",
        "current_age": 66,
        "savings": 500000.0,
        "checking_account": 0.0,
        "monthly_ss": 3500.0,
        "monthly_withdrawal": 1500.0,
        "monthly_expenses": 2500.0,
        "inflation_rate": 0.03,
        "mean_return": 0.07,
        "volatility": 0.15,
        "years_projected": 30,
        "health_shock_base": 50000.0
    }

    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump(default_config, file, indent=4)
        return default_config

    with open(filename, 'r') as file:
        return json.load(file)


def estimate_federal_tax(annual_ss, ordinary_withdrawal, inflation_factor, is_married=False):
    """
    Estimates annual US Federal Income Tax on Social Security + pre-tax retirement withdrawals.
    Uses IRS Provisional Income rules and inflation-indexed standard deductions & brackets.
    """
    # 1. Calculate Provisional Income (unindexed statutory IRS thresholds)
    provisional_income = ordinary_withdrawal + (0.5 * annual_ss)

    t1 = 32000.0 if is_married else 25000.0
    t2 = 44000.0 if is_married else 34000.0
    max_t1_taxable = 6000.0 if is_married else 4500.0

    if provisional_income <= t1:
        taxable_ss = 0.0
    elif provisional_income <= t2:
        taxable_ss = min(0.5 * annual_ss, 0.5 * (provisional_income - t1))
    else:
        taxable_ss = min(
            0.85 * annual_ss,
            0.85 * (provisional_income - t2) + min(0.5 * annual_ss, max_t1_taxable)
        )

    agi = ordinary_withdrawal + taxable_ss

    # 2. Apply Standard Deduction (65+ base indexed for inflation)
    base_std_deduction = 33100.0 if is_married else 16550.0
    std_deduction = base_std_deduction * inflation_factor
    taxable_income = max(0.0, agi - std_deduction)

    # 3. Apply Marginal Tax Brackets (indexed for inflation)
    if is_married:
        brackets = [
            (23200.0 * inflation_factor, 0.10),
            (94300.0 * inflation_factor, 0.12),
            (201050.0 * inflation_factor, 0.22),
            (383900.0 * inflation_factor, 0.24),
            (float('inf'), 0.32)
        ]
    else:
        brackets = [
            (11600.0 * inflation_factor, 0.10),
            (47150.0 * inflation_factor, 0.12),
            (100525.0 * inflation_factor, 0.22),
            (191950.0 * inflation_factor, 0.24),
            (float('inf'), 0.32)
        ]

    tax = 0.0
    prev_limit = 0.0
    for limit, rate in brackets:
        if taxable_income > prev_limit:
            taxable_in_bracket = min(taxable_income, limit) - prev_limit
            tax += taxable_in_bracket * rate
            prev_limit = limit
        else:
            break

    return tax, taxable_ss, taxable_income


def calculate_retirement_runway():
    config = load_parameters()

    client_name = config.get("client_name", "John Doe")
    start_age = int(config.get("current_age", 66))

    start_invested_savings = float(config.get("savings", 500000.0))
    start_checking_account = float(config.get("checking_account", 0.0))

    start_monthly_ss = float(config.get("monthly_ss", 3500.0))
    start_monthly_withdrawal = float(config.get("monthly_withdrawal", 1500.0))
    start_monthly_expenses = float(config.get("monthly_expenses", 2500.0))

    inflation_rate = float(config.get("inflation_rate", 0.03))
    mean_return = float(config.get("mean_return", 0.07))
    volatility = float(config.get("volatility", 0.15))
    years_projected = int(config.get("years_projected", 30))
    health_shock_base = float(config.get("health_shock_base", 50000.0))

    irs_uniform_table = {
        75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
        80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8,
        85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
        90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5,
        95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4
    }

    num_simulations = 1000
    successful_runs = 0
    all_simulations = []

    for sim in range(num_simulations):
        invested_savings = start_invested_savings
        checking_account = start_checking_account
        current_age = start_age
        monthly_ss = start_monthly_ss
        monthly_withdrawal = start_monthly_withdrawal
        monthly_expenses = start_monthly_expenses

        sim_history = []
        is_successful = True

        sim_peak_net_worth = invested_savings + checking_account
        sim_max_drawdown = 0.0

        for year in range(1, years_projected + 1):
            inflation_factor = (1 + inflation_rate) ** (year - 1)
            annual_ss = monthly_ss * 12
            planned_withdrawal = monthly_withdrawal * 12
            annual_expenses = monthly_expenses * 12

            rmd_amount = 0.0
            if current_age >= 75:
                divisor = irs_uniform_table.get(current_age, 6.4)
                rmd_amount = invested_savings / divisor

            actual_withdrawal = max(planned_withdrawal, rmd_amount)

            # Calculate Annual Federal Taxes (Single vs Married)
            fed_tax_single, taxable_ss_single, taxable_inc_single = estimate_federal_tax(
                annual_ss, actual_withdrawal, inflation_factor, is_married=False
            )
            fed_tax_married, _, _ = estimate_federal_tax(
                annual_ss, actual_withdrawal, inflation_factor, is_married=True
            )

            # Monthly Affordability Calculations
            gross_monthly_income = (annual_ss + actual_withdrawal) / 12.0
            net_monthly_affordable = gross_monthly_income - (fed_tax_single / 12.0)
            monthly_discretionary_surplus = net_monthly_affordable - monthly_expenses

            shock_cost = 0.0
            if current_age >= 80 and random.random() < 0.15:
                shock_cost = health_shock_base * ((1 + inflation_rate) ** year)

            annual_return = random.gauss(mean_return, volatility)
            investment_growth = invested_savings * annual_return

            invested_savings = invested_savings + investment_growth - actual_withdrawal
            if invested_savings < 0: invested_savings = 0

            checking_cash_flow = (annual_ss + actual_withdrawal) - (annual_expenses + shock_cost)
            checking_account += checking_cash_flow

            total_net_worth = invested_savings + checking_account

            if total_net_worth > sim_peak_net_worth:
                sim_peak_net_worth = total_net_worth

            if sim_peak_net_worth > 0:
                current_drawdown = (sim_peak_net_worth - total_net_worth) / sim_peak_net_worth
                if current_drawdown > sim_max_drawdown:
                    sim_max_drawdown = current_drawdown

            sim_history.append({
                "year": year,
                "age": current_age,
                "annual_return": annual_return,
                "annual_ss": annual_ss,
                "rmd_amount": rmd_amount,
                "actual_withdrawal": actual_withdrawal,
                "monthly_expenses": monthly_expenses,
                "gross_monthly_income": gross_monthly_income,
                "net_monthly_affordable": net_monthly_affordable,
                "monthly_discretionary_surplus": monthly_discretionary_surplus,
                "fed_tax_single": fed_tax_single,
                "fed_tax_married": fed_tax_married,
                "taxable_inc_single": taxable_inc_single,
                "shock_cost": shock_cost,
                "invested_savings": invested_savings,
                "checking_account": checking_account,
                "total_net_worth": total_net_worth
            })

            if total_net_worth <= 0:
                is_successful = False
                break

            current_age += 1
            monthly_expenses += (monthly_expenses * inflation_rate)
            monthly_ss += (monthly_ss * inflation_rate)
            monthly_withdrawal += (monthly_withdrawal * inflation_rate)

        if is_successful:
            successful_runs += 1

        all_simulations.append({
            "final_net_worth": total_net_worth if is_successful else 0,
            "max_drawdown": sim_max_drawdown,
            "history": sim_history
        })

    success_probability = (successful_runs / num_simulations) * 100
    absolute_worst_drawdown = max(sim["max_drawdown"] for sim in all_simulations) * 100

    all_simulations.sort(key=lambda x: x["final_net_worth"])
    median_simulation = all_simulations[num_simulations // 2]
    median_history = median_simulation["history"]

    chart_years = [f"Year {data['year']} (Age {data['age']})" for data in median_history]
    chart_invested = [round(data['invested_savings'], 2) for data in median_history]
    chart_checking = [round(data['checking_account'], 2) for data in median_history]
    chart_net_worth = [round(data['total_net_worth'], 2) for data in median_history]

    # --- HTML GENERATION ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Retirement Projection: {client_name}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: lightblue; color: #333; max-width: 1200px; margin-left: auto; margin-right: auto; }}
            .header-container {{ text-align: center; margin-bottom: 30px; }}
            h1 {{ color: #111827; margin-bottom: 5px; }}
            h2 {{ color: #4b5563; margin-top: 0; font-weight: normal; font-size: 1.2em; }}
            .success-badge {{ display: inline-block; background-color: #10b981; color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 1.1em; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .chart-container {{ width: 100%; height: 400px; margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); box-sizing: border-box; }}
            table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; font-size: 0.95em; margin-bottom: 30px; }}
            th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #e5e7eb; }}
            th {{ background-color: #2563eb; color: #ffffff; font-weight: 600; font-size: 0.9em; }}
            tr:hover {{ background-color: #f3f4f6; }}
            .shock-row {{ background-color: #fee2e2 !important; }}
            .summary {{ margin-top: 20px; padding: 15px; background-color: #dbeafe; border-left: 5px solid #2563eb; font-size: 1.1em; border-radius: 4px; line-height: 1.5; }}
            .depleted {{ color: #dc2626; font-weight: bold; background-color: #fee2e2; border-left: 5px solid #dc2626; }}
            .recommendations {{ margin-top: 30px; margin-bottom: 40px; padding: 25px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; }}
            .recommendations h3 {{ color: #111827; margin-top: 0; font-size: 1.2em; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
            .recommendations ul {{ line-height: 1.7; color: #374151; padding-left: 20px; }}
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>Retirement Projection: {client_name}</h1>
            <h2>Monte Carlo Analysis: 1,000 Simulated Markets</h2>
            <div class="success-badge">
                In {success_probability:.1f}% of simulated markets, your money lasted {years_projected} years.
            </div>
        </div>

        <div class="chart-container">
            <canvas id="savingsChart"></canvas>
        </div>

        <h3 style="color: #111827;">Detailed Breakdown: 50th Percentile (Median) Market Scenario</h3>
        <table>
            <tr>
                <th>Year</th>
                <th>Age</th>
                <th>Market Return</th>
                <th>RMD Taken</th>
                <th>Health Shock</th>
                <th>Invested Savings</th>
                <th>Checking Account</th>
                <th>Total Net Worth</th>
            </tr>
    """

    for data in median_history:
        shock_display = f"-${data['shock_cost']:,.2f}" if data['shock_cost'] > 0 else "$0.00"
        rmd_display = f"${data['rmd_amount']:,.2f}" if data['rmd_amount'] > 0 else "N/A"
        return_color = "#10b981" if data['annual_return'] >= 0 else "#dc2626"
        row_class = "class='shock-row'" if data['shock_cost'] > 0 else ""

        html_content += f"""
            <tr {row_class}>
                <td style="text-align: center;">{data['year']}</td>
                <td style="text-align: center; font-weight: bold;">{data['age']}</td>
                <td style="color: {return_color}; font-weight: bold;">{data['annual_return'] * 100:.2f}%</td>
                <td style="color: {'#d97706' if data['rmd_amount'] > (start_monthly_withdrawal * 12) else '#333'};">{rmd_display}</td>
                <td style="color: {'#dc2626' if data['shock_cost'] > 0 else '#333'};">{shock_display}</td>
                <td>${data['invested_savings']:,.2f}</td>
                <td>${data['checking_account']:,.2f}</td>
                <td style="font-weight: bold;">${data['total_net_worth']:,.2f}</td>
            </tr>
        """

    html_content += "</table>"

    if len(median_history) == years_projected:
        html_content += f"""
        <div class="summary">
            <strong>Monte Carlo Conclusion:</strong> Based on 1,000 randomized market simulations, your portfolio has a <strong>{success_probability:.1f}% probability of success</strong>.<br>
            The table and chart above display the median trajectory (the 50th percentile run). In this specific scenario, your final net worth at the end of {years_projected} years is <strong>${median_history[-1]['total_net_worth']:,.2f}</strong>.
        </div>
        """
    else:
        html_content += f"""
        <div class="summary depleted">
            <strong>Monte Carlo Conclusion:</strong> Based on 1,000 randomized market simulations, your portfolio has a <strong>{success_probability:.1f}% probability of success</strong>.<br>
            The table above displays the median trajectory, which unfortunately depleted in Year {median_history[-1]['year']} at age {median_history[-1]['age']}.
        </div>
        """

    html_content += f"""
        <div class="recommendations">
            <h3>Risk, Volatility & Stress Testing Analysis</h3>
            <ul>
                <li><strong>Volatility & Market Movement:</strong> Your portfolio has an assumed standard deviation (volatility) of <strong>{volatility * 100:.1f}%</strong> around a mean expected return of <strong>{mean_return * 100:.1f}%</strong>. Statistically, this means that in roughly two-thirds of years, your returns are expected to swing anywhere between {(mean_return - volatility) * 100:.1f}% and {(mean_return + volatility) * 100:.1f}%. Mentally prepare for these normal fluctuations—they are the "cost of admission" for long-term growth.</li>
                <li><strong>Stress Testing & Worst-Case Scenario:</strong> We tracked the <em>Maximum Drawdown</em> (the largest percentage drop from a portfolio high-water mark to a trough) across all 1,000 simulations. The absolute worst-case scenario modeled a peak-to-trough decline of <strong>{absolute_worst_drawdown:.1f}%</strong>. Acknowledging this severe stress-test limit helps you mentally brace for severe bear markets so you can stay the course without panic-selling.</li>
            </ul>
        </div>

        <div class="recommendations">
            <h3>Asset Allocation & Portfolio Strategy</h3>
            <ul>
                <li><strong>Risk Adjustments (Dialing Volatility Up or Down):</strong> If you wish to pursue <em>Aggressive Growth</em>, consider shifting capital toward small-cap equities, emerging markets, or growth-oriented technology sectors—this increases your portfolio's beta and expected return, but significantly widens the potential drawdown. Conversely, for <em>Capital Preservation</em>, increase allocations to short-duration fixed income, dividend-paying blue-chip stocks, and cash equivalents to lower overall volatility and protect your principal during market corrections.</li>
                <li><strong>Fixed Income & Yield Strategy:</strong> In the current interest rate environment, consider building a "bond ladder" utilizing short-to-intermediate term U.S. Treasuries (e.g., 1- to 5-year durations). This captures attractive risk-free yields while minimizing duration risk (sensitivity to unexpected rate hikes). High-quality investment-grade corporate bonds or tax-exempt municipal bonds (if you fall into a higher tax bracket) can also provide a reliable income floor to buffer against equity market shocks.</li>
            </ul>
        </div>

        <div class="recommendations">
            <h3>Strategic Next Steps & Recommendations</h3>
            <ul>
                <li><strong>Tax Optimization & Gain Harvesting:</strong> Because your Social Security income covers your baseline expenses, your recognized taxable income is remarkably low. Consider strategically harvesting capital gains in your brokerage account to lock in the 0% federal tax bracket.</li>
                <li><strong>Estate Planning & Asset Titling:</strong> Ensure your assets—particularly any real estate properties or high-value vehicles—are formally titled within a Revocable Living Trust to avoid lengthy and expensive California probate courts. Verify that all your investment accounts have primary and contingent beneficiaries named.</li>
                <li><strong>Healthcare & Travel Coverage:</strong> We have modeled a potential long-term care shock in your 80s. Continue reviewing your Medicare and supplemental coverage annually during open enrollment to ensure it meets your evolving medical needs and provides adequate emergency coverage for any extended international travel.</li>
            </ul>
        </div>
    """

    # --- BOTTOM SECTION: MONTHLY AFFORDABILITY & ANNUAL FEDERAL TAX SCHEDULE ---
    y1 = median_history[0]
    html_content += f"""
        <div class="recommendations">
            <h3>Monthly Affordability & Year-End Federal Tax Breakdown</h3>
            <ul>
                <li><strong>How Much You Can Afford Each Month (Year 1):</strong> Your gross monthly cash inflow is <strong>${y1['gross_monthly_income']:,.2f}</strong> (${start_monthly_ss:,.2f} Social Security + ${start_monthly_withdrawal:,.2f} withdrawal). After setting aside <strong>${(y1['fed_tax_single'] / 12):,.2f}/month</strong> for estimated Single-filer federal taxes, your <strong>Total Net Spendable Budget is ${y1['net_monthly_affordable']:,.2f}/month</strong>. Because your baseline living expenses are <strong>${y1['monthly_expenses']:,.2f}/month</strong>, you have an extra <strong>${y1['monthly_discretionary_surplus']:,.2f}/month</strong> of discretionary cushion.</li>
                <li><strong>Annual Year-End Federal Taxes:</strong> Assuming withdrawals come from a traditional pre-tax retirement account (IRA/401k), IRS "provisional income" rules cause a portion of your Social Security benefits to become taxable. In Year 1, a <strong>Single filer (65+)</strong> faces an estimated year-end Federal tax liability of <strong>${y1['fed_tax_single']:,.2f}</strong> (on ${y1['taxable_inc_single']:,.2f} of taxable income). A <strong>Married Filing Jointly</strong> couple owes <strong>${y1['fed_tax_married']:,.2f}</strong> in Year 1 due to the higher standard deduction. Note that once Required Minimum Distributions (RMDs) kick in at age 75, your forced withdrawals and annual year-end tax bill will increase accordingly.</li>
            </ul>

            <h4 style="color: #111827; margin-top: 25px; margin-bottom: 10px;">Year-by-Year Monthly Affordability & Estimated Federal Tax Schedule (Median Scenario)</h4>
            <table>
                <tr>
                    <th>Year</th>
                    <th>Age</th>
                    <th>Gross Monthly Income</th>
                    <th>Net Monthly Affordable (After Tax)</th>
                    <th>Planned Monthly Expenses</th>
                    <th>Monthly Surplus Cushion</th>
                    <th>Year-End Fed Tax (Single)</th>
                    <th>Year-End Fed Tax (Married)</th>
                </tr>
    """

    for data in median_history:
        html_content += f"""
                <tr>
                    <td style="text-align: center;">{data['year']}</td>
                    <td style="text-align: center; font-weight: bold;">{data['age']}</td>
                    <td>${data['gross_monthly_income']:,.2f}</td>
                    <td style="font-weight: bold; color: #2563eb;">${data['net_monthly_affordable']:,.2f}</td>
                    <td>${data['monthly_expenses']:,.2f}</td>
                    <td style="color: #10b981; font-weight: bold;">+${data['monthly_discretionary_surplus']:,.2f}</td>
                    <td style="color: #dc2626; font-weight: bold;">${data['fed_tax_single']:,.2f}</td>
                    <td style="color: #4b5563;">${data['fed_tax_married']:,.2f}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    """

    html_content += f"""
        <script>
            const ctx = document.getElementById('savingsChart').getContext('2d');
            const savingsChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(chart_years)},
                    datasets: [
                        {{
                            label: 'Total Net Worth',
                            data: {json.dumps(chart_net_worth)},
                            borderColor: '#111827',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            fill: false,
                            tension: 0.3
                        }},
                        {{
                            label: 'Invested Savings (Randomized Returns)',
                            data: {json.dumps(chart_invested)},
                            borderColor: '#2563eb',
                            backgroundColor: 'rgba(37, 99, 235, 0.1)',
                            borderWidth: 3,
                            pointRadius: 4,
                            fill: true,
                            tension: 0.3
                        }},
                        {{
                            label: 'Cash Checking (0%)',
                            data: {json.dumps(chart_checking)},
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 3,
                            pointRadius: 4,
                            fill: true,
                            tension: 0.3
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, ticks: {{ callback: function(value) {{ return '$' + value.toLocaleString(); }} }} }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    let label = context.dataset.label || '';
                                    if (label) {{ label += ': '; }}
                                    if (context.parsed.y !== null) {{
                                        label += new Intl.NumberFormat('en-US', {{ style: 'currency', currency: 'USD' }}).format(context.parsed.y);
                                    }}
                                    return label;
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    return html_content


if __name__ == "__main__":
    html_output = calculate_retirement_runway()
    output_filename = "retirement_projection.html"

    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(html_output)

    print(f"Script processed successfully. Saved report to '{output_filename}'.")