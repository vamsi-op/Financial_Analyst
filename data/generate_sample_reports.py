"""
Generate synthetic earnings report PDFs for testing.

Creates realistic-looking quarterly earnings reports with financial data
that exercises all extraction patterns in the financial parser.

Usage:
    python -m data.generate_sample_reports
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Sample report text templates
SAMPLE_REPORTS = {
    "acme_corp_q3_2025": {
        "filename": "ACME_Corp_Q3_2025_Earnings.txt",
        "content": """
ACME CORPORATION
QUARTERLY EARNINGS REPORT
Third Quarter Fiscal Year 2025
For the Period Ended September 30, 2025

═══════════════════════════════════════════════════════════
FINANCIAL HIGHLIGHTS
═══════════════════════════════════════════════════════════

ACME Corporation (NYSE: ACME) today reported financial results for its
fiscal 2025 third quarter ended September 30, 2025.

Revenue was $45.2 billion for the quarter, an increase of 8% compared to
$41.9 billion in the year-ago quarter. On a constant currency basis,
revenue grew 9%.

Net income of $11.3 billion, or $2.85 per diluted share, compared to net
income of $9.8 billion, or $2.46 per diluted share, in the year-ago quarter.

Operating income was $14.7 billion, an increase of 12% year-over-year.

Gross margin was 47.2%, compared to 45.8% in the prior year period.
Operating margin was 32.5%, up from 30.4% last year.

═══════════════════════════════════════════════════════════
CONSOLIDATED STATEMENTS OF OPERATIONS
═══════════════════════════════════════════════════════════

                                    Q3 2025         Q3 2024
                                   (in millions)
Revenue                            $45,200         $41,900
Cost of Revenue                    $23,866         $22,710
Gross Profit                       $21,334         $19,190

Operating Expenses:
  Research & Development            $4,120          $3,890
  Sales & Marketing                 $2,510          $2,430

Operating Income                   $14,704         $12,870

Interest & Other                     ($320)          ($290)

Income Before Taxes                $14,384         $12,580
Income Tax Provision                $3,084          $2,780

Net Income                         $11,300          $9,800

Earnings Per Share:
  Basic                              $2.88           $2.49
  Diluted                            $2.85           $2.46

═══════════════════════════════════════════════════════════
BALANCE SHEET HIGHLIGHTS
═══════════════════════════════════════════════════════════

Total Assets                       $285.6 billion
Total Liabilities                  $198.3 billion
Total Debt                          $87.5 billion
Stockholders' Equity                $87.3 billion

═══════════════════════════════════════════════════════════
CASH FLOW
═══════════════════════════════════════════════════════════

Cash provided by operating activities was $16.8 billion.
Free cash flow was $14.2 billion.

Capital Expenditures were $2.6 billion.

═══════════════════════════════════════════════════════════
MANAGEMENT COMMENTARY
═══════════════════════════════════════════════════════════

"We are pleased with our strong third quarter results, which reflect
broad-based demand across our product portfolio," said Jane Smith, CEO of
ACME Corporation. "Revenue grew 8% year-over-year, driven by strength in
our cloud services and enterprise segments."

"Our cloud business continues to be our primary growth driver, with
annual recurring revenue exceeding $20 billion for the first time. We are
also seeing strong traction in our AI-powered solutions, which contributed
$3.2 billion in revenue this quarter."

GROWTH DRIVERS:
- Cloud Services revenue grew 22% to $12.8 billion
- AI Solutions revenue of $3.2 billion, up 45% YoY
- Enterprise segment grew 11% to $18.5 billion
- International revenue grew 14% on constant currency basis

CHALLENGES:
- Consumer segment declined 3% due to macroeconomic headwinds
- Supply chain costs increased 5% due to semiconductor shortages
- Competitive pressure in the mid-market segment

OUTLOOK:
For Q4 2025, the company expects revenue in the range of $46-48 billion,
representing year-over-year growth of 7-11%. The company also expects
operating margin expansion of 100-150 basis points.

═══════════════════════════════════════════════════════════
RISK FACTORS
═══════════════════════════════════════════════════════════

The company faces certain risks including:
- Macroeconomic uncertainty affecting consumer spending
- Foreign currency exchange rate fluctuations
- Increasing regulatory requirements in key markets
- Competition for talent in the technology sector
- Supply chain disruptions

Despite these challenges, management believes the company is well-positioned
for sustainable long-term growth.
""",
    },
    "acme_corp_q2_2025": {
        "filename": "ACME_Corp_Q2_2025_Earnings.txt",
        "content": """
ACME CORPORATION
QUARTERLY EARNINGS REPORT
Second Quarter Fiscal Year 2025
For the Period Ended June 30, 2025

Revenue was $43.1 billion for the quarter, an increase of 6% compared to
$40.7 billion in the year-ago quarter.

Net income of $10.5 billion, or $2.64 per diluted share.

Operating income was $13.8 billion.

Gross margin was 46.5%.
Operating margin was 32.0%.

Total debt was $89.2 billion. Total assets of $280.1 billion.
Total liabilities was $196.7 billion.

Cash provided by operating activities was $15.9 billion.
Free cash flow was $13.4 billion.
""",
    },
    "edge_case_negative": {
        "filename": "Edge_Case_Negative_Report.txt",
        "content": """
STRUGGLING INDUSTRIES INC.
QUARTERLY EARNINGS REPORT
Q4 Fiscal Year 2025

Revenue was $2.1 billion, a decrease of 15% from $2.5 billion.

Net income was ($340 million), compared to net income of $120 million.

Operating income was ($280 million).

EPS was ($0.85) per diluted share.

Operating cash flow of ($150 million).

Gross margin was 28.3%, down from 35.2%.
Operating margin was -13.3%.

Total debt of $4.8 billion, increased from $3.2 billion.
Total assets was $8.5 billion.
Total liabilities was $7.1 billion.

Management expressed concerns about the deteriorating market conditions
and announced a restructuring plan to reduce costs by $500 million
annually.
""",
    },
}


def generate_reports(output_dir: str = None):
    """Generate sample reports as text files."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "sample")

    os.makedirs(output_dir, exist_ok=True)

    for report_id, report_data in SAMPLE_REPORTS.items():
        filepath = os.path.join(output_dir, report_data["filename"])
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_data["content"])
        print(f"  ✅ Created: {filepath}")

    # Also save as a JSON dataset for programmatic use
    dataset_path = os.path.join(output_dir, "sample_kpis.json")
    import json

    sample_kpis = {
        "ACME_Corp_Q3_2025": {
            "revenue": "$45.2 billion",
            "net_income": "$11.3 billion",
            "operating_income": "$14.7 billion",
            "eps": "$2.85",
            "cash_flow": "$16.8 billion",
            "free_cash_flow": "$14.2 billion",
            "gross_margin": "47.2%",
            "operating_margin": "32.5%",
            "total_debt": "$87.5 billion",
            "total_assets": "$285.6 billion",
            "total_liabilities": "$198.3 billion",
        },
        "ACME_Corp_Q2_2025": {
            "revenue": "$43.1 billion",
            "net_income": "$10.5 billion",
            "operating_income": "$13.8 billion",
            "eps": "$2.64",
            "cash_flow": "$15.9 billion",
            "free_cash_flow": "$13.4 billion",
            "gross_margin": "46.5%",
            "operating_margin": "32.0%",
            "total_debt": "$89.2 billion",
            "total_assets": "$280.1 billion",
            "total_liabilities": "$196.7 billion",
        },
    }

    with open(dataset_path, "w") as f:
        json.dump(sample_kpis, f, indent=2)
    print(f"  ✅ Created: {dataset_path}")


if __name__ == "__main__":
    print("Generating sample reports...")
    generate_reports()
    print("Done!")
