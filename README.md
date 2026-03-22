# E-Commerce Customer Segmentation
### Global RFM Analysis · German Market Deep Dive · CLV · K-Means · Cohort Retention

**Live Dashboard →** [https://rfm-german-customers-hhs7e7zwtqfp9qvjmvrhlm.streamlit.app]  
**Author →** Purnachandar Vallala | MSc Data Science · Germany  
**LinkedIn →** [www.linkedin.com/in/vallala-purnachandar-051314226)

---

## Background

I built this project to answer a question I genuinely had: how do German 
customers behave compared to the global average in e-commerce? Most publicly 
available analyses treat international markets as secondary footnotes. I wanted 
to put Germany at the centre and back it up with statistics, not assumptions.

The dataset is the UCI Online Retail II — real transactional data from a 
UK-based retailer operating across 38 countries between December 2009 and 
December 2011. Over one million rows. Real customer IDs, real invoice dates, 
real products.

The analysis goes through five layers: RFM segmentation, Customer Lifetime 
Value prediction, K-Means clustering as a validation layer, statistical 
hypothesis testing on the Germany vs global comparison, and cohort retention 
analysis across 25 monthly cohorts.

---

## The Market

Before looking at segments, it helps to understand the market structure.

### Scale

| Metric | Value |
|--------|-------|
| Raw transactions | 1,067,371 |
| After cleaning | 805,549 |
| Unique customers | 5,878 |
| Countries | 38 |
| Total revenue | £17,743,429 |
| Date range | December 2009 – December 2011 |

### Revenue by Country (Top 10)

| Country | Transactions | Revenue | Customers | Avg per Customer |
|---------|-------------|---------|-----------|-----------------|
| United Kingdom | 725,250 | £14,723,147 | 3,950+ | £2,751 |
| Germany | 16,694 | £431,262 | 107 | £4,030 |
| EIRE | 15,743 | £621,631 | 182 | £3,447 |
| France | 13,812 | £355,257 | 110 | £3,229 |
| Netherlands | 5,088 | £554,232 | 22 | £25,192 |
| Spain | 3,719 | £109,178 | 31 | £3,522 |
| Belgium | 3,068 | £65,753 | 25 | £2,630 |
| Switzerland | 3,011 | £100,365 | 38 | £2,641 |
| Australia | 1,812 | £169,968 | 15 | £11,331 |
| Portugal | 2,446 | £57,285 | 29 | £1,975 |

### Market Concentration

- The UK accounts for **90.1% of transactions** but only **83% of revenue** 
  — international customers spend proportionally more
- Germany is the **#2 market by transaction volume** internationally
- International customers spend on average **1.4× more** per customer 
  than UK customers
- Germany represents **1.8% of customers** but generates **2.43% of revenue** 
  — already punching above its weight before segmentation

### Customer Value Distribution

The global customer base shows extreme value concentration:

- **Median lifetime spend:** £898
- **Mean lifetime spend:** £3,018 — pulled up significantly by high-value buyers
- **Single highest-value customer:** £608,821 across 398 orders
- **Top 25% of customers (Platinum CLV tier):** generate the majority of revenue

This skew between median and mean is why fixed scoring thresholds would 
fail here. A customer spending £1,500 looks like a high spender against the 
median but falls in the middle of the mean distribution. Quantile-based 
scoring resolves this by ranking customers relative to each other, not against 
an arbitrary number.

---

## Data Cleaning

Real transactional data is messy. Four categories of bad data were identified 
and removed before any analysis:

| Problem | Rows Affected | Reason for Removal |
|---------|--------------|-------------------|
| Missing Customer ID | 243,007 | Cannot assign purchase to any customer |
| Cancelled orders (Invoice starting with C) | 18,744 | Not real purchases |
| Negative or zero quantity | varies | Returns and data entry errors |
| Negative or zero unit price | 71 | Pricing errors |

**Remaining after cleaning: 805,549 transactions across 5,878 customers.**

---

## RFM Segmentation

RFM stands for Recency, Frequency, Monetary. Each customer receives three 
scores based on their transaction history:

| Dimension | Definition | Scoring Direction |
|-----------|-----------|------------------|
| Recency | Days since last purchase | Lower = better (score reversed: 5→1) |
| Frequency | Number of unique orders | Higher = better (score: 1→5) |
| Monetary | Total lifetime spend | Higher = better (score: 1→5) |

Each score runs from 1 (worst) to 5 (best), assigned using quantile 
segmentation — each score represents exactly 20% of the customer base.

**Why quantiles and not fixed thresholds?** The monetary data is right-skewed 
(median £898, max £608,821). Fixed thresholds would cluster most customers 
in the lowest bucket. Quantile scoring ensures each band always contains 
exactly 20% of customers, making the scoring fair and relative to the 
actual population.

### Global Segment Results

| Segment | Customers | Share | Avg Spend | Total Revenue | Revenue Share |
|---------|-----------|-------|-----------|---------------|---------------|
| Champions | 1,300 | 22.1% | £9,329 | £12,128,115 | 68.4% |
| Loyal Customers | 1,134 | 19.3% | £2,295 | £2,603,183 | 14.7% |
| At Risk | 976 | 16.6% | £958 | £935,193 | 5.3% |
| Hibernating | 642 | 10.9% | £208 | £134,005 | 0.8% |
| New Customers | 443 | 7.5% | £890 | £394,638 | 2.2% |
| About To Sleep | 419 | 7.1% | £316 | £132,552 | 0.7% |
| Needs Attention | 381 | 6.5% | £479 | £182,506 | 1.0% |
| Potential Loyalists | 356 | 6.1% | £602 | £214,366 | 1.2% |
| Can't Lose Them | 227 | 3.9% | £4,488 | £1,018,866 | 5.7% |

Champions — 22% of customers — generate **68.4% of all revenue**. 
The Pareto principle confirmed in real data.

The "Can't Lose Them" segment is the most urgent business risk: 227 customers 
who historically spent an average of £4,488 but have not returned in over 
340 days on average. That is over £1 million in revenue that is actively 
at risk of being lost permanently.

---

## Customer Lifetime Value

CLV was calculated using the formula:
```
CLV = (Frequency × Monetary) / Recency
```

This rewards recent, frequent, high-spending customers and penalises 
customers who were valuable in the past but have gone quiet — even if 
their historical spend was high.

| CLV Tier | Customers | Share |
|----------|-----------|-------|
| Platinum | 1,470 | 25% |
| Gold | 1,469 | 25% |
| Silver | 1,469 | 25% |
| Bronze | 1,470 | 25% |

The top customer by CLV: **Customer 12471**, who bought 2 days before the 
reference date, placed 79 orders, spent £39,963, and has a CLV score of 
1,578,569 — approximately 60,000× the global median CLV.

---

## German Market Deep Dive

### Raw Comparison

| Metric | Global | Germany | Difference |
|--------|--------|---------|-----------|
| Avg Recency (days) | 201.3 | 133.1 | −68.2 days |
| Avg Frequency | 6.3 | 7.4 | +1.1 orders |
| Avg Monetary | £3,018 | £4,039 | +£1,020 |
| Median CLV | 26.9 | 97.6 | +70.7 (3.6×) |
| Champions % | 22.1% | 29.9% | +7.8% |
| Hibernating % | 10.9% | 4.7% | −6.2% |

### Statistical Validation

Observing that German customers appear to spend more is not enough. With 
only 107 German customers, there is a real risk of drawing conclusions from 
noise. Two statistical tests were applied for each RFM metric:

| Test | Reasoning |
|------|-----------|
| Independent t-test | Standard parametric benchmark |
| Mann-Whitney U | More appropriate for skewed distributions |

Cohen's d was also calculated to measure practical effect size — not just 
whether a difference exists, but how large it is.

**Results:**

| Metric | p-value | Significant? | Effect Size |
|--------|---------|-------------|-------------|
| Monetary | 0.0000 | Yes | 0.091 (small) |
| Recency | 0.0005 | Yes | 0.360 (small) |
| Frequency | 0.1169 | No | 0.093 |

The frequency result is the most important one to highlight honestly: 
**German customers do not buy significantly more often**. The difference in 
frequency is within the range of random variation. What is statistically 
confirmed is that German customers spend more per order and return more 
recently — they place fewer but larger orders.

This distinction has direct implications for marketing strategy. A campaign 
designed to increase purchase frequency for German customers would be 
targeting a behaviour that isn't actually broken. The opportunity is 
in increasing order value, not order count.

### German Segment Breakdown

| Segment | German Count | German % | Global % |
|---------|-------------|----------|----------|
| Champions | 32 | 29.9% | 22.1% |
| Loyal Customers | 27 | 25.2% | 19.3% |
| New Customers | 13 | 12.1% | 7.5% |
| At Risk | 13 | 12.1% | 16.6% |
| Needs Attention | 7 | 6.5% | 6.5% |
| About To Sleep | 5 | 4.7% | 7.1% |
| Hibernating | 5 | 4.7% | 10.9% |
| Potential Loyalists | 4 | 3.7% | 6.1% |
| Can't Lose Them | 1 | 0.9% | 3.9% |

---

## K-Means Clustering

After building rule-based RFM segments, K-Means clustering was applied 
as an independent validation layer. The question was simple: if an 
algorithm with no knowledge of our rules segments customers from scratch, 
does it arrive at the same groupings?

### Choosing k

The Elbow method tested k from 2 to 10. Mathematically, k=2 produced 
the highest silhouette score (0.916). However, k=2 produces only "high 
value" and "low value" — not actionable for a marketing team. k=4 was 
chosen: silhouette score of 0.59 (above the 0.5 threshold for reasonable 
separation) with four groups distinct enough to carry different strategies.

**Scaling is mandatory before K-Means.** Monetary values range from £3 to 
£608,821; Recency from 1 to 739 days. Without scaling, monetary would 
dominate the distance calculation completely and Recency would become 
invisible. StandardScaler was applied to bring all three features to 
mean=0, standard deviation=1.

### Cluster Results

| Cluster | Customers | Avg Recency | Avg Frequency | Avg Spend |
|---------|-----------|-------------|---------------|-----------|
| Lost / Inactive | 1,998 | 463 days | 2.2 orders | £765 |
| Mid Value Engaged | 3,841 | 67 days | 7.3 orders | £3,009 |
| High Value (Whales) | 35 | 26 days | 103.7 orders | £83,086 |
| Ultra High Value (Mega Whales) | 4 | 4 days | 212.5 orders | £436,836 |

The algorithm independently separated the four ultra-high-value customers 
from the rest — without any instruction to do so. It found the same 
structure the RFM rules identified, which validates both approaches.

---

## Cohort Retention Analysis

Customers were grouped by the month of their first purchase. For each 
cohort, retention was tracked month by month for up to 24 months.

### Key Findings

| Milestone | Average Retention |
|-----------|-----------------|
| Month 0 (first purchase) | 100% |
| Month 1 | 21% |
| Month 2 | 22% |
| Month 6 | 19% |
| Month 12 | 18% |
| Month 24 | 20% |

Two observations stand out:

**Stabilisation after Month 1.** The sharp drop from 100% to 21% between 
Month 0 and Month 1 is expected — not every first-time buyer returns. 
What is notable is that retention barely changes after Month 1. Customers 
who return once tend to keep returning. This means the highest-ROI 
intervention point is the period immediately after the first purchase.

**December spike at Month 12.** Every single cohort shows elevated 
retention at Month 12 relative to Month 11 and Month 13. Customers who 
bought in any month of the year come back in December for Christmas 
shopping. This is consistent across all 25 cohorts and suggests a 
pre-Christmas campaign targeting customers in their 10th–11th month 
would capture this window proactively.

---

## Business Recommendations

### Global

**1. Protect Champions unconditionally**  
1,300 customers generating 68% of revenue cannot be treated as a standard 
segment. Dedicated account management, priority support queues, and 
exclusive early access to new products. The cost of losing one Champion 
is on average £9,329 in annual revenue.

**2. Emergency intervention for Can't Lose Them**  
227 customers, average historical spend £4,488, last purchased over 
340 days ago. This is £1,018,866 in revenue that is actively leaving. 
A personalised win-back campaign with a meaningful discount (20–25%) 
should be the immediate priority — every additional month of inactivity 
reduces re-engagement probability significantly.

**3. Automated trigger for At-Risk customers**  
976 customers showing declining engagement. An automated email sequence 
triggered at 90 days of inactivity — before they move into Hibernating — 
is far cheaper than re-acquisition. Winning back 25% of At-Risk customers 
at their average spend recovers approximately £234,000.

**4. Second-purchase campaign for New Customers**  
443 customers bought once and have not returned. Cohort data confirms 
that getting the second purchase is the critical conversion point — 
retention stabilises significantly once a customer buys twice. A 
personalised follow-up offer within 7 days of first purchase has the 
highest ROI of any new customer intervention.

**5. Pre-Christmas campaign in October**  
Every cohort shows a retention spike at Month 12. Targeting customers 
in their 10th and 11th month with an early Christmas offer in October 
captures this natural shopping behaviour proactively rather than 
reactively.

### Germany-Specific

**1. Dedicated VIP tier for German Champions and Platinum customers**  
32 Champions and 38 Platinum CLV customers. These customers have median 
CLV 3.6× the global median. Free express shipping (delivery reliability 
matters significantly to German consumers), early product access, and 
a dedicated contact point are relatively low-cost interventions for 
customers of this value.

**2. Win-back sequence for 13 At-Risk German customers**  
German-language personalised outreach within 30 days. Escalate to a 
20% discount offer if no engagement within 14 days. At the German 
average spend of £4,039, recovering even half of these customers has 
meaningful revenue impact.

**3. Onboarding sequence for 13 New German customers**  
Day 3: satisfaction check. Day 7: personalised product recommendations 
based on first order. Day 21: 10% discount on second order. 
Target: convert at least 30% to repeat buyers within 45 days.

**4. Increase German marketing allocation by 5–8%**  
Germany generates 3.6× higher CLV per customer than the global median. 
Estimated marketing ROI is 1.4× higher than the UK market. 
Additional allocation should include Klarna, SEPA direct debit, and 
PayPal at checkout — German consumers have a well-documented preference 
for local payment methods and lower credit card adoption than other 
European markets.

**5. Proactive intervention for About-To-Sleep German customers**  
Germany's Hibernating rate is already 4.7% vs 10.9% globally — 
significantly better. This advantage is maintained by catching customers 
before they hibernate, not after. An automated alert at 90 days without 
purchase keeps this metric favourable without requiring reactive 
win-back spend.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| pandas | Data manipulation, aggregation, cleaning |
| numpy | Numerical operations |
| scikit-learn | K-Means clustering, StandardScaler |
| scipy | Mann-Whitney U test, t-test, effect size |
| plotly | Interactive charts and dashboard visualisations |
| matplotlib / seaborn | Heatmaps, box plots, static charts |
| streamlit | Live dashboard deployment |
| reportlab | Automated PDF report generation |

---

## Dashboard

Four pages, fully interactive, filters applied globally across all views.

**Global Overview** — KPI metrics (customers, revenue, Champions count, 
At-Risk count, average customer value), segment bar chart, revenue 
treemap, top 10 countries by revenue, full customer data table with 
CSV export

**German Deep Dive** — Germany vs global segment comparison chart, 
statistical significance results table, top 15 German customers by CLV

**CLV Analysis** — CLV tier distribution for global and German customers 
side by side, top 20 highest-value customers globally

**Cohort Retention** — Full 25-cohort retention heatmap, average 
retention curve with industry benchmark line, seasonal pattern annotation

---

## Project Structure
```
rfm-german-customers/
│
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── RFM_Analysis_Report.pdf     # Auto-generated analysis report
│
├── notebooks/
│   └── rfm_analysis.ipynb      # Full analysis (38 cells)
│
├── data/
│   └── online_retail_II.csv    # UCI Online Retail II
│
├── rfm_heatmap.png             # R vs F score heatmap
├── cohort_retention.png        # 25-cohort retention matrix
├── clv_comparison.png          # CLV tiers: Germany vs Global
├── elbow_curve.png             # K-Means elbow and silhouette
├── kmeans_vs_rfm.png           # Cluster vs segment scatter
├── retention_curve.png         # Average retention over 24 months
└── statistical_tests.png       # Box plots with p-values
```

---

## Run Locally
```bash
git clone https://github.com/Purnachandarvallala/rfm-german-customers.git
cd rfm-german-customers
pip install -r requirements.txt
streamlit run app.py
```

Full notebook:
```bash
jupyter notebook notebooks/rfm_analysis.ipynb
```

---

## Author

**Purnachandar Vallala**  
MSc Data Science · Germany  
[GitHub](https://github.com/Purnachandarvallala) · 
[LinkedIn](www.linkedin.com/in/vallala-purnachandar-051314226)

*Available for Werkstudent and full-time Data Science 
and Data Analyst positions in Germany.*


