"""Verbatim custom section bodies for the six generated contracts.

WHY THIS EXISTS
---------------
The contract corpus is deliberately long-form: each agreement carries four fully
custom sections (Scope, Service Levels, Late Delivery Penalties, Invoicing &
Payment) whose distinct, quotable numbers are the ground truth for the Week-4
RAG labs and quizzes. That legal prose dwarfed the generator logic in
``generate_contracts.py``, so it lives here as plain string constants while the
generator keeps only the shared skeleton, the per-contract spec dicts, and the
rendering loop — honoring the "< ~350 lines per file" convention
(ARCHITECTURE.md §7).

EDITING RULES
-------------
- These strings are TEACHING DATA, not boilerplate: every bolded number is a
  quiz answer somewhere (see the numbers table in generate_contracts.py's
  docstring). If you change a number here, update that table, the carrier rows
  seeded by build_database.py, and any lab or quiz that quotes it.
- The markdown structure matters: the ``### 5.x`` / ``### 6.x`` subsection
  headings are exactly what the PageIndex tree builder turns into navigable
  nodes. Keep headings intact or lab06/lab07 retrieval targets shift.

Naming: ``<PARTY>_<SECTION>`` — e.g. ``SWIFTSHIP_PENALTY`` is the body of
SwiftShip's "## 6. Late Delivery Penalties" section. PCC = Pacific Crest
Carriers. Consumed only by ``generate_contracts.py``.
"""

from __future__ import annotations

SWIFTSHIP_SCOPE = """SwiftShip shall provide time-definite express transportation services for GlobalTrade's **Same Day** and **First Class** shipping lanes worldwide, including pickup within two (2) hours of tender for Same Day shipments, airport-to-airport and door-to-door linehaul, coordination of customs brokerage, and real-time milestone tracking (pickup, departure, arrival, out-for-delivery, proof of delivery) via SwiftShip's API. Same Day shipments carry a Scheduled Delivery Date of one (1) day from tender; First Class shipments carry a Scheduled Delivery Date of two (2) days from tender. SwiftShip is GlobalTrade's primary carrier for premium express lanes in all markets. Tender volumes are forecast quarterly by GlobalTrade in good faith but do not constitute guaranteed minimums; SwiftShip shall maintain capacity to absorb peak-season volumes up to 140% of the trailing-quarter average."""

SWIFTSHIP_SLA = """### 5.1 Service Level Framework

SwiftShip's performance is measured monthly against the commitments in this Section 5 using GlobalTrade's tender records and SwiftShip's proof-of-delivery timestamps, reconciled through the monthly scorecard process in Section 5.3. Time-definite express service is the essence of this Agreement: the Parties acknowledge that GlobalTrade sells premium delivery promises to its own customers in reliance on these Service Levels.

### 5.2 On-Time Delivery Commitment

SwiftShip commits to an On-Time Delivery Rate of at least **96.0%** of all shipments tendered in each calendar month, measured separately for the Same Day lane and the First Class lane as well as in aggregate. A shipment is on time only if the proof-of-delivery timestamp falls on or before 23:59 local time at destination on the Scheduled Delivery Date. Shipments excluded from measurement: shipments affected by a Force Majeure Event under Section 9, and shipments where lateness is solely attributable to a GlobalTrade documentation error confirmed through the dispute process.

### 5.3 Measurement & Reporting

SwiftShip shall deliver a monthly performance scorecard by the fifth (5th) Business Day of the following month, broken down by lane, market, and origin-destination pair, including every Late Delivery with its delay in Business Days and root-cause code. GlobalTrade may dispute scorecard entries within ten (10) Business Days; undisputed entries are final for penalty calculation."""

SWIFTSHIP_PENALTY = """### 6.1 Per-Shipment Penalty

For each Late Delivery, SwiftShip shall pay GlobalTrade a penalty equal to **2% of the Shipment Value for each Late Business Day**, beginning on the first Business Day after the Scheduled Delivery Date. There is no grace period: Same Day and First Class are time-definite products and any delivery after the Scheduled Delivery Date is a Late Delivery.

### 6.2 Penalty Cap

The aggregate penalty for any single shipment is capped at **15% of that shipment's Shipment Value**, regardless of the total length of the delay.

### 6.3 Worked Example

A First Class shipment with a Shipment Value of $10,000 is delivered three (3) Business Days after its Scheduled Delivery Date: the penalty is 3 x 2% = 6% of $10,000 = **$600**. Had the same shipment been delivered nine (9) Business Days late, the uncapped penalty of 18% would be reduced to the 15% cap, i.e. **$1,500**.

### 6.4 Set-Off and Exclusions

Penalties accrued in a month are itemized on the monthly scorecard and may be set off by GlobalTrade against SwiftShip's next invoice. No penalty accrues for shipments excluded from measurement under Section 5.2. Penalties under this Section 6 are in addition to, and do not limit, GlobalTrade's termination right for chronic Service Level failure under Section 10(e)."""

SWIFTSHIP_PAYMENT = """SwiftShip shall invoice GlobalTrade monthly in arrears, in US dollars, with one consolidated invoice per calendar month itemizing each shipment by tracking number, lane, Shipment Value, and applicable charges, accompanied by the corresponding scorecard. Undisputed invoices are payable **net thirty (30) days** from the invoice date. GlobalTrade must raise invoice disputes in writing within ten (10) Business Days of receipt, and the Parties shall resolve disputes within thirty (30) days; the undisputed portion remains payable when due. Late payments accrue interest at **1.5% per month** (or the maximum rate permitted by law, if lower) on the overdue undisputed amount. SwiftShip may not suspend services for non-payment unless undisputed amounts remain unpaid sixty (60) days past due and SwiftShip has given ten (10) Business Days' written warning."""

ATLAS_SCOPE = """Atlas shall provide surface and ocean transportation services for GlobalTrade's **Standard Class** shipping lanes, comprising full-truckload and less-than-truckload ground linehaul across North America and intercontinental ocean freight consolidation (LCL and FCL) on trans-Atlantic and trans-Pacific trade lanes, together with drayage, container management, and weekly departure schedules from GlobalTrade's consolidation centers. Standard Class shipments carry a Scheduled Delivery Date of six (6) days from tender. Atlas is GlobalTrade's primary carrier for cost-optimized Standard lanes; GlobalTrade may tender Second Class shipments to Atlas at the rates in Schedule B where capacity allows. Atlas shall provide electronic status updates at pickup, port arrival/departure, customs clearance, and delivery."""

ATLAS_SLA = """### 5.1 Service Level Framework

Atlas's performance is measured against the commitments in this Section 5 on a calendar-month basis across all Standard Class lanes in aggregate, using the delivery timestamps recorded in Atlas's transport management system and reconciled to GlobalTrade's tender records.

### 5.2 On-Time Delivery Commitment

Atlas commits to an On-Time Delivery Rate of at least **92.5%** of all shipments tendered in each calendar month. The Parties acknowledge that Standard Class is a cost-optimized, non-express product and that the schedule already incorporates normal variability in ground and ocean transit; the commitment in this Section 5.2 is calibrated accordingly. Shipments affected by Force Majeure Events and shipments delayed solely by customs holds not attributable to Atlas documentation errors are excluded from measurement.

### 5.3 Measurement & Reporting

Atlas shall deliver a monthly scorecard by the seventh (7th) calendar day of the following month showing, per lane: shipments tendered, shipments delivered on time, the On-Time Delivery Rate, every Late Delivery with delay duration in hours, and accrued penalties under Section 6. GlobalTrade may dispute entries within fifteen (15) calendar days. Atlas shall additionally attend a quarterly business review covering capacity, performance trends, and corrective actions."""

ATLAS_PENALTY = """### 6.1 Flat-Fee Penalty Structure

The Parties agree on a flat-fee penalty structure reflecting the lower unit economics of Standard Class freight. For each Late Delivery where the delay exceeds **forty-eight (48) hours** beyond the Scheduled Delivery Date, Atlas shall pay GlobalTrade a fixed penalty of **$250 per shipment**. Where the delay exceeds **five (5) Business Days**, the penalty for that shipment increases to **$400**. Deliveries late by forty-eight (48) hours or less incur no financial penalty but still count against the On-Time Delivery Rate in Section 5.2.

### 6.2 Quarterly Penalty Cap

Atlas's aggregate penalty liability under this Section 6 is capped at **$75,000 per calendar quarter**. Penalties accrued beyond the cap in a quarter are extinguished, not carried forward.

### 6.3 Worked Example

In a given month Atlas delivers 38 shipments late: 22 by less than 48 hours (no penalty), 13 by between 48 hours and five Business Days (13 x $250 = $3,250), and 3 by more than five Business Days (3 x $400 = $1,200). The penalty payable for the month is **$4,450**, subject to the quarterly cap.

### 6.4 Payment of Penalties

Accrued penalties are credited on Atlas's next monthly invoice. Penalties are GlobalTrade's exclusive financial remedy for Late Delivery as such, without prejudice to Section 10(e) and to claims for loss of or damage to the goods."""

ATLAS_PAYMENT = """Atlas shall invoice GlobalTrade monthly in arrears, in US dollars, itemizing each shipment by bill-of-lading number, lane, weight or container count, and applicable accessorial charges. Undisputed invoices are payable **net forty-five (45) days** from the invoice date. GlobalTrade must raise invoice disputes within fifteen (15) calendar days of receipt; the undisputed portion remains payable when due. Late payments accrue interest at **1.0% per month** on overdue undisputed amounts. Fuel surcharges adjust monthly per the index in Schedule C; all other rates are fixed per Section 4. Atlas may suspend acceptance of new tenders if undisputed amounts remain unpaid seventy-five (75) days past due, after ten (10) Business Days' written warning."""

PCC_SCOPE = """Pacific Crest shall provide regional transportation services for GlobalTrade's **Pacific Asia** market, comprising intra-Asia air and sea freight, trans-Pacific ocean services, and final-mile distribution in Southeast Asia, Eastern Asia, South Asia, and Oceania, primarily for Second Class and Standard Class shipments. Second Class shipments carry a Scheduled Delivery Date of four (4) days from tender and Standard Class shipments six (6) days from tender. Pacific Crest shall operate consolidation hubs in Singapore and Shanghai for GlobalTrade volume, provide DDP and DAP customs handling per shipment instructions, and maintain EDI integration for milestone events. Pacific Crest is GlobalTrade's preferred regional carrier for the Pacific Asia market; lanes may be added or removed by amendment to Schedule A."""

PCC_SLA = """### 5.1 Service Level Framework

Pacific Crest's performance is measured against the commitments in this Section 5 on a **calendar-quarter** basis, in recognition of the longer transit cycles and monsoon-season variability of the region's lanes. Measurement uses Pacific Crest's milestone EDI events validated against terminal operator records.

### 5.2 On-Time Delivery Commitment

Pacific Crest commits to an On-Time Delivery Rate of at least **94.0%** of all shipments tendered in each calendar quarter, measured across all Pacific Asia lanes in aggregate. Shipments affected by Force Majeure Events, port congestion declared by the relevant port authority, or customs inspection holds not attributable to Pacific Crest documentation errors are excluded from measurement, provided Pacific Crest evidences the exclusion in the quarterly scorecard.

### 5.3 Measurement & Reporting

Pacific Crest shall deliver a quarterly performance scorecard within ten (10) Business Days of quarter end, broken down by lane and by origin country, listing every Late Delivery with its delay in calendar days and the service credits accrued under Section 6. Monthly flash reports (summary On-Time Delivery Rate only) are due by the fifth (5th) Business Day of each month. GlobalTrade may dispute scorecard entries within fifteen (15) Business Days."""

PCC_PENALTY = """### 6.1 Service Credit Structure

The Parties adopt a service-credit remedy rather than cash penalties. For each Late Delivery, Pacific Crest shall issue GlobalTrade a service credit equal to **1.5% of that shipment's invoiced freight charges for each Late Calendar Day**, beginning on the first calendar day after the Scheduled Delivery Date. For clarity, the credit base is the freight charges invoiced by Pacific Crest for the shipment, not the commercial value of the goods.

### 6.2 Credit Cap

Service credits for any single shipment are capped at **10% of that shipment's invoiced freight charges**.

### 6.3 Application of Credits

Accrued credits are itemized in the quarterly scorecard and applied against Pacific Crest's next monthly invoice following scorecard finalization. Credits are not redeemable for cash, do not expire while the Agreement remains in force, and any unapplied balance is payable in cash only upon termination or expiry of this Agreement.

### 6.4 Worked Example

A Second Class shipment with invoiced freight charges of $2,000 is delivered four (4) calendar days after its Scheduled Delivery Date: the credit is 4 x 1.5% = 6% of $2,000 = **$120**. A delay of ten (10) calendar days would compute to 15% but is capped at 10%, i.e. **$200**.

### 6.5 Exclusive Remedy

Service credits under this Section 6 are GlobalTrade's sole and exclusive financial remedy for Late Delivery as such, without prejudice to GlobalTrade's termination rights under Section 10 and claims for loss of or damage to cargo."""

PCC_PAYMENT = """Pacific Crest shall invoice GlobalTrade monthly in arrears, in US dollars, with separate invoice lines per shipment showing freight charges, fuel and security surcharges, and customs disbursements with supporting receipts. Undisputed invoices are payable **net sixty (60) days** from the invoice date. GlobalTrade must raise invoice disputes within fifteen (15) Business Days of receipt. No interest accrues on late payments; however, Pacific Crest may suspend acceptance of new tenders if undisputed amounts remain unpaid **seventy-five (75) days** past due, after giving ten (10) Business Days' written warning. Customs duties and taxes advanced by Pacific Crest on GlobalTrade's behalf are re-invoiced at cost plus a 1.0% disbursement fee and are payable net thirty (30) days."""

NORDHAUL_SCOPE = """NordHaul shall provide premium road and short-sea transportation services across GlobalTrade's **Europe** market, comprising scheduled full-truckload and groupage linehaul across Western, Northern, Eastern, and Southern Europe, short-sea connections to the Nordics and the UK, and final-mile delivery coordination, serving all shipping modes with emphasis on First Class and Second Class lanes. Each lane in Schedule A carries the Scheduled Delivery Date corresponding to its shipping mode (Same Day: one day; First Class: two days; Second Class: four days; Standard Class: six days). NordHaul shall operate under a named-account control tower in Rotterdam providing 24/7 exception management, predictive ETA updates, and CO2 emissions reporting per shipment in accordance with the GLEC framework."""

NORDHAUL_SLA = """### 5.1 Service Level Framework

NordHaul's performance is measured against the commitments in this Section 5 on a calendar-month basis, **per lane** as defined in Schedule A, using control-tower telematics timestamps reconciled to proof-of-delivery documents. NordHaul is engaged as GlobalTrade's premium European carrier, and the Service Levels in this Section reflect best-in-class network performance.

### 5.2 On-Time Delivery Commitment

NordHaul commits to an On-Time Delivery Rate of at least **97.5%** on each lane in each calendar month — the highest commitment in GlobalTrade's carrier portfolio. Shipments affected by Force Majeure Events are excluded from measurement. Lanes with fewer than twenty (20) shipments in a month are measured on a rolling three-month basis to avoid small-sample distortion.

### 5.3 Measurement & Reporting

NordHaul shall publish a continuously updated performance dashboard and deliver a formal monthly scorecard by the third (3rd) Business Day of the following month, per lane and in aggregate, including delay root-cause codes, the Monthly Lane Fees per lane, and the service credits computed under Section 6. GlobalTrade may dispute entries within ten (10) Business Days of scorecard delivery."""

NORDHAUL_PENALTY = """### 6.1 Lane-Fee Service Credit Structure

Remedies are computed per lane per calendar month against the Monthly Lane Fees. If the On-Time Delivery Rate on a lane falls below the 97.5% commitment in a month, NordHaul shall issue GlobalTrade a service credit equal to **3% of that lane's Monthly Lane Fees for each commenced 0.5-percentage-point shortfall** below 97.5%.

### 6.2 Credit Cap

Service credits for any lane in any month are capped at **12% of that lane's Monthly Lane Fees**.

### 6.3 Worked Example

A lane achieves a 96.4% On-Time Delivery Rate in a month with Monthly Lane Fees of EUR 40,000. The shortfall is 1.1 percentage points, which spans three commenced 0.5-point increments (0.1-0.5, 0.5-1.0, 1.0-1.5), so the credit is 3 x 3% = 9% of EUR 40,000 = **EUR 3,600**. A rate of 93.0% would compute to ten increments (30%) but is capped at 12%, i.e. **EUR 4,800**.

### 6.4 Application of Credits

Credits are itemized on the monthly scorecard and deducted from the next monthly invoice. Credits under this Section 6 are GlobalTrade's exclusive financial remedy for Service Level shortfall as such, without prejudice to Section 10(e) and to cargo loss or damage claims under the CMR Convention."""

NORDHAUL_PAYMENT = """NordHaul shall invoice GlobalTrade monthly in arrears, in **euros (EUR)**, one invoice per lane plus a consolidated summary, payable by SEPA credit transfer. Undisputed invoices are payable **net thirty (30) days** from the invoice date. GlobalTrade must raise invoice disputes within ten (10) Business Days of receipt; the undisputed portion remains payable when due. Late payments accrue statutory interest in accordance with Directive 2011/7/EU on combating late payment in commercial transactions (European Central Bank reference rate plus eight (8) percentage points). Diesel and road-toll surcharges adjust quarterly per the index mechanism in Schedule C. NordHaul shall not suspend services for non-payment unless undisputed amounts remain unpaid sixty (60) days past due and NordHaul has given fifteen (15) Business Days' written warning."""

MERIDIAN_SCOPE = """The Supplier shall manufacture, sell, and deliver to the Buyer the products listed in Schedule A (the "Products"), comprising corrugated and protective packaging materials, maintenance/repair/operations (MRO) consumables, and finished-goods components, for delivery to the Buyer's distribution centers in the USCA and LATAM markets as designated on each purchase order ("PO"). The Buyer shall provide non-binding twelve-month rolling forecasts updated monthly; only released POs constitute binding commitments. The Supplier shall hold safety stock equal to four (4) weeks of forecast demand for the A-class SKUs identified in Schedule A. Each PO states the quantities, Products, delivery location, and the Scheduled Delivery Date, which shall be no earlier than the lead times in Schedule A. Title and risk transfer on delivery DDP (Incoterms 2020) at the named destination."""

MERIDIAN_SLA = """### 5.1 Service Level Framework

The Supplier's delivery and quality performance is measured monthly across all POs delivered in the calendar month, using the Buyer's goods-receipt records as the system of record, reconciled with the Supplier's advance shipping notices.

### 5.2 On-Time Delivery Commitment

The Supplier commits to an On-Time Delivery Rate of at least **95.0%** of PO lines in each calendar month, and an on-time-in-full (OTIF) rate — PO lines delivered both on time and complete in quantity — of at least **93.0%**. A PO line is on time if received at the named destination on or before its Scheduled Delivery Date; early delivery more than five (5) Business Days before the Scheduled Delivery Date requires the Buyer's consent and may be refused at the dock.

### 5.3 Quality Commitment

Delivered Products shall conform to the specifications in Schedule B with a defect rate not exceeding **1.5%** per delivery, measured by the Buyer's incoming inspection. If the defect rate of any Product exceeds **3.0%** across two consecutive months (an "Epidemic Failure"), the Supplier shall, at the Buyer's election, replace the affected population at its own cost and reimburse the Buyer's reasonable screening, rework, and recall costs.

### 5.4 Measurement & Reporting

The Supplier shall deliver a monthly performance report by the fifth (5th) Business Day of the following month covering On-Time Delivery Rate, OTIF, defect rates by Product, open corrective actions, and penalties accrued under Section 6, and shall attend a quarterly business review."""

MERIDIAN_PENALTY = """### 6.1 Late Delivery Penalty

For each PO line constituting a Late Delivery, the Supplier shall pay the Buyer a penalty equal to **0.75% of the PO Value of that line for each commenced Late Week** (any period of seven calendar days, or part thereof, beginning the day after the Scheduled Delivery Date). A delay of one (1) day therefore incurs the same penalty as a delay of seven (7) days.

### 6.2 Penalty Cap

The aggregate penalty for any single PO line is capped at **8% of that line's PO Value**.

### 6.3 Cancellation and Cover

If a PO line remains undelivered **twenty-one (21) calendar days** after its Scheduled Delivery Date, the Buyer may cancel the line without liability and procure substitute goods from a third party, in which case the Supplier shall reimburse the Buyer **110% of the difference** between the cover price and the PO price, in addition to penalties accrued up to cancellation.

### 6.4 Worked Example

A PO line with a PO Value of $20,000 is delivered ten (10) calendar days late: the delay spans two commenced Late Weeks, so the penalty is 2 x 0.75% = 1.5% of $20,000 = **$300**. A delay of fifteen (15) weeks would compute to 11.25% but is capped at 8%, i.e. **$1,600**.

### 6.5 Set-Off

The Buyer may set off accrued penalties against any amounts payable to the Supplier. Penalties for Late Delivery do not limit the Buyer's rights under Section 6.3, Section 5.3, or Section 10."""

MERIDIAN_PAYMENT = """The Supplier shall invoice the Buyer upon delivery of each PO line, in US dollars, referencing the PO number, line number, and goods-receipt confirmation. Undisputed invoices are payable on **2/10 net 45** terms: the Buyer may deduct a **2% early-payment discount** if payment is made within ten (10) calendar days of the invoice date; otherwise the full undisputed amount is due within forty-five (45) days. The Buyer must raise invoice disputes within fifteen (15) calendar days of receipt. Late payments accrue interest at 1.0% per month on overdue undisputed amounts. Prices are fixed for the Initial Term per Schedule A; raw-material indexation applies thereafter per Section 4. The Supplier shall maintain electronic invoicing through the Buyer's procurement portal."""

HELIOS_SCOPE = """The Supplier shall manufacture, sell, and deliver to the Buyer the electronic components listed in Schedule A (the "Products"), comprising sensor modules, printed circuit board assemblies (PCBAs), connectors, and power-management components manufactured at the Supplier's Cambridge and Penang facilities, for delivery to the Buyer's distribution centers and contract manufacturers worldwide as designated on each purchase order ("PO"). The Buyer shall provide non-binding six-month rolling forecasts updated monthly; only released POs are binding. Lead times per Product are fixed in Schedule A and the Scheduled Delivery Date on each PO line shall respect them. Delivery is DAP (Incoterms 2020) at the named destination; title passes on delivery and risk passes per the Incoterm. The Supplier shall maintain component traceability to lot level and provide certificates of conformance with each shipment."""

HELIOS_SLA = """### 5.1 Service Level Framework

The Supplier's performance is measured monthly across all PO lines due in the calendar month, using the Buyer's goods-receipt records reconciled with the Supplier's advance shipping notices and lot documentation.

### 5.2 On-Time Delivery Commitment

The Supplier commits to an On-Time Delivery Rate of at least **97.0%** of PO lines in each calendar month, reflecting the line-down criticality of the Products to the Buyer's downstream manufacturing. Each PO line benefits from a **grace period of three (3) Business Days** after its Scheduled Delivery Date: deliveries within the grace period count as on time for penalty purposes under Section 6 but are reported separately on the monthly scorecard as "grace deliveries".

### 5.3 Quality Commitment

Delivered Products shall meet an acceptance quality limit (AQL) of **0.65**, general inspection level II, per ISO 2859-1, at the Buyer's incoming inspection. Rejected lots shall be collected by the Supplier within ten (10) Business Days and replaced within the original lead time at the Supplier's cost; return merchandise authorizations (RMAs) must be issued within ten (10) Business Days of notification.

### 5.4 Measurement & Reporting

The Supplier shall deliver a monthly scorecard by the fourth (4th) Business Day of the following month covering On-Time Delivery Rate, grace deliveries, AQL results, open RMAs, and liquidated damages accrued under Section 6, and shall attend a quarterly business review including a capacity and obsolescence outlook."""

HELIOS_PENALTY = """### 6.1 Liquidated Damages

The Parties agree that the Buyer's losses from late component deliveries (production-line stoppage, expedited re-planning, air-freight substitution) are real but difficult to quantify, and fix liquidated damages as a genuine pre-estimate of loss, not a penalty. For each PO line delivered after its grace period under Section 5.2, the Supplier shall pay the Buyer liquidated damages of **1% of the Line-Item Value for each Late Business Day counted from the end of the three (3) Business Day grace period**.

### 6.2 Damages Cap

Liquidated damages for any single PO line are capped at **20% of that line's Line-Item Value** — the highest cap in the Buyer's supplier portfolio, reflecting the criticality of the Products.

### 6.3 Worked Example

A PO line with a Line-Item Value of GBP 8,000 is delivered seven (7) Business Days after its Scheduled Delivery Date. The first three (3) Business Days fall within the grace period; four (4) chargeable Late Business Days remain, so liquidated damages are 4 x 1% = 4% of GBP 8,000 = **GBP 320**. A delay of thirty (30) Business Days would compute to 27% but is capped at 20%, i.e. **GBP 1,600**.

### 6.4 Relationship to Other Remedies

Liquidated damages under this Section 6 are the Buyer's exclusive financial remedy for late delivery of conforming Products, but do not limit the Buyer's remedies for non-conforming Products under Section 5.3, its cancellation rights for lines more than twenty (20) Business Days late, or its termination rights under Section 10."""

HELIOS_PAYMENT = """The Supplier shall invoice the Buyer upon dispatch of each PO line, in **pounds sterling (GBP)**, referencing the PO number, line number, lot numbers, and certificate of conformance. Undisputed invoices are payable **net sixty (60) days** from the invoice date. The Buyer must raise invoice disputes within fifteen (15) Business Days of receipt; the undisputed portion remains payable when due. Late payments accrue interest at the Bank of England base rate plus four (4) percentage points per annum on overdue undisputed amounts. Prices are fixed per Schedule A for each twelve-month period; exchange-rate and material-cost adjustments require the review mechanism in Section 4. Tooling owned by the Buyer and held by the Supplier is itemized in Schedule D and invoiced separately."""
