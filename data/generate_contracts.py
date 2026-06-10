"""Generates the six carrier / supplier contracts used by the contracts_analyst agent.

WHY THIS EXISTS
---------------
The Week-4 "PageIndex" RAG lessons need a corpus where answers are *verifiable*: every
contract carries distinct, quotable numbers (penalty formulas, SLA percentages, cure
periods, liability caps), so a trainee can immediately check whether the retriever
found the right section in the right document. Six documents with parallel section
structure but different terms is the ideal stress test — naive keyword matching will
confuse "2% per late business day" (SwiftShip) with "1.5% per late calendar day"
(Pacific Crest), while tree-based reasoning retrieval should not.

DESIGN
------
Content is **static text** — the shared skeleton and per-contract terms below, the
six contracts' verbatim section bodies in the sibling ``contract_texts.py`` (fully
deterministic — re-running produces byte-identical files). A shared legal skeleton (Parties, Definitions, Term,
Liability, Force Majeure, Termination, Governing Law, Miscellaneous) is parameterized
per contract, while the four commercially interesting sections (Scope, Service Levels,
Late Delivery Penalties, Invoicing & Payment) are fully custom per contract. The
parallel structure is intentional teaching material: retrieval must discriminate by
*party and number*, not by section title.

THE NUMBERS AT A GLANCE (each one is a quiz answer somewhere in Week 4)
-----------------------------------------------------------------------
| Contract        | On-time SLA | Late penalty                                | Cap   |
|-----------------|-------------|---------------------------------------------|-------|
| SwiftShip       | 96.0%/mo    | 2% of shipment value per late business day  | 15%   |
| Atlas Freight   | 92.5%/mo    | flat $250 (>48h), $400 (>5 business days)   | $75k/q|
| Pacific Crest   | 94.0%/qtr   | 1.5% of freight charges per late cal. day   | 10%   |
| NordHaul        | 97.5%/mo    | 3% of monthly lane fees per 0.5pp shortfall | 12%   |
| Meridian Supply | 95.0%/mo    | 0.75% of PO value per commenced late week   | 8%    |
| Helios          | 97.0%/mo    | 1% of line-item value per late business day | 20%   |
|                 |             |   (after a 3-business-day grace period)     |       |

Run: ``python data/generate_contracts.py`` (from the project root).
"""

from __future__ import annotations

from pathlib import Path

# The 24 verbatim section bodies live in the sibling module ``contract_texts``
# (this generator stays under the ~350-line convention, ARCHITECTURE.md §7).
# Two import paths because this file legitimately runs both ways:
#   * as a package module — ``from data import generate_contracts`` (used by
#     tests/conftest.py, main.py, lab06): project root is on sys.path.
#   * as a script — ``python data/generate_contracts.py``: sys.path[0] is the
#     data/ directory itself, so the sibling imports as a top-level module.
try:
    from data import contract_texts as texts
except ImportError:  # direct script execution
    import contract_texts as texts  # type: ignore[no-redef]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "contracts"

# ----------------------------------------------------------------------------------
# Shared legal skeleton. {placeholders} are filled per contract from SPECS below.
# ----------------------------------------------------------------------------------
TEMPLATE = """# {doc_title}

**Agreement No. {number}** | Effective Date: {effective} | Between **GlobalTrade Logistics Inc.** ("GlobalTrade") and **{party}** ("{party_short}")

## 1. Parties

This {agreement_type} (the "Agreement") is entered into as of {effective} (the "Effective Date") by and between **GlobalTrade Logistics Inc.**, a Delaware corporation with its principal place of business at 400 Harborview Plaza, Suite 1200, Wilmington, Delaware, USA ("GlobalTrade", the "{client_role}"), and **{party}**, {party_legal}, with its principal place of business at {party_address} ("{party_short}", the "{party_role}"). GlobalTrade and {party_short} are each a "Party" and together the "Parties". This Agreement supersedes all prior agreements, proposals, term sheets, and letters of intent between the Parties relating to its subject matter.

## 2. Definitions

For the purposes of this Agreement, the following capitalized terms have the meanings set out below. Other capitalized terms are defined where they first appear.

1. **"Business Day"** means any day other than a Saturday, Sunday, or public holiday in the jurisdiction of the delivery destination.
2. **"Scheduled Delivery Date"** means {sched_def}.
3. **"Late Delivery"** means {late_def}.
4. **"On-Time Delivery Rate"** means, for any {measurement_period}, the number of {unit_plural} delivered on or before their Scheduled Delivery Date divided by the total number of {unit_plural} {unit_verb} in that period, expressed as a percentage and rounded to one decimal place.
5. **"{value_term}"** means {value_def}.
6. **"Service Levels"** means the performance commitments set out in Section 5 of this Agreement.
7. **"Confidential Information"** means all non-public business, technical, pricing, volume, and operational information disclosed by one Party to the other, whether marked confidential or not, that a reasonable person would understand to be confidential.
{definitions_extra}

## 3. {scope_title}

{scope_body}

## 4. Term

This Agreement commences on the Effective Date and continues for an initial term of {term_years} (the "Initial Term"). Upon expiry of the Initial Term, this Agreement automatically renews for successive {renewal} renewal periods (each a "Renewal Term") unless either Party gives written notice of non-renewal at least {nonrenewal_notice} days before the end of the then-current term. Rates and Service Levels remain fixed during the Initial Term; thereafter, either Party may request one rate review per Renewal Term, provided that any adjustment exceeding {rate_cap}% per annum requires mutual written agreement supported by a documented cost index. Sections 2, 6 (with respect to accrued penalties and credits), 8, 11, and the confidentiality obligations survive expiry or termination of this Agreement for a period of three (3) years.

## 5. Service Levels

{sla_body}

## 6. {penalty_title}

{penalty_body}

## 7. Invoicing & Payment

{payment_body}

## 8. Liability & Indemnification

{party_short}'s aggregate liability under this Agreement, whether arising in contract, tort (including negligence), breach of statutory duty, or otherwise, shall not exceed {liability_cap}. The foregoing cap does not apply to: (a) liability arising from gross negligence, willful misconduct, or fraud; (b) {party_short}'s indemnification obligations in respect of third-party claims for bodily injury, death, or damage to tangible property; (c) breaches of confidentiality; or (d) {cargo_carveout}. {remedy_sentence} Neither Party is liable to the other for indirect, consequential, special, or punitive damages, or for loss of profit, revenue, or goodwill, except to the extent such damages fall within the carve-outs above. {party_short} shall maintain, at its own cost, {insurance}, and shall provide certificates of insurance to GlobalTrade upon request and at each renewal.

## 9. Force Majeure

Neither Party is liable for any failure or delay in performance caused by events beyond its reasonable control, including natural disasters, severe weather, epidemics, war, terrorism, civil unrest, acts of government, embargoes, port or airport closures, customs stoppages not attributable to documentation errors by the affected Party, national or industry-wide strikes (expressly excluding strikes limited to {party_short}'s own workforce), and widespread failures of public infrastructure, utilities, or telecommunications (each a "Force Majeure Event"). The affected Party must notify the other Party in writing within {fm_notice} of becoming aware of a Force Majeure Event, describing the event, the performance affected, and the estimated duration, and must use commercially reasonable efforts to mitigate its impact and resume performance. Service Level measurement under Section 5 is suspended for {unit_plural} demonstrably affected by the Force Majeure Event, and no penalties or credits under Section 6 accrue with respect to such {unit_plural}. If a Force Majeure Event continues for more than {fm_term_days} consecutive days, either Party may terminate this Agreement upon written notice without penalty, and the {client_role} shall pay only for services performed and conforming goods delivered up to the effective date of termination.

## 10. Termination

Either Party may terminate this Agreement: (a) for material breach by the other Party that remains uncured {cure} after written notice describing the breach in reasonable detail; (b) immediately upon written notice if the other Party becomes insolvent, files or has filed against it a petition in bankruptcy that is not dismissed within sixty (60) days, or makes a general assignment for the benefit of creditors; or (c) as provided in Section 9. GlobalTrade may additionally terminate this Agreement: (d) for convenience on {convenience_notice} days' written notice; and (e) for chronic Service Level failure, defined as {chronic_def}, on thirty (30) days' written notice without any further cure period. Upon any termination or expiry, {party_short} shall complete all {unit_plural} in transit or accepted prior to the effective date of termination in accordance with this Agreement, shall return or destroy all Confidential Information of GlobalTrade, and shall cooperate in good faith in an orderly transition to a successor provider for up to sixty (60) days at the rates then in effect. Termination does not relieve either Party of obligations accrued before the effective date of termination, including penalties and service credits accrued under Section 6.

## 11. Governing Law & Dispute Resolution

This Agreement is governed by and construed in accordance with the laws of {law}, without regard to its conflict-of-laws principles. {dispute_clause} Before initiating any proceeding, the Parties shall escalate the dispute first to their respective account managers and, failing resolution within fifteen (15) Business Days, to a senior executive of each Party; only if the dispute remains unresolved thirty (30) days after escalation to executives may a proceeding be initiated. Nothing in this Section prevents either Party from seeking interim injunctive relief to protect its Confidential Information or intellectual property.

## 12. Miscellaneous

This Agreement, together with its Schedules and any executed amendments, constitutes the entire agreement between the Parties with respect to its subject matter. Amendments must be in writing and signed by authorized representatives of both Parties. Neither Party may assign this Agreement without the other Party's prior written consent, except to an affiliate or in connection with a merger, acquisition, or sale of substantially all of its assets, with written notice to the other Party. {party_short} shall not subcontract {subcontract_limit} without GlobalTrade's prior written approval, and remains fully liable for the acts and omissions of any approved subcontractor as if they were its own. All notices must be in writing and delivered to the addresses stated in Section 1 by courier or by email with delivery confirmation. If any provision of this Agreement is held invalid or unenforceable, the remaining provisions remain in full force and effect. No waiver of any provision is effective unless made in writing; the failure of either Party to enforce any provision is not a waiver of its right to do so later.
"""

# ----------------------------------------------------------------------------------
# Per-contract specifications. The four custom section bodies per contract
# (scope / SLA / penalties / payment) are imported from ``contract_texts`` and
# referenced as ``texts.<PARTY>_<SECTION>`` to keep the spec dicts readable.
# ----------------------------------------------------------------------------------

SPECS: list[dict[str, str]] = [
    {
        "filename": "swiftship_express_msa.md",
        "doc_title": "Master Service Agreement — GlobalTrade Logistics Inc. & SwiftShip Express",
        "number": "SS-MSA-2024-117", "effective": "January 15, 2024",
        "party": "SwiftShip Express, Inc.", "party_short": "SwiftShip",
        "party_legal": "a Delaware corporation", "party_address": "2200 Aerodrome Way, Memphis, Tennessee, USA",
        "agreement_type": "Master Service Agreement", "client_role": "Shipper", "party_role": "Carrier",
        "sched_def": "for each shipment, the delivery date promised at tender: one (1) day after tender for Same Day shipments and two (2) days after tender for First Class shipments",
        "late_def": "any shipment delivered after its Scheduled Delivery Date; because Same Day and First Class are time-definite products, **no grace period applies**",
        "measurement_period": "calendar month", "unit_plural": "shipments", "unit_verb": "tendered",
        "value_term": "Shipment Value", "value_def": "the invoiced sales value of the goods contained in a shipment, exclusive of freight charges, duties, and taxes",
        "definitions_extra": "8. **\"Late Business Day\"** means each Business Day, or part thereof, elapsed after the Scheduled Delivery Date until actual delivery.\n9. **\"Tender\"** means GlobalTrade's electronic handover of a shipment booking to SwiftShip via API or portal.",
        "scope_title": "Scope of Services", "scope_body": texts.SWIFTSHIP_SCOPE,
        "term_years": "three (3) years", "renewal": "one (1)-year", "nonrenewal_notice": "ninety (90)", "rate_cap": "4",
        "sla_body": texts.SWIFTSHIP_SLA, "penalty_title": "Late Delivery Penalties", "penalty_body": texts.SWIFTSHIP_PENALTY,
        "payment_body": texts.SWIFTSHIP_PAYMENT,
        "liability_cap": "the total fees paid or payable by GlobalTrade under this Agreement in the twelve (12) months preceding the event giving rise to liability",
        "cargo_carveout": "loss of or damage to cargo, which is governed by the Montreal Convention for international air carriage or applicable national surface-transport law",
        "remedy_sentence": "Penalties under Section 6 are GlobalTrade's exclusive financial remedy for Late Delivery as such, but do not limit remedies for chronic failure under Section 10(e) or for loss of or damage to the goods themselves.",
        "insurance": "commercial general liability insurance of at least $5,000,000 per occurrence, cargo liability insurance of at least $1,000,000 per conveyance, and statutory employer liability cover",
        "fm_notice": "five (5) Business Days", "fm_term_days": "thirty (30)",
        "cure": "thirty (30) days", "convenience_notice": "ninety (90)",
        "chronic_def": "an On-Time Delivery Rate below 96.0% for two (2) consecutive calendar quarters",
        "law": "the State of New York, USA",
        "dispute_clause": "The state and federal courts located in New York County, New York have exclusive jurisdiction over any dispute arising out of or in connection with this Agreement, and each Party irrevocably submits to that jurisdiction.",
        "subcontract_limit": "linehaul services on any lane",
    },
    {
        "filename": "atlas_freight_msa.md",
        "doc_title": "Master Service Agreement — GlobalTrade Logistics Inc. & Atlas Freight Co.",
        "number": "AF-MSA-2024-031", "effective": "March 1, 2024",
        "party": "Atlas Freight Co.", "party_short": "Atlas",
        "party_legal": "an Illinois corporation", "party_address": "880 Lakeshore Industrial Park, Chicago, Illinois, USA",
        "agreement_type": "Master Service Agreement", "client_role": "Shipper", "party_role": "Carrier",
        "sched_def": "for each shipment, the delivery date promised at tender: six (6) days after tender for Standard Class shipments and four (4) days after tender for Second Class shipments where tendered under Schedule B",
        "late_def": "any shipment delivered after its Scheduled Delivery Date; financial penalties under Section 6 attach only where the delay exceeds forty-eight (48) hours",
        "measurement_period": "calendar month", "unit_plural": "shipments", "unit_verb": "tendered",
        "value_term": "Shipment Value", "value_def": "the invoiced sales value of the goods contained in a shipment, exclusive of freight charges, duties, and taxes; used for cargo claims, not for penalty calculation, which under this Agreement is flat-fee based",
        "definitions_extra": "8. **\"Late Shipment\"** means a shipment whose delay beyond the Scheduled Delivery Date exceeds forty-eight (48) hours.\n9. **\"Consolidation Center\"** means a GlobalTrade facility listed in Schedule D from which Atlas operates scheduled departures.",
        "scope_title": "Scope of Services", "scope_body": texts.ATLAS_SCOPE,
        "term_years": "five (5) years", "renewal": "two (2)-year", "nonrenewal_notice": "one hundred twenty (120)", "rate_cap": "3",
        "sla_body": texts.ATLAS_SLA, "penalty_title": "Late Delivery Penalties", "penalty_body": texts.ATLAS_PENALTY,
        "payment_body": texts.ATLAS_PAYMENT,
        "liability_cap": "two million US dollars ($2,000,000) in the aggregate over the term of this Agreement",
        "cargo_carveout": "loss of or damage to cargo, which is governed by the Carmack Amendment for US ground carriage, COGSA for ocean legs, or other mandatorily applicable transport law",
        "remedy_sentence": "Flat-fee penalties under Section 6 are GlobalTrade's exclusive financial remedy for Late Delivery as such, but do not limit remedies for chronic failure under Section 10(e) or for loss of or damage to the goods themselves.",
        "insurance": "commercial general liability insurance of at least $3,000,000 per occurrence, motor carrier cargo liability of at least $500,000 per conveyance, and statutory workers' compensation cover",
        "fm_notice": "seven (7) calendar days", "fm_term_days": "forty-five (45)",
        "cure": "forty-five (45) days", "convenience_notice": "one hundred twenty (120)",
        "chronic_def": "an On-Time Delivery Rate below 92.5% in any three (3) calendar months within a rolling six (6)-month window",
        "law": "the State of Illinois, USA",
        "dispute_clause": "The state and federal courts located in Cook County, Illinois have exclusive jurisdiction over any dispute arising out of or in connection with this Agreement, and each Party irrevocably submits to that jurisdiction.",
        "subcontract_limit": "more than 30% of monthly linehaul volume to interline partners",
    },
    {
        "filename": "pacific_crest_carriers_msa.md",
        "doc_title": "Master Service Agreement — GlobalTrade Logistics Inc. & Pacific Crest Carriers",
        "number": "PCC-MSA-2024-208", "effective": "February 12, 2024",
        "party": "Pacific Crest Carriers Pte. Ltd.", "party_short": "Pacific Crest",
        "party_legal": "a private limited company incorporated in Singapore", "party_address": "71 Keppel Harbour Road, Singapore 098419",
        "agreement_type": "Master Service Agreement", "client_role": "Shipper", "party_role": "Carrier",
        "sched_def": "for each shipment, the delivery date promised at tender per its shipping mode: four (4) days after tender for Second Class shipments and six (6) days after tender for Standard Class shipments",
        "late_def": "any shipment delivered after its Scheduled Delivery Date, measured in Late Calendar Days",
        "measurement_period": "calendar quarter", "unit_plural": "shipments", "unit_verb": "tendered",
        "value_term": "Freight Charges", "value_def": "the transportation charges invoiced by Pacific Crest for a shipment, excluding customs duties, taxes, and disbursements; service credits under Section 6 are computed on Freight Charges, not on the commercial value of the goods",
        "definitions_extra": "8. **\"Late Calendar Day\"** means each calendar day, or part thereof, elapsed after the Scheduled Delivery Date until actual delivery.\n9. **\"Service Credit\"** means a credit issued by Pacific Crest against future invoices pursuant to Section 6.",
        "scope_title": "Scope of Services", "scope_body": texts.PCC_SCOPE,
        "term_years": "two (2) years", "renewal": "one (1)-year", "nonrenewal_notice": "sixty (60)", "rate_cap": "5",
        "sla_body": texts.PCC_SLA, "penalty_title": "Late Delivery Penalties (Service Credits)", "penalty_body": texts.PCC_PENALTY,
        "payment_body": texts.PCC_PAYMENT,
        "liability_cap": "an amount equal to six (6) times Pacific Crest's average monthly charges to GlobalTrade over the six (6) months preceding the event giving rise to liability",
        "cargo_carveout": "loss of or damage to cargo, which is governed by the Hague-Visby Rules for sea carriage, the Montreal Convention for air carriage, or other mandatorily applicable transport conventions",
        "remedy_sentence": "Service credits under Section 6 are GlobalTrade's exclusive financial remedy for Late Delivery as such, but do not limit remedies for chronic failure under Section 10(e) or claims for loss of or damage to cargo.",
        "insurance": "commercial general liability insurance of at least $3,000,000 per occurrence, marine cargo legal liability cover of at least $1,000,000 per vessel or aircraft, and statutory employer liability cover",
        "fm_notice": "ten (10) calendar days", "fm_term_days": "sixty (60)",
        "cure": "twenty-one (21) days", "convenience_notice": "sixty (60)",
        "chronic_def": "a quarterly On-Time Delivery Rate below 94.0% in any two (2) quarters within four (4) consecutive calendar quarters",
        "law": "Singapore",
        "dispute_clause": "Any dispute arising out of or in connection with this Agreement shall be referred to and finally resolved by arbitration administered by the Singapore International Arbitration Centre (SIAC) in accordance with its rules, seated in Singapore, before a single arbitrator, in the English language.",
        "subcontract_limit": "ocean or air linehaul to carriers outside its approved partner list in Schedule E",
    },
    {
        "filename": "nordhaul_logistics_msa.md",
        "doc_title": "Master Service Agreement — GlobalTrade Logistics Inc. & NordHaul Logistics",
        "number": "NH-MSA-2024-052", "effective": "April 1, 2024",
        "party": "NordHaul Logistics B.V.", "party_short": "NordHaul",
        "party_legal": "a besloten vennootschap organized under the laws of the Netherlands", "party_address": "Havenkade 18, 3072 AB Rotterdam, the Netherlands",
        "agreement_type": "Master Service Agreement", "client_role": "Shipper", "party_role": "Carrier",
        "sched_def": "for each shipment, the delivery date promised at tender per its shipping mode (Same Day: one (1) day; First Class: two (2) days; Second Class: four (4) days; Standard Class: six (6) days after tender)",
        "late_def": "any shipment delivered after its Scheduled Delivery Date; remedies under Section 6 are computed at lane level on monthly On-Time Delivery Rate shortfall, not per individual shipment",
        "measurement_period": "calendar month", "unit_plural": "shipments", "unit_verb": "tendered",
        "value_term": "Monthly Lane Fees", "value_def": "the total transportation fees invoiced by NordHaul for a given Schedule A lane in a given calendar month, excluding surcharges, duties, and taxes; service credits under Section 6 are computed on Monthly Lane Fees",
        "definitions_extra": "8. **\"Lane\"** means an origin-destination corridor listed in Schedule A, measured separately for Service Level purposes.\n9. **\"Control Tower\"** means NordHaul's 24/7 named-account operations desk in Rotterdam responsible for exception management and reporting.",
        "scope_title": "Scope of Services", "scope_body": texts.NORDHAUL_SCOPE,
        "term_years": "four (4) years", "renewal": "one (1)-year", "nonrenewal_notice": "one hundred eighty (180)", "rate_cap": "3",
        "sla_body": texts.NORDHAUL_SLA, "penalty_title": "Service Level Credits for Late Delivery", "penalty_body": texts.NORDHAUL_PENALTY,
        "payment_body": texts.NORDHAUL_PAYMENT,
        "liability_cap": "one million five hundred thousand euros (EUR 1,500,000) per contract year",
        "cargo_carveout": "loss of or damage to cargo, which is governed by the CMR Convention for international road carriage and by mandatorily applicable law for short-sea legs",
        "remedy_sentence": "Service credits under Section 6 are GlobalTrade's exclusive financial remedy for Service Level shortfall as such, but do not limit remedies for chronic failure under Section 10(e) or cargo claims under the CMR Convention.",
        "insurance": "commercial general liability insurance of at least EUR 5,000,000 per occurrence, CMR liability insurance at the limits of the Convention, and statutory employer liability cover",
        "fm_notice": "three (3) Business Days", "fm_term_days": "thirty (30)",
        "cure": "fifteen (15) days", "convenience_notice": "one hundred eighty (180)",
        "chronic_def": "an On-Time Delivery Rate below 96.0% on any lane in any single calendar month, or below 97.5% on any lane for three (3) consecutive calendar months",
        "law": "the Netherlands",
        "dispute_clause": "The competent courts of Amsterdam, the Netherlands have exclusive jurisdiction over any dispute arising out of or in connection with this Agreement, and each Party irrevocably submits to that jurisdiction.",
        "subcontract_limit": "any lane representing more than 10% of monthly volume",
    },
    {
        "filename": "meridian_supply_procurement.md",
        "doc_title": "Procurement Agreement — GlobalTrade Logistics Inc. & Meridian Supply Partners",
        "number": "MSP-PA-2024-009", "effective": "January 8, 2024",
        "party": "Meridian Supply Partners LP", "party_short": "Meridian",
        "party_legal": "a Texas limited partnership", "party_address": "1500 Gulfport Commerce Drive, Houston, Texas, USA",
        "agreement_type": "Procurement Agreement", "client_role": "Buyer", "party_role": "Supplier",
        "sched_def": "for each purchase order line, the delivery date stated on the released PO, which shall respect the lead times fixed in Schedule A",
        "late_def": "any PO line received at the named destination after its Scheduled Delivery Date, measured in commenced Late Weeks for penalty purposes under Section 6",
        "measurement_period": "calendar month", "unit_plural": "PO lines", "unit_verb": "due for delivery",
        "value_term": "PO Value", "value_def": "the extended price of a purchase order line (unit price multiplied by ordered quantity) as stated on the released PO, exclusive of taxes and freight",
        "definitions_extra": "8. **\"Late Week\"** means any period of seven (7) calendar days, or part thereof, beginning the day after the Scheduled Delivery Date and ending on actual receipt.\n9. **\"OTIF\"** means on-time-in-full: a PO line delivered on or before its Scheduled Delivery Date and complete in ordered quantity.\n10. **\"Epidemic Failure\"** has the meaning given in Section 5.3.",
        "scope_title": "Scope of Supply", "scope_body": texts.MERIDIAN_SCOPE,
        "term_years": "three (3) years", "renewal": "one (1)-year", "nonrenewal_notice": "ninety (90)", "rate_cap": "4",
        "sla_body": texts.MERIDIAN_SLA, "penalty_title": "Late Delivery Penalties", "penalty_body": texts.MERIDIAN_PENALTY,
        "payment_body": texts.MERIDIAN_PAYMENT,
        "liability_cap": "five million US dollars ($5,000,000) in the aggregate over the term of this Agreement",
        "cargo_carveout": "the Supplier's obligations for Epidemic Failure under Section 5.3 and product-liability claims arising from defective Products",
        "remedy_sentence": "Penalties under Section 6 are the Buyer's exclusive financial remedy for late delivery of conforming Products as such, but do not limit the Buyer's cover rights under Section 6.3, quality remedies under Section 5.3, or termination rights under Section 10.",
        "insurance": "commercial general liability insurance of at least $5,000,000 per occurrence including products liability, and statutory workers' compensation cover",
        "fm_notice": "seven (7) calendar days", "fm_term_days": "sixty (60)",
        "cure": "thirty (30) days", "convenience_notice": "ninety (90)",
        "chronic_def": "an On-Time Delivery Rate below 95.0% or an OTIF rate below 93.0% in any three (3) calendar months within a rolling six (6)-month window",
        "law": "the State of Delaware, USA",
        "dispute_clause": "The state and federal courts located in Wilmington, Delaware have exclusive jurisdiction over any dispute arising out of or in connection with this Agreement, and each Party irrevocably submits to that jurisdiction.",
        "subcontract_limit": "the manufacture of any Product to a facility not listed in Schedule A",
    },
    {
        "filename": "helios_components_procurement.md",
        "doc_title": "Procurement Agreement — GlobalTrade Logistics Inc. & Helios Components Ltd.",
        "number": "HC-PA-2024-074", "effective": "May 20, 2024",
        "party": "Helios Components Ltd.", "party_short": "Helios",
        "party_legal": "a company registered in England and Wales (No. 08841276)", "party_address": "42 Causeway Science Park, Cambridge CB4 0WS, United Kingdom",
        "agreement_type": "Procurement Agreement", "client_role": "Buyer", "party_role": "Supplier",
        "sched_def": "for each purchase order line, the delivery date stated on the released PO, which shall respect the lead times fixed in Schedule A",
        "late_def": "any PO line received at the named destination after its Scheduled Delivery Date; liquidated damages under Section 6 accrue only after the three (3) Business Day grace period in Section 5.2",
        "measurement_period": "calendar month", "unit_plural": "PO lines", "unit_verb": "due for delivery",
        "value_term": "Line-Item Value", "value_def": "the extended price of a purchase order line (unit price multiplied by ordered quantity) as stated on the released PO, exclusive of taxes and freight; liquidated damages under Section 6 are computed on Line-Item Value",
        "definitions_extra": "8. **\"Grace Period\"** means the three (3) Business Days immediately following a PO line's Scheduled Delivery Date, during which no liquidated damages accrue.\n9. **\"Late Business Day\"** means each Business Day, or part thereof, elapsed after the end of the Grace Period until actual receipt.\n10. **\"AQL\"** means the acceptance quality limit defined in Section 5.3 per ISO 2859-1.",
        "scope_title": "Scope of Supply", "scope_body": texts.HELIOS_SCOPE,
        "term_years": "two (2) years", "renewal": "two (2)-year", "nonrenewal_notice": "one hundred twenty (120)", "rate_cap": "5",
        "sla_body": texts.HELIOS_SLA, "penalty_title": "Liquidated Damages for Late Delivery", "penalty_body": texts.HELIOS_PENALTY,
        "payment_body": texts.HELIOS_PAYMENT,
        "liability_cap": "one hundred fifty percent (150%) of the total PO Value of all purchase orders released by the Buyer in the twelve (12) months preceding the event giving rise to liability",
        "cargo_carveout": "claims arising from non-conforming or defective Products, including the Supplier's replacement and RMA obligations under Section 5.3",
        "remedy_sentence": "Liquidated damages under Section 6 are the Buyer's exclusive financial remedy for late delivery of conforming Products as such, but do not limit the Buyer's quality remedies under Section 5.3, cancellation rights under Section 6.4, or termination rights under Section 10.",
        "insurance": "commercial general liability insurance of at least GBP 5,000,000 per occurrence including products liability, and employer's liability insurance as required by UK law",
        "fm_notice": "five (5) Business Days", "fm_term_days": "ninety (90)",
        "cure": "twenty (20) Business Days", "convenience_notice": "one hundred twenty (120)",
        "chronic_def": "an On-Time Delivery Rate below 97.0% in any three (3) calendar months within a rolling six (6)-month window",
        "law": "England and Wales",
        "dispute_clause": "The courts of England and Wales sitting in London have exclusive jurisdiction over any dispute arising out of or in connection with this Agreement, and each Party irrevocably submits to that jurisdiction.",
        "subcontract_limit": "the manufacture of any Product to a facility not listed in Schedule A",
    },
]


def render_contract(spec: dict[str, str]) -> str:
    """Fill the shared skeleton with one contract's terms and custom sections."""
    return TEMPLATE.format(**spec).strip() + "\n"


def main() -> None:
    """Write all six contracts to data/contracts/ and print a size summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        text = render_contract(spec)
        path = OUTPUT_DIR / spec["filename"]
        path.write_text(text, encoding="utf-8")
        n_words = len(text.split())
        print(f"Wrote {path.name}: {n_words:,} words, {len(text):,} chars")


if __name__ == "__main__":
    main()
