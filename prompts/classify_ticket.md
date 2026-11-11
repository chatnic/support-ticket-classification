You are routing customer support tickets for a digital bank. Read the ticket and choose the ONE queue that matches the customer's main request. If the ticket mentions more than one topic, choose the queue for the first or most urgent request.

Queues:
- card_issue: card declined, lost or damaged card, activation, replacement, PIN, limits, digital wallet setup
- fraud_dispute: unauthorized or unrecognized charges, stolen card details, scams or phishing, disputes with merchants, identity theft
- login_access: cannot sign in, passwords, locked profile, verification codes, two-factor or biometric login
- transfer_payment: transfers, wires, Zelle, ACH, bill pay, direct deposit, mobile deposit availability, payees
- account_fees: fees, charges, interest charged, refunds or waivers of fees
- loan_inquiry: loans, mortgages, rates, applications, payments, payoff, refinancing, hardship
- technical_bug: app or website errors, crashes, slow pages, display problems, broken features
- account_closure: closing or cancelling accounts, deceased customer accounts

Respond with JSON only, in exactly this form:
{"queue": "<one queue name from the list>", "confidence": <number between 0 and 1>}

Ticket:
"""
{ticket}
"""
