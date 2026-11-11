"""Generate synthetic customer support tickets for a digital bank.

Tickets are built from varied templates with slot fillers, then made messy the way real
tickets are: inconsistent casing, typos, greetings and sign-offs, extra context, tickets
that mention a second topic, and a small share of mislabeled examples (human agents
disagree too).

The template_id column records which phrasing pattern produced each ticket. It is never a
model input; it exists so evaluation can hold out whole phrasings and test generalization
to wordings the model has never seen.

Usage:
    python data/generate_data.py --n 12000 --seed 42 --out data/raw/tickets.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = {
    "card": ["debit card", "credit card", "card", "visa card", "virtual card", "card ending in {d4}"],
    "amount": ["${n2}", "${n3}", "${n3}.{d2}", "{n2} dollars", "${n1},{d3}"],
    "merchant": ["Amazon", "a gas station", "Uber", "an online store", "Walmart", "a restaurant", "Netflix",
                 "a hotel", "Target", "a merchant I don't recognize", "an app store"],
    "when": ["yesterday", "today", "this morning", "last night", "on Monday", "two days ago", "last week",
             "over the weekend", "on the 3rd"],
    "device": ["iPhone", "Android phone", "phone", "laptop", "tablet", "work computer", "new Samsung"],
    "transfer": ["transfer", "wire", "Zelle payment", "ACH transfer", "payment", "bill payment", "direct deposit"],
    "fee": ["overdraft fee", "monthly maintenance fee", "foreign transaction fee", "late fee", "ATM fee",
            "wire fee", "fee"],
    "loan": ["personal loan", "auto loan", "mortgage", "home equity line", "loan", "student loan refinance"],
}

TEMPLATES = {
    "card_issue": [
        "Is there a way to freeze my {card} temporarily",
        "The magnetic strip on my {card} is worn out",
        "My {card} was swallowed by an ATM {when}",
        "I need to update the billing address linked to my {card}",
        "why does my {card} get rejected at {merchant}",
        "Can you send me a new {card} with my updated name",
        "The {card} won't tap to pay anymore",
        "I'd like to add my {card} to Apple Pay but it fails",
        "Requesting an expedited replacement for my {card}",
        "My {card} shows as inactive",
        "My {card} was declined at {merchant} {when} even though I have money in my account",
        "I need a replacement {card}, mine is damaged and the chip doesn't work",
        "How do I activate my new {card}?",
        "I lost my {card} {when}, can you lock it and send a new one",
        "My {card} keeps getting declined for online purchases",
        "The new {card} never arrived in the mail, it's been two weeks",
        "Can I increase the daily spending limit on my {card}",
        "My {card} PIN isn't working at the ATM",
        "card expired and I didn't get a new one",
        "Contactless payments stopped working on my {card}",
    ],
    "fraud_dispute": [
        "I never authorized the {amount} payment to {merchant}",
        "Please investigate a suspicious transaction from {merchant}",
        "I think someone has my card details, there are charges I don't recognize",
        "Reporting identity theft, someone opened an account in my name",
        "A refund from {merchant} was promised but never posted, I want to file a claim",
        "Weird small charges of $1 keep appearing on my statement",
        "I clicked a phishing link and entered my details, what do I do",
        "someone withdrew {amount} from an ATM and it wasn't me",
        "need to report a fraudulent transaction asap",
        "There are purchases from another country on my statement",
        "There is a charge of {amount} from {merchant} that I did not make",
        "I see transactions on my account {when} that aren't mine, I think my {card} was stolen",
        "Someone used my {card} at {merchant} for {amount}, I need to dispute this",
        "I got a text saying there was suspicious activity, was that really you",
        "I want to dispute a charge from {merchant}, they never delivered my order",
        "Unauthorized purchase of {amount} {when}, please help",
        "I was charged twice by {merchant} for the same purchase",
        "Got a call from someone claiming to be the bank asking for my code, is this a scam",
        "fraudulent charges on my statement, I need my money back",
        "my account was hacked and money was taken out",
    ],
    "login_access": [
        "The app says my credentials are invalid",
        "Password reset link expired before I could use it",
        "I can't remember my security questions",
        "Online banking won't accept my new password",
        "My one-time passcode never shows up",
        "How do I unlock my profile after failed attempts",
        "Can't access my account since I changed my email address",
        "biometric sign in not working on my {device}",
        "It keeps asking me to verify my identity in a loop",
        "I'm unable to get into my account at all",
        "I can't log in to my account on my {device}",
        "I forgot my password and the reset email never comes",
        "My account is locked after too many login attempts",
        "Not receiving the verification code by text when I try to sign in",
        "I got a new {device} and now I can't get past two-factor authentication",
        "Face ID login stopped working on my {device}",
        "It says my username doesn't exist but I've had an account for years",
        "How do I change the phone number used for login codes",
        "keep getting logged out of the app every few minutes",
        "locked out of online banking, need access urgently",
    ],
    "transfer_payment": [
        "Money I moved to savings {when} hasn't shown up",
        "The recipient says they never received the {amount} I sent",
        "Why is my outgoing {transfer} on hold",
        "I need the confirmation number for a {transfer} I made {when}",
        "Can I set up recurring payments to my landlord",
        "A {transfer} from my other bank was returned",
        "Funds from a mobile deposit are not available yet",
        "my rent payment bounced even though the money was there",
        "How do I add a new payee",
        "I want to stop a payment I just sent",
        "My {transfer} of {amount} {when} still hasn't arrived",
        "I sent a {transfer} to the wrong account, can it be reversed",
        "How long does a {transfer} take to clear",
        "My {transfer} is stuck as pending since {when}",
        "The {transfer} was deducted from my account but the recipient never got it",
        "I need to cancel a scheduled {transfer}",
        "What is the limit for a {transfer} per day",
        "My paycheck {transfer} hasn't shown up {when}",
        "Can I send an international {transfer} from the app",
        "bill pay sent {amount} to the wrong company",
    ],
    "account_fees": [
        "I was billed {amount} for something called a service charge",
        "Why am I paying a monthly charge on a free checking account",
        "Please reverse the penalty charged when my balance dipped below the minimum",
        "What does the {fee} on my account mean",
        "I keep getting charged for paper statements",
        "Is there a charge for using my card abroad",
        "Can I get the {fee} credited back",
        "the {fee} seems too high",
        "Why was {amount} deducted as a charge",
        "how much does it cost to send a wire",
        "Why was I charged a {fee} {when}",
        "I'd like a refund of the {fee}, it was my first time",
        "There's a {fee} of {amount} on my statement I don't understand",
        "How do I avoid the {fee} going forward",
        "You charged me a {fee} even though I had enough balance",
        "Can you waive the {fee}, I've been a customer for years",
        "what fees do you charge for using an out of network atm",
        "I was charged interest on my credit card but I paid in full",
        "Please explain the {fee} on my account",
        "Got hit with a {fee} again, this is the third month",
    ],
    "loan_inquiry": [
        "How much could I borrow for a {loan}",
        "When is my next {loan} installment due",
        "Can I defer payments on my {loan}",
        "I'd like to discuss consolidating my debt",
        "What term lengths do you offer for a {loan}",
        "Has my {loan} been approved yet",
        "I want to pay off my {loan} early",
        "How is the APR on a {loan} calculated",
        "Do you offer pre-approval for a {loan}",
        "my {loan} statement shows the wrong balance",
        "What is the current interest rate on a {loan}",
        "I want to apply for a {loan}, what documents do I need",
        "Can I make an extra payment on my {loan} without penalty",
        "My {loan} application has been pending since {when}",
        "How do I get a payoff quote for my {loan}",
        "Am I eligible to refinance my {loan}",
        "I need to change the due date of my {loan} payment",
        "Why was my {loan} application declined",
        "Is there a hardship program for my {loan}, I lost my job",
        "what's the minimum credit score for a {loan}",
    ],
    "technical_bug": [
        "The app freezes when I try to make a deposit",
        "Something is broken on the website, pages won't load",
        "Since the update the app keeps showing an unexpected error",
        "I can't see my accounts in the app, it just spins",
        "The check image upload fails every time on my {device}",
        "Your site gives me a 404 when I click on statements",
        "The app logs my spending in the wrong categories",
        "Charts in the app aren't displaying",
        "The app drains my battery and runs really slow",
        "I found a bug where the transfer screen cuts off text",
        "The app crashes every time I open it on my {device}",
        "The balance shown in the app is different from the website",
        "Mobile check deposit keeps failing with an error",
        "The app won't load my transaction history on my {device}",
        "Getting an error code 500 when I try to view statements",
        "Push notifications stopped working after the last update",
        "The website is extremely slow and times out",
        "I can't download my statements as PDF, the button does nothing",
        "the app shows a blank screen after the update on my {device}",
        "Dark mode makes the text unreadable in the app",
    ],
    "account_closure": [
        "Please terminate my checking account",
        "I no longer need my savings account, shut it down",
        "Steps to permanently delete my account",
        "I'd like to cancel all my services with you",
        "Can you close the account I opened for my business",
        "Going to switch banks, how do I end my relationship with you",
        "How long does it take to close an account and get my remaining funds",
        "remove my account, I don't use it anymore",
        "My father passed away and we need to close his account",
        "Close the joint account I share with my ex",
        "I want to close my account",
        "How do I close my savings account and transfer the balance",
        "Please close my account, I'm moving to another bank",
        "What happens to my money if I close my account",
        "I'd like to cancel my credit card and close the account",
        "Can I close a joint account online or do I need to visit a branch",
        "close my checking account effective immediately",
        "I'm unhappy with the service and want to close all my accounts",
    ],
}
# Rough real-world mix: card and transfer issues dominate, closures are rare.
SHARES = {"card_issue": 0.19, "fraud_dispute": 0.14, "login_access": 0.16, "transfer_payment": 0.17,
          "account_fees": 0.11, "loan_inquiry": 0.08, "technical_bug": 0.10, "account_closure": 0.05}
URGENT = {"fraud_dispute", "login_access"}

GREETINGS = ["", "", "", "Hi, ", "Hello, ", "Hi there. ", "Good morning, ", "Hey ", "To whom it may concern, "]
SIGNOFFS = ["", "", "", " Thanks.", " Thank you!", " Please help.", " Thanks in advance.",
            " This is really frustrating.", " Appreciate it.", " ASAP please."]
CONTEXT = ["", "", "", "", " I've been a customer for 10 years.", " I already tried calling.",
           " I'm traveling right now.", " This happened before too.", " I need this resolved today.",
           " The chat bot didn't help."]


def _fill(text: str, rng: np.random.Generator) -> str:
    def slot(m):
        opts = SLOTS[m.group(1)]
        return opts[rng.integers(len(opts))]
    text = re.sub(r"\{(card|amount|merchant|when|device|transfer|fee|loan)\}", slot, text)
    for _ in range(2):
        text = re.sub(r"\{(card|amount|merchant|when|device|transfer|fee|loan)\}", slot, text)
    text = text.replace("{d4}", f"{rng.integers(1000, 9999)}").replace("{d3}", f"{rng.integers(100, 999)}")
    text = text.replace("{d2}", f"{rng.integers(10, 99)}").replace("{n1}", f"{rng.integers(1, 9)}")
    text = text.replace("{n2}", f"{rng.integers(10, 99)}").replace("{n3}", f"{rng.integers(100, 999)}")
    return text


def _typos(text: str, rng: np.random.Generator, rate: float) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.integers(3)
            if op == 0:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == 1:
                chars[i] = ""
            else:
                chars[i] = chars[i] * 2
    return "".join(chars)


def generate(n: int = 12_000, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(SHARES)
    probs = np.array([SHARES[c] for c in cats])
    labels = rng.choice(cats, size=n, p=probs / probs.sum())
    rows = []
    for i, cat in enumerate(labels):
        t_idx = rng.integers(len(TEMPLATES[cat]))
        body = _fill(TEMPLATES[cat][t_idx], rng)
        # 12% of tickets also mention a secondary topic; the label stays the primary intent.
        secondary = ""
        if rng.random() < 0.12:
            other = cats[rng.integers(len(cats))]
            if other != cat:
                o_idx = rng.integers(len(TEMPLATES[other]))
                body += ". Also, " + _fill(TEMPLATES[other][o_idx], rng).lower()
                secondary = f"{other}#{o_idx}"
        text = (GREETINGS[rng.integers(len(GREETINGS))] + body + CONTEXT[rng.integers(len(CONTEXT))]
                + SIGNOFFS[rng.integers(len(SIGNOFFS))])
        style = rng.random()
        if style < 0.25:
            text = text.lower()
        elif style < 0.28:
            text = text.upper()
        text = _typos(text, rng, rate=rng.choice([0, 0, 0.01, 0.025]))
        channel = rng.choice(["email", "chat", "app_form"], p=[0.35, 0.45, 0.20])
        rows.append({"ticket_id": f"T{500000 + i}", "channel": channel, "text": text, "category": cat,
                     "template_id": f"{cat}#{t_idx}", "secondary_template_id": secondary})
    df = pd.DataFrame(rows)
    # Label noise: agents sometimes pick the wrong queue.
    flip = rng.random(n) < label_noise
    df.loc[flip, "category"] = rng.choice(cats, size=flip.sum())
    df["priority"] = np.where(df["category"].isin(URGENT), "high", "normal")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw/tickets.csv"))
    a = p.parse_args()
    df = generate(a.n, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} tickets to {a.out}\n{df.category.value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()
