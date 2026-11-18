"""Plausible English filler text for synthetic documents.

We deliberately generate *real English words* (not lorem ipsum) so the OCR verification
stage in Phase 2 has something meaningful to check against a dictionary.
"""

from __future__ import annotations

import random

SUBJECTS = [
    "The committee", "Our finance team", "The project lead", "Management", "The auditor",
    "Each department", "The board", "Our legal counsel", "The research group",
    "The new supplier", "The engineering unit", "Every staff member", "The client",
    "The regional office", "The security officer", "The procurement desk",
]
VERBS = [
    "has reviewed", "will approve", "must confirm", "recommends", "has rejected",
    "should verify", "intends to revise", "is preparing", "has requested", "will publish",
    "needs to update", "agreed to extend", "is concerned about", "plans to reduce",
    "has completed", "will investigate",
]
OBJECTS = [
    "the quarterly budget", "the attached proposal", "the payment schedule",
    "the revised contract", "the safety report", "the annual forecast",
    "the transport costs", "the hiring plan", "the delivery timeline",
    "the account balance", "the training programme", "the power supply upgrade",
    "the maintenance records", "the draft agreement", "the customer complaints",
    "the inventory figures", "the meeting minutes", "the travel request",
]
TAILS = [
    "before the end of the month", "as soon as possible", "without further delay",
    "in the next meeting", "following the recent inspection", "for the coming year",
    "despite the earlier concerns", "in line with company policy",
    "after consulting the manager", "by Friday afternoon", "with immediate effect",
    "once the funds are released", "pending final approval", "in good time",
]
CONNECTORS = [
    "Furthermore,", "However,", "In addition,", "As a result,", "Meanwhile,",
    "Please note that", "For this reason,", "In summary,", "Nevertheless,",
]
FIRST_NAMES = ["Adaeze", "Tunde", "Ngozi", "Emeka", "Funmi", "David", "Sarah", "Ibrahim",
               "Grace", "Samuel", "Amina", "Daniel", "Chioma", "Michael", "Kemi", "Peter"]
LAST_NAMES = ["Okafor", "Adeyemi", "Bello", "Johnson", "Eze", "Williams", "Balogun",
              "Smith", "Okonkwo", "Mohammed", "Taylor", "Adebayo", "Nwosu", "Brown"]
COMPANIES = ["Lagoon Systems Ltd", "Harbour Freight Co", "Northfield Energy",
             "Crescent Logistics", "Bluewater Analytics", "Summit Electrical Works",
             "Greenline Foods", "Atlas Engineering Group", "Kestrel Insurance"]
ITEMS = ["Copper cable 2.5mm", "Circuit breaker 32A", "LED panel light", "Power inverter",
         "Solar charge controller", "Distribution board", "Voltage stabiliser",
         "Network switch", "Office chairs", "Printer toner", "Safety helmets",
         "Steel conduit pipe", "Battery bank 200Ah", "Extension sockets", "Site survey"]


def sentence(rng: random.Random) -> str:
    s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}."
    if rng.random() < 0.35:
        s = f"{rng.choice(CONNECTORS)} {s[0].lower()}{s[1:]}"
    return s


def paragraph(rng: random.Random, n_min: int = 2, n_max: int = 5) -> str:
    return " ".join(sentence(rng) for _ in range(rng.randint(n_min, n_max)))


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def company(rng: random.Random) -> str:
    return rng.choice(COMPANIES)


def date(rng: random.Random) -> str:
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    return f"{rng.randint(1, 28)} {rng.choice(months)} {rng.randint(2019, 2026)}"
