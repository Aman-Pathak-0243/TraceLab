"""Synthetic fixtures for CareDesk, a FICTIONAL preventive-care booking service.

All members, appointments, prices and policies are invented. Nothing here is
medical guidance and no real patient data is processed.
"""

from __future__ import annotations

from datetime import date

# A fixed "today" keeps date-dependent behaviour reproducible across runs.
REFERENCE_DATE = date(2026, 11, 1)

# What the scheduling backend picks when a booking request has no date.
NEXT_AVAILABLE_DATE = "2026-11-05"

PACKAGES: dict[str, dict] = {
    "PKG-BASIC": {
        "package_id": "PKG-BASIC",
        "name": "Basic Wellness Screening",
        "price_usd": 120,
        "duration_min": 45,
        "includes": ["blood pressure check", "BMI measurement", "basic blood panel"],
    },
    "PKG-HEART": {
        "package_id": "PKG-HEART",
        "name": "Heart Health Check",
        "price_usd": 210,
        "duration_min": 60,
        "includes": ["resting ECG", "lipid panel", "blood pressure check"],
    },
    "PKG-SENIOR": {
        "package_id": "PKG-SENIOR",
        "name": "Healthy Ageing Review",
        "price_usd": 260,
        "duration_min": 90,
        "includes": ["bone density screening", "vision check", "basic blood panel"],
    },
}

MEMBERS: set[str] = {"M-1001", "M-1002"}

APPOINTMENTS: dict[str, dict] = {
    "APT-1001": {
        "appointment_id": "APT-1001",
        "member_id": "M-1001",
        "package_id": "PKG-BASIC",
        "date": "2026-11-03",
        "time": "09:30",
        "location": "Northside Clinic (fictional)",
        "status": "confirmed",
    },
    "APT-1002": {
        "appointment_id": "APT-1002",
        "member_id": "M-1002",
        "package_id": "PKG-HEART",
        "date": "2026-11-12",
        "time": "14:00",
        "location": "Riverside Clinic (fictional)",
        "status": "cancelled",
    },
}

POLICIES: dict[str, str] = {
    "cancellation": (
        "Appointments can be cancelled free of charge up to 24 hours before the scheduled "
        "time. Later cancellations incur a $25 fee."
    ),
    "rescheduling": (
        "You can reschedule once at no cost if you do so at least 24 hours in advance."
    ),
    "preparation": (
        "For packages that include a blood panel, do not eat for 8 hours before the "
        "appointment. Water is fine."
    ),
    "payment": "Payment is collected at the clinic. Card and bank transfer are accepted.",
}
