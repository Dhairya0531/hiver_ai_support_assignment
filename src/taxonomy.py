"""
Taxonomy definition for Apple Support customer inquiries.
Includes 8 mutually exclusive and collectively exhaustive (MECE) intents,
definitions, diagnostic keywords, and escalation criteria.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class IntentDefinition:
    id: str
    display_name: str
    description: str
    keywords: List[str]
    default_escalate: bool
    escalation_reason: str
    sample_queries: List[str]

INTENT_DEFINITIONS: Dict[str, IntentDefinition] = {
    "software_update_os": IntentDefinition(
        id="software_update_os",
        display_name="Software & OS Update Issues",
        description="Bugs, crashes, freezes, or glitches introduced during or after an iOS/macOS/watchOS update, including keyboard autocorrect bugs.",
        keywords=["ios", "update", "updated", "bug", "glitch", "crash", "freezing", "restart", "os", "software", "letter i", "auto-brightness"],
        default_escalate=False,
        escalation_reason="Standard software troubleshooting steps (restart, reset settings, wait for patch) can be provided automatically.",
        sample_queries=[
            "So there’s a glitch with .. you type “I “ and get “I” sometimes.",
            "My phone keeps freezing ever since updating to iOS 11.1. Any fix?",
            "Apps keep crashing after the latest software update."
        ]
    ),
    "battery_performance": IntentDefinition(
        id="battery_performance",
        display_name="Battery, Power & Performance",
        description="Rapid battery drainage, device overheating, unexpected shutdowns (dying at 20-30%), slow charging, or laggy performance.",
        keywords=["battery", "drain", "draining", "dying", "charge", "charging", "slow", "lag", "overheating", "hot", "battery life"],
        default_escalate=False,
        escalation_reason="Initial triage can provide battery health diagnostics, low power mode suggestions, and background refresh steps before hardware replacement.",
        sample_queries=[
            "iOS 11 has destroyed the speed and battery life on my iPhone 6. It dies at 30%!",
            "My battery drops 20% in 15 minutes without even using the phone.",
            "Phone is heating up while charging and running super slow."
        ]
    ),
    "account_apple_id_security": IntentDefinition(
        id="account_apple_id_security",
        display_name="Account, Apple ID & Security",
        description="Locked Apple ID accounts, forgotten passwords, 2FA verification codes, unauthorized access, stolen devices, or iCloud sync security.",
        keywords=["apple id", "password", "locked", "disabled", "security", "two factor", "2fa", "hacked", "stolen", "verification code", "icloud login"],
        default_escalate=True,
        escalation_reason="Security & PII policy: Account access, credentials, and stolen devices require authenticated private verification and human intervention.",
        sample_queries=[
            "My Apple ID has been locked for security reasons and I can't receive the verification code.",
            "My iPhone was stolen yesterday and I see unknown devices on my iCloud.",
            "Forgot my Apple ID password and the recovery phone number is old."
        ]
    ),
    "hardware_repair_service": IntentDefinition(
        id="hardware_repair_service",
        display_name="Hardware Damage & Physical Repair",
        description="Cracked screens, swollen batteries, broken physical buttons, camera blur, water damage, or booking Genius Bar appointments.",
        keywords=["screen", "cracked", "broken", "glass", "button", "camera", "water damage", "genius bar", "appointment", "store", "repair", "hardware"],
        default_escalate=True,
        escalation_reason="Physical hardware failures cannot be resolved over social media and require appointment scheduling or certified technician dispatch.",
        sample_queries=[
            "Dropped my phone and the screen shattered, touch is not working.",
            "Can you help me setup an appointment at the Genius Bar to fix my 6s battery?",
            "Explain to me why the lock button fell off my brand new iPhone 8."
        ]
    ),
    "billing_subscriptions": IntentDefinition(
        id="billing_subscriptions",
        display_name="App Store, Billing & Subscriptions",
        description="Unexpected credit card charges, refund requests, iTunes/App Store billing disputes, or cancelling subscriptions.",
        keywords=["billing", "charge", "charged", "refund", "subscription", "double billed", "itunes", "app store", "cancel", "payment", "card"],
        default_escalate=True,
        escalation_reason="Financial dispute policy: Payments, refunds, and billing details involve sensitive financial transactions requiring human review.",
        sample_queries=[
            "Hi, how do I inquire about being double billed for a purchase on Apple TV?",
            "I was charged $9.99 for an app subscription I cancelled last month. I want a refund.",
            "My credit card was charged by iTunes without my authorization."
        ]
    ),
    "connectivity_peripherals": IntentDefinition(
        id="connectivity_peripherals",
        display_name="Connectivity & Peripheral Sync",
        description="Wi-Fi disconnecting, Bluetooth pairing issues, cellular 'No Service', AirPods audio sync, or Apple Watch connection failure.",
        keywords=["bluetooth", "wifi", "wi-fi", "connection", "connect", "airpods", "apple watch", "pairing", "cellular", "no service", "carrier", "sync"],
        default_escalate=False,
        escalation_reason="Network and Bluetooth resets (toggle Airplane mode, forget device, reset network settings) can be auto-guided.",
        sample_queries=[
            "Since the newest update my Mac won't stay connected to my bluetooth mouse.",
            "AirPods keep disconnecting during phone calls on my iPhone 8.",
            "My phone says No Service constantly even after reinserting the SIM card."
        ]
    ),
    "escalation_human_complaint": IntentDefinition(
        id="escalation_human_complaint",
        display_name="Inbound Complaint & Escalation Request",
        description="High frustration, abusive or severe complaints, explicit demands to speak with a manager/human, threats to abandon brand.",
        keywords=["worst", "terrible", "useless", "human", "manager", "unacceptable", "lawsuit", "sucks", "disgusted", "agent", "supervisor", "hate"],
        default_escalate=True,
        escalation_reason="Customer sentiment policy: Severe customer dissatisfaction and explicit escalation requests must be routed immediately to a senior human agent.",
        sample_queries=[
            "The customer service is about the worst I have ever experienced, 2 weeks had my money, no device, useless!!",
            "I demand to speak to a human supervisor right now. Your automated bot is infuriating.",
            "This is unacceptable. I have called 5 times and no one resolved my issue."
        ]
    ),
    "general_guidance": IntentDefinition(
        id="general_guidance",
        display_name="General Inquiry & Feature Guidance",
        description="General questions about device compatibility, settings configuration, how-to usage, trade-in, or official documentation requests.",
        keywords=["how to", "how do i", "can i", "is it possible", "feature", "settings", "trade in", "compatible", "options", "spanish", "language"],
        default_escalate=False,
        escalation_reason="Informational guidance, official documentation links, and feature steps can be served safely without human agent involvement.",
        sample_queries=[
            "How do I transfer photos from my iPhone to my PC without iTunes?",
            "Can I use an iPhone X purchased in the US with a European SIM card?",
            "How do I turn on two-factor authentication in settings?"
        ]
    )
}

ALL_INTENTS = list(INTENT_DEFINITIONS.keys())

# Escalation reasons enum/constants
ESCALATION_REASONS = {
    "ACCOUNT_SECURITY": "Account Security: Account recovery, credentials, or potential security breach requires private identity verification.",
    "BILLING_DISPUTE": "Financial Dispute: Transactions, payment adjustments, or refunds require human billing authorization.",
    "HARDWARE_REPAIR": "Physical Hardware: Device damage or hardware failure requires in-store Genius Bar or depot inspection.",
    "FRUSTRATED_CUSTOMER": "Customer Dissatisfaction: High negative sentiment or explicit human agent request requires senior support handling.",
    "LOW_CONFIDENCE": "Low Model Confidence: Query is ambiguous, off-topic, or multi-intent, posing hallucination risk if auto-handled.",
    "AUTO_HANDLE": "Auto-Handled: Resolvable via verified knowledge base troubleshooting steps without human intervention."
}
