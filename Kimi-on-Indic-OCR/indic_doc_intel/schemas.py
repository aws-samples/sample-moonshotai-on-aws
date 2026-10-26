# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Field definitions for the document types used in this sample.

Add a new entry here to support another document type. Each field maps to a
short instruction that is passed to the model verbatim.
"""

DOCUMENT_SCHEMAS = {
    "electricity_bill": {
        "description": "An Indian electricity (bijli) bill issued by a state power distribution company.",
        "fields": {
            "distribution_company": "Name of the electricity board or distribution company, in English.",
            "consumer_name": "Account holder name exactly as printed, in the original script.",
            "consumer_name_english": "Account holder name transliterated into English (Latin script).",
            "consumer_number": "Consumer / account number, digits only.",
            "address": "Supply address translated into English.",
            "billing_period": "Billing period as 'YYYY-MM-DD to YYYY-MM-DD'.",
            "bill_date": "Bill issue date as YYYY-MM-DD.",
            "due_date": "Payment due date as YYYY-MM-DD.",
            "units_consumed_kwh": "Units consumed in this period, as a number.",
            "amount_due_inr": "Total payable amount on or before the due date, as a number.",
            "amount_after_due_date_inr": "Amount payable after the due date, as a number.",
        },
    },
    "kyc_form": {
        "description": "An Indian address-update / KYC application form filled in Hindi.",
        "fields": {
            "applicant_name": "Applicant name exactly as written, in the original script.",
            "applicant_name_english": "Applicant name transliterated into English (Latin script).",
            "father_or_spouse_name_english": "Father's or spouse's name transliterated into English.",
            "date_of_birth": "Date of birth as YYYY-MM-DD.",
            "gender": "One of 'male', 'female', 'other'.",
            "mobile_number": "10-digit mobile number, digits only.",
            "address": "Full address translated into English.",
            "pin_code": "6-digit PIN code.",
            "id_document_type": "Type of identity document referenced, in English.",
            "id_document_number": "Identity document number exactly as written.",
            "declaration_signed": "true if the declaration is signed, otherwise false.",
        },
    },
    "generic": {
        "description": "Any Indic-language document. Choose the most useful key fields yourself.",
        "fields": {},
    },
}


def get_schema(document_type: str) -> dict:
    try:
        return DOCUMENT_SCHEMAS[document_type]
    except KeyError:
        supported = ", ".join(sorted(DOCUMENT_SCHEMAS))
        raise ValueError(
            f"Unknown document type '{document_type}'. Supported: {supported}"
        ) from None
