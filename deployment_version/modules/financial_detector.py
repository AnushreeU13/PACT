import spacy
from spacy.matcher import Matcher
import re

class FinancialDetector:
    """
    A class to detect and redact financial information from text using spaCy and regex.
    
    This detector identifies monetary values, credit card numbers, and bank account 
    information to ensure privacy before data is shared with external services.
    """
    def __init__(self):
        """
        Initializes the FinancialDetector by loading the spaCy model and 
        setting up custom matching patterns.
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not downloaded
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.matcher = Matcher(self.nlp.vocab)
        self._add_custom_patterns()

    def _add_custom_patterns(self):
        """
        Adds custom regex patterns to the spaCy Matcher for credit cards 
        and bank accounts.
        """
        # Pattern for Credit Cards (handles spaces, dashes, or no separators)
        card_pattern = [{"TEXT": {"REGEX": r"\b(?:\d{4}[ -]?){3}\d{4}\b"}}]
        self.matcher.add("CREDIT_CARD", [card_pattern])
        
        # Pattern for Bank Account Numbers (conceptually simple for demo)
        account_pattern = [{"TEXT": {"REGEX": r"\b\d{8,12}\b"}}]
        self.matcher.add("BANK_ACCOUNT", [account_pattern])

    def detect_and_redact(self, text):
        """
        Analyzes the input text to find and redact financial data points.
        
        Args:
            text (str): The raw input text to be sanitized.
            
        Returns:
            tuple: A tuple containing (redacted_text, sanitizations) where 
                   sanitizations is a list of dictionaries documenting what was found.
        """
        doc = self.nlp(text)
        redacted_text = text
        sanitizations = []

        # 1. Detect Standard spaCy MONEY entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                sanitizations.append({"original": ent.text, "type": "MONEY"})
                # We use a placeholder to avoid overlapping issues later
                redacted_text = redacted_text.replace(ent.text, "[REDACTED FINANCIAL]")

        # 2. Regex-based detection for more robust Credit Card matching
        # Handles 1111 2222 3333 4444, 1111-2222-3333-4444, and 1111222233334444
        cc_regex = r"\b(?:\d{4}[ -]?){3}\d{4}\b"
        for match in re.finditer(cc_regex, text):
            original = match.group()
            if original not in [s["original"] for s in sanitizations]:
                sanitizations.append({"original": original, "type": "CREDIT_CARD"})
                redacted_text = redacted_text.replace(original, "[REDACTED CREDIT_CARD]")

        # 3. Regex-based detection for Bank Accounts (8-12 digits)
        account_regex = r"\b\d{8,12}\b"
        for match in re.finditer(account_regex, text):
            original = match.group()
            if original not in [s["original"] for s in sanitizations]:
                sanitizations.append({"original": original, "type": "BANK_ACCOUNT"})
                redacted_text = redacted_text.replace(original, "[REDACTED BANK_ACCOUNT]")

        return redacted_text, sanitizations

if __name__ == "__main__":
    detector = FinancialDetector()
    sample = "Send $500 to account 1234567890. My card is 1234 5678 1234 5678."
    redacted, logs = detector.detect_and_redact(sample)
    print(f"Original: {sample}")
    print(f"Redacted: {redacted}")
    print(f"Logs: {logs}")
