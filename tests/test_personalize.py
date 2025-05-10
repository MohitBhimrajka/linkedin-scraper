from src.personalize import craft_messages
sample = {"firstName":"Sara","headline":"Product Manager @Acme"}
out = craft_messages(sample)
assert set(out.keys())=={"connection","comment","followups","inmail_subject","inmail_body"}
assert all(out["followups"]) 