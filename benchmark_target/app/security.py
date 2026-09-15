"""Password hashing — stdlib only (hashlib.pbkdf2_hmac), no new dependency.

Not trying to be production-grade crypto here; this is a local benchmark fixture app whose
credentials are all deterministic seed data documented in seed.py anyway. The point is just
"not plaintext in the DB", not defense against a real attacker.

W_determinism_salt (Phase 8): the salt is derived from the password itself (HMAC over a
fixed fixture-only key), not os.urandom(16). A random salt made every reseed produce a
byte-different `users` table even though nothing meaningful changed — caught live by
test_determinism.py's reseed-equality tests, which is exactly the kind of drift spec 47
("avoid nondeterministic: random seed...") says a benchmark fixture must not have. The
security cost (same password -> same hash, enabling correlation across accounts sharing a
password) is fine here since nothing in this file was ever trying to resist a real
attacker in the first place — see the module docstring above.
"""

import hashlib
import hmac

_ITERATIONS = 200_000
_FIXED_SALT_KEY = b"benchmark-target-fixed-salt-key-v1"


def hash_password(password: str) -> str:
    salt = hmac.new(_FIXED_SALT_KEY, password.encode("utf-8"), hashlib.sha256).digest()[:16]
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return hmac.compare_digest(actual, expected)
