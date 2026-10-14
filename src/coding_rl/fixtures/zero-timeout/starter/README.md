Fix normalize_timeout(timeout=None, default=30): omitted or null timeout uses the default; every supplied nonnegative integer, including zero, must be preserved. Preserve the public API.
