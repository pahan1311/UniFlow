"""Business rules.

Every rule the system enforces lives here, and nowhere else. Services take
a SQLAlchemy session and a Principal, and are callable from the API, from
a scheduled job, from a script or from a test with no framework attached.

This is the layer the old design had no room for: routing, persistence and
duplicate detection were orchestrated inline in app.py between st.markdown
calls, and ingest_email.py held a second, divergent copy of the same flow.
"""
