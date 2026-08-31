"""Scheduled background work.

Runs in-process, inside the API's lifespan, with no broker. Each job is
guarded by a database lock so that running several API workers does not
run the job several times.
"""
