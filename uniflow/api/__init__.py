"""The HTTP surface.

Routers translate between JSON and the domain, and translate domain
exceptions into status codes. They contain no business rules: if a rule
appears in this package it is in the wrong place.
"""
