"""A2A (agent-to-agent) layer: an external partner agent + the client tool.

``external_agent`` is a separate FastAPI service simulating SwiftShip's own
carrier agent — another company's system, reached over the network, never
imported into our graph. Our side talks to it via ``src.agents.tools_a2a``.
"""
