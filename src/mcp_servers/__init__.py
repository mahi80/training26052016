"""Mock MCP servers for the production-layers build (BUILD_MANUAL.md Part II).

Three FastMCP servers expose company systems over the Model Context Protocol:

- ``sap_server``        — mock SAP: purchase orders per vendor
- ``servicenow_server`` — mock ServiceNow: incident tickets per carrier
- ``sql_server``        — thin MCP shim over the existing guard-railed SQL tool

``client`` is the sync stdio MCP client the agent tools dial through.
"""
