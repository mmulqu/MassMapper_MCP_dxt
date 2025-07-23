#!/usr/bin/env python3
"""
MassGIS MCP Server with Notepad
================================
* Keyword-based layer search using metadata and search terms
* Reads per-layer JSON schemas from a `layers/` folder  
* Includes notepad tool for recording and exporting large datasets
* Exposes tools: search_layers · list_categories · get_layer_details · 
  describe_layer_schema · query_spatial · intersect_with_town · find_nearby · 
  massmapper_link · notepad_write · notepad_read · notepad_clear · notepad_export

"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import httpx
from pyproj import Transformer

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities, TextContent, Tool, EmbeddedResource
import mcp.server.stdio

# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
LAYER_JSON_ROOT = ROOT_DIR / "layers_with_schema"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("massgis-mcp")
logger.setLevel(logging.DEBUG)


class MassGISCatalogServer:
    """Lightweight MCP server for MassGIS WFS with notepad functionality."""

    def __init__(self) -> None:
        self.server = Server("massgis-vector")
        # Draft-spec extras (plain dicts)
        self.server.prompts   = {}
        self.server.resources = {}
        self.session: Optional[httpx.AsyncClient] = None
        self.export_dir = (ROOT_DIR / "exports").resolve()
        self.export_dir.mkdir(exist_ok=True)

        self.endpoints = {
            "wfs_base": "https://gis-prod.digital.mass.gov/geoserver/wfs",
            "wms_base": "https://gis-prod.digital.mass.gov/geoserver/wms",
        }

        self.layer_catalog: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, str] = {}
        self.inv_index: Dict[str, List[str]] = {}
        self.geom_col_cache: Dict[str, str] = {}
        self._srid_cache: Dict[str, str] = {}
        self._bbox_by_municipality: dict[str, str] = {}
        self._schema_cache: Dict[str, List[Tuple[str, str]]] = {}

        # Track used layers for MassMapper link generation
        self.used_layers: List[str] = []
        self.last_municipality: Optional[str] = None

        # Notepad storage
        self.notepad: Dict[str, Any] = {
            "entries": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "entry_count": 0
            }
        }

        self._load_layer_catalog()
        self._register_handlers()
        self._prepare_prompt_roots()
        self._prepare_resource_roots()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_layer_catalog(self) -> None:
        if not LAYER_JSON_ROOT.exists():
            logger.error("layers folder %s not found", LAYER_JSON_ROOT)
            return
        skipped = 0
        for p in LAYER_JSON_ROOT.rglob("*.json"):
            try:
                js = json.loads(p.read_text(encoding="utf-8"))
                lid = js.get("layer_id") or p.stem
                self.layer_catalog[lid] = js
                cat = js.get("category", "uncategorized")
                self.categories.setdefault(cat, cat)
                for term in js.get("search_terms", []):
                    self.inv_index.setdefault(term.lower(), []).append(lid)
                col_sum = js.get("column_summary") or ""
                for tok in re.split(r"[\s,]+", col_sum):
                    if tok:
                        self.inv_index.setdefault(tok.lower(), []).append(lid)
                if cat:
                    self.inv_index.setdefault(cat.lower(), []).append(lid)
            except Exception as e:
                skipped += 1
                logger.debug("Skip %s – %s", p, e)
        logger.info("Loaded %d layers (%d skipped)", len(self.layer_catalog), skipped)

    # ------------------------------------------------------------------
    # MCP wiring
    # ------------------------------------------------------------------
    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def _list_tools() -> List[Tool]:
            cats = list(self.categories) + ["all"]
            return [
                Tool(
                    name="search_layers",
                    description="Search MassGIS layers by keyword. Always run first.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {"type": "string", "enum": cats},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                ),
                
                Tool(
                    name="describe_layer_schema",
                    description=(
                        "🔥 CRITICAL: Run IMMEDIATELY after search_layers to get exact column names. "
                        "MANDATORY before any spatial queries. Never assume column names."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"layer_name": {"type": "string"}},
                        "required": ["layer_name"],
                    },
                ),
                
                Tool(
                    name="query_spatial",
                    description=(
                        "⭐ PRIMARY TOOL for spatial queries using ECQL. "
                        "🌟 UNIVERSAL PATTERN: DWITHIN(geom, collectGeometries(queryCollection('layer', 'geom_col', 'filter')), distance, meters). "
                        "Works for single/multiple/all features with same syntax. "
                        "Supports: INTERSECTS, WITHIN, CONTAINS, TOUCHES, CROSSES, DWITHIN, BEYOND."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "layer_name": {"type": "string"},
                            "cql": {"type": "string"},
                            "max_features": {"type": "integer", "default": 50},
                            "start_index": {"type": "integer", "default": 0},
                            "sort_by": {"type": "string"},
                            "use_notepad": {"type": "boolean", "default": False, "description": "Save results to notepad for large datasets"},
                        },
                        "required": ["layer_name", "cql"],
                    },
                ),
                
                Tool(
                    name="intersect_with_town",
                    description=(
                        "Find features within municipal boundaries. Use for 'features in [town/city]' queries. "
                        "Handles coordinate systems automatically."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "layer_name": {"type": "string"},
                            "municipality": {"type": "string"},
                            "max_features": {"type": "integer", "default": 50},
                            "use_notepad": {"type": "boolean", "default": False, "description": "Save results to notepad for large datasets"},
                        },
                        "required": ["layer_name", "municipality"],
                    },
                ),
                
                Tool(
                    name="find_nearby",
                    description=(
                        "🚨 LAST RESORT: Find features near lat/lon coordinates. "
                        "⚠️ ONLY use when user provides raw coordinates, NOT specific features! "
                        "Use query_spatial with universal DWITHIN pattern for exact features."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "layer_name": {"type": "string"},
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                            "radius_meters": {"type": "number", "default": 1000},
                            "use_notepad": {"type": "boolean", "default": False, "description": "Save results to notepad for large datasets"},
                        },
                        "required": ["layer_name", "latitude", "longitude"],
                    },
                ),
                
                Tool(
                    name="list_categories",
                    description="List all available data categories.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                
                Tool(
                    name="get_layer_details", 
                    description="Get detailed metadata for a specific layer.",
                    inputSchema={
                        "type": "object",
                        "properties": {"layer_name": {"type": "string"}},
                        "required": ["layer_name"],
                    },
                ),
                
                Tool(
                    name="massmapper_link",
                    description=(
                        "🗺️ Generate interactive map link showing analysis results. "
                        "ALWAYS call this after spatial analysis to visualize results."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "municipalities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "use_union_bbox": {"type": "boolean", "default": True},
                            "bbox": {"type": "string"},
                            "include_all_used": {"type": "boolean", "default": True},
                            "specific_layers": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        }
                    },
                ),
                
                # Notepad tools
                Tool(
                    name="notepad_write",
                    description="Write arbitrary content to the notepad for later export.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "title": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["content"],
                    },
                ),
                
                Tool(
                    name="notepad_read",
                    description="Read notepad contents. Set last_n to limit entries, summary_only for overview.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "last_n": {"type": "integer"},
                            "summary_only": {"type": "boolean", "default": False},
                        },
                    },
                ),
                
                Tool(
                    name="notepad_clear",
                    description="Clear all notepad entries. Set confirm=True to actually clear.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "confirm": {"type": "boolean", "default": False},
                        },
                        "required": ["confirm"],
                    },
                ),
                
                Tool(
                    name="notepad_export",
                    description="Export notepad contents directly to file system.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "format": {"type": "string", "enum": ["csv", "json"], "default": "csv"},
                            "filename_prefix": {"type": "string", "default": "massgis_export"},
                        },
                    },
                ),
            ]

        # -- NEW: advertise prompts ------------------------------------------------
        @self.server.list_prompts()
        async def _list_prompts():
            return [
                {"name": name, "text": text}
                for name, text in self.server.prompts.items()
            ]

        # -- NEW: advertise static resources --------------------------------------
        @self.server.list_resources()
        async def _list_resources():
            # You don’t ship PDFs or icons yet; return empty list for now
            return []

        @self.server.call_tool()
        async def _call_tool(name: str, args: Dict[str, Any]):  # noqa: C901
            try:
                if name == "search_layers":
                    return await self._search_layers(
                        args["query"], args.get("category"), args.get("limit", 10)
                    )
                if name == "list_categories":
                    return await self._list_categories()
                if name == "get_layer_details":
                    return await self._get_layer_details(args["layer_name"])
                if name == "intersect_with_town":
                    return await self._intersect_with_town(
                        args["layer_name"],
                        args["municipality"],
                        args.get("max_features", 50),
                        args.get("use_notepad", False),
                    )
                if name == "describe_layer_schema":
                    return await self._describe_layer_schema(args["layer_name"])
                if name == "query_spatial":
                    return await self._query_spatial(
                        args["layer_name"],
                        args["cql"],
                        args.get("max_features", 50),
                        args.get("start_index", 0),
                        args.get("sort_by"),
                        args.get("use_notepad", False),
                    )
                if name == "find_nearby":
                    return await self._find_nearby(
                        args["layer_name"],
                        args["latitude"],
                        args["longitude"],
                        args.get("radius_meters", 1000),
                        args.get("use_notepad", False),
                    )
                if name == "massmapper_link":
                    return await self._generate_massmapper_link(
                        municipalities=args.get("municipalities"),
                        use_union_bbox=args.get("use_union_bbox", True),
                        bbox=args.get("bbox"),
                        include_all_used=args.get("include_all_used", True),
                        specific_layers=args.get("specific_layers"),
                    )
                # Notepad tools
                if name == "notepad_write":
                    return await self._notepad_write(
                        args["content"],
                        args.get("title"),
                        args.get("metadata"),
                    )
                if name == "notepad_read":
                    return await self._notepad_read(
                        args.get("last_n"),
                        args.get("summary_only", False),
                    )
                if name == "notepad_clear":
                    return await self._notepad_clear(args["confirm"])
                if name == "notepad_export":
                    return await self._notepad_export(
                        args.get("format", "csv"),
                        args.get("filename_prefix", "massgis_export"),
                    )
            except Exception as e:
                logger.error("Tool %s failed – %s", name, e, exc_info=True)
                return [TextContent(type="text", text=f"Error in {name}: {e}")]
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _prepare_prompt_roots(self) -> None:
        """Enhanced prompts with universal spatial pattern."""
        
        self.server.prompts["systemPrompt"] = (
            "You are the MassGIS MCP expert specializing in GeoServer WFS 1.1.0 and spatial analysis.\n\n"
            
            "🔥 UNIVERSAL SPATIAL QUERY PATTERN (CRITICAL):\n"
            "For ANY distance-based query ('find A within distance of B'), use this ONE pattern:\n"
            "DWITHIN(target_geom, collectGeometries(queryCollection('source_layer', 'geom_col', 'filter')), distance, meters)\n\n"
            
            "✅ This pattern works for ALL cases:\n"
            "• Single feature: 'name = \"EDITH M. FOX LIBRARY\"'\n"
            "• Multiple features: 'type = \"PUBLIC\"'\n"
            "• All features: 'INCLUDE'\n\n"
            
            "🚨 CRITICAL RULES:\n"
            "• NEVER use find_nearby() when you have specific features\n"
            "• NEVER approximate coordinates from training data\n"
            "• ALWAYS use exact feature geometries with the universal pattern\n\n"
            
            "📋 MANDATORY WORKFLOW:\n"
            "1. search_layers → 2. describe_layer_schema → 3. query_spatial → 4. massmapper_link\n\n"
            
            "📏 DISTANCE CONVERSIONS: 0.5mi=805m, 1mi=1609m, 2mi=3218m, 5mi=8047m\n\n"
            
            "📝 NOTEPAD USAGE:\n"
            "For queries returning many results (>20), consider using use_notepad=True to save all results.\n"
            "Users can then export the complete dataset as CSV or JSON.\n"
        )

        self.server.prompts["developerPrompt"] = (
            "### Enhanced GeoServer ECQL Spatial Operations\n\n"
            
            "🌟 UNIVERSAL PATTERN:\n"
            "DWITHIN(target_geom, collectGeometries(queryCollection('source_layer', 'geom_col', 'filter')), distance, meters)\n\n"
            
            "📋 FILTER EXAMPLES:\n"
            "• Specific: 'name = \"FOX LIBRARY\"'\n"
            "• Type: 'type = \"PUBLIC\"'\n" 
            "• Area: 'town = \"BOSTON\"'\n"
            "• Multiple: 'town IN (\"BOSTON\", \"CAMBRIDGE\")'\n"
            "• All: 'INCLUDE'\n\n"
            
            "🎯 CORE SPATIAL PREDICATES:\n"
            "• DWITHIN(geom, other, distance, meters) - within distance\n"
            "• INTERSECTS(geom, other) - any overlap\n"
            "• WITHIN(geom, other) - completely inside\n"
            "• CONTAINS(geom, other) - completely contains\n"
            "• TOUCHES(geom, other) - boundaries touch\n\n"
            
            "🛠️ GEOMETRY FUNCTIONS:\n"
            "• buffer(geometry, distance) - create buffer\n"
            "• area(geometry) - calculate area\n"
            "• distance(geom_a, geom_b) - measure distance\n\n"
            
            "⚠️ QUOTE ESCAPING: Use 'name = \"VALUE\"' in filters\n\n"
            
            "📝 NOTEPAD TOOLS:\n"
            "• notepad_write - Add arbitrary content\n"
            "• notepad_read - View entries (with summary option)\n"
            "• notepad_clear - Clear all entries (requires confirmation)\n"
            "• notepad_export - Export as CSV or JSON\n"
        )

    def _prepare_resource_roots(self) -> None:
        """
        Advertise large static artefacts so the client can fetch / cache them.
        """
        # This is now handled dynamically for schemas, and other resources
        # are not needed for this server's logic.
        pass

    # ------------------------------------------------------------------
    # Tool logic
    # ------------------------------------------------------------------
    async def _search_layers(self, query: str, category: Optional[str], limit: int):
        q_words = set(query.lower().split())

        # Use keyword search only
        kw_hits = [
            lid for lid, info in self.layer_catalog.items()
            if q_words & {t.lower() for t in info.get("search_terms", [])}
        ]
        
        # Start with keyword hits, or all layers if no hits
        cands = kw_hits if kw_hits else list(self.layer_catalog)
        logger.debug("keyword candidates: %d layers", len(cands))

        scored: List[Tuple[int, str, Dict[str, Any]]] = []
        for lid in cands:
            info = self.layer_catalog[lid]
            if category and category != "all" and info.get("category") != category:
                continue
            score = 0
            st = {t.lower() for t in info.get("search_terms", [])}
            score += 7 * len(q_words & st)

            # Add scoring for column_summary tokens
            col_tokens = set(re.split(r"[\s,]+", info.get("column_summary", "").lower()))
            score += 5 * len(q_words & col_tokens)  # tweak weight as you like

            title = info.get("title") or info.get("document_title", lid)
            blob = f"{title} {info.get('description','')}".lower()
            score += sum(2 for w in q_words if w in blob)
            if category and info.get("category") == category:
                score += 3
            if score:
                scored.append((score, lid, info))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:limit]

        if not scored:
            return [TextContent(type="text", text=f"No layers found for '{query}'.")]

        out: List[str] = [f"# Search Results for '{query}'\n\n"]
        out.append("**IMPORTANT**: Always run `describe_layer_schema` FIRST before querying any layer!\n\n")
        for i, (_, lid, info) in enumerate(scored, 1):
            title = info.get("title") or info.get("document_title", lid)
            out += [
                f"## {i}. {title}\n",
                f"**Layer**: `{lid}`  |  **Category**: {info.get('category','?')}\n",
                f"{info.get('description','(no description)')}\n\n",
                f"🔧 **Next steps**: \n",
                f"1. `describe_layer_schema('{lid}')` - Get column names FIRST\n",
                f"2. `intersect_with_town('{lid}', 'TOWN_NAME')` - Find features in a town\n",
                f"3. `query_spatial('{lid}', cql='your_filter')` - Custom queries\n",
                f"4. `get_layer_details('{lid}')` - More metadata\n\n",
            ]
        return [TextContent(type="text", text="".join(out))]

    async def _list_categories(self):
        lines = ["# Categories\n\n"]
        for cat, desc in self.categories.items():
            cnt = sum(1 for inf in self.layer_catalog.values() if inf.get("category") == cat)
            lines.append(f"* **{cat}** – {desc} ({cnt} layers)\n")
        return [TextContent(type="text", text="".join(lines))]

    async def _get_layer_details(self, lid: str):
        info = self.layer_catalog.get(lid)
        if not info:
            return [TextContent(type="text", text=f"Layer '{lid}' not found.")]
        title = info.get("title") or info.get("document_title", lid)
        out = [
            f"# {title}\n\n",
            f"**Technical name**: `{lid}`\n",
            f"**Category**: {info.get('category','?')}\n",
            f"**Geometry**: {info.get('geometry_type','?')}\n\n",
            f"{info.get('description','')}\n\n",
        ]
        keys = info.get("key_fields", [])
        if keys:
            out.append("**Key Fields**: " + ", ".join(keys[:10]) + "\n\n")
        return [TextContent(type="text", text="".join(out))]

    async def _describe_layer_schema(self, lid: str):
        """
        Fetches a layer's schema if not cached, and returns it as inline text.
        """
        # Check cache first
        if lid in self._schema_cache:
            fields = self._schema_cache[lid]
            logger.debug(f"Schema for '{lid}' served from cache")
        else:
            # Fetch schema if not cached
            if lid not in self.layer_catalog:
                return [TextContent(type="text", text=f"Layer '{lid}' not found.")]

            # Build DescribeFeatureType request
            params = {
                "service": "WFS",
                "version": "1.1.0",
                "request": "DescribeFeatureType",
                "typeName": lid if ":" in lid else f"massgis:{lid}",
            }
            session = await self._get_session()
            try:
                r = await session.get(self.endpoints["wfs_base"], params=params)
                r.raise_for_status()
                xml_text = r.text
            except Exception as e:
                return [TextContent(type="text", text=f"WFS DescribeFeatureType error: {e}")]

            # Parse the XML schema
            import xml.etree.ElementTree as ET

            ns = {"xsd": "http://www.w3.org/2001/XMLSchema"}
            fields: list[tuple[str, str]] = []
            try:
                root = ET.fromstring(xml_text)
                elems = root.findall(".//xsd:complexType//xsd:sequence/xsd:element", ns)
                for el in elems:
                    name = el.attrib.get("name")
                    ftype = el.attrib.get("type", "").split(":")[-1]
                    if name:
                        fields.append((name, ftype))
            except Exception as e:
                return [TextContent(type="text", text=f"Failed to parse schema XML: {e}")]

            if not fields:
                return [TextContent(type="text", text="Schema returned no attributes.")]

            # Cache the result in memory for this session
            self._schema_cache[lid] = fields
            logger.info(f"Fetched and cached schema for '{lid}' ({len(fields)} fields)")

        # Format as markdown table
        lines = [
            f"# `{lid}` – attribute schema\n\n",
            "| Field | Type |\n",
            "|-------|------|\n"
        ]
        for name, ftype in fields:
            lines.append(f"| {name} | {ftype} |\n")
        
        # Add helpful usage note
        lines.append(f"\n**Geometry column**: `{await self._get_default_geom(lid)}`\n")
        lines.append(f"**Total fields**: {len(fields)}\n")
        
        return [TextContent(type="text", text="".join(lines))]

    # ------------------------------------------------------------------
    # WFS helpers
    # ------------------------------------------------------------------
    async def _get_session(self):
        if not self.session:
            self.session = httpx.AsyncClient(timeout=30)
        return self.session

    async def _get_default_geom(self, lid: str) -> str:
        if lid in self.geom_col_cache:
            return self.geom_col_cache[lid]

        # In case it's in the layer catalog already
        info = self.layer_catalog.get(lid, {})
        geom_col = info.get("geometry_column")
        if geom_col:
            self.geom_col_cache[lid] = geom_col
            return geom_col

        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "DescribeFeatureType",
            "typeName": lid if ":" in lid else f"massgis:{lid}",
        }
        session = await self._get_session()
        try:
            r = await session.get(self.endpoints["wfs_base"], params=params)
            r.raise_for_status()
            xml_text = r.text
        except Exception as e:
            logger.warning(
                "Failed to get schema for %s, falling back to 'geom': %s", lid, e
            )
            self.geom_col_cache[lid] = "geom"
            return "geom"  # fallback

        import xml.etree.ElementTree as ET

        ns = {"xsd": "http://www.w3.org/2001/XMLSchema"}

        try:
            root = ET.fromstring(xml_text)
            # Find elements that are geometry types
            # gml:MultiSurfacePropertyType, gml:PointPropertyType etc.
            # So the type attribute contains 'gml:' and ends with 'PropertyType'
            elems = root.findall(".//xsd:complexType//xsd:sequence/xsd:element", ns)
            for el in elems:
                el_type = el.attrib.get("type", "")
                if "gml:" in el_type and el_type.endswith("PropertyType"):
                    geom_col_name = el.attrib.get("name")
                    if geom_col_name:
                        logger.debug("Found geom col '%s' for %s", geom_col_name, lid)
                        self.geom_col_cache[lid] = geom_col_name
                        return geom_col_name
        except Exception as e:
            logger.warning(
                "Failed to parse schema for %s, falling back to 'geom': %s", lid, e
            )

        # Default fallback
        logger.debug("No geom col found for %s, falling back to 'geom'", lid)
        self.geom_col_cache[lid] = "geom"
        return "geom"

    async def _query_spatial(
        self,
        lid: str,
        cql: str,
        max_features: int,
        start_index: int,
        sort_by: Optional[str],
        use_notepad: bool = False,
    ):
        if lid not in self.layer_catalog:
            return [TextContent(type="text", text=f"Layer '{lid}' not found.")]
        
        # Track this layer as being used
        if lid not in self.used_layers:
            self.used_layers.append(lid)
            logger.debug(f"Added {lid} to used layers")
        
        params: Dict[str, Any] = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": f"massgis:{lid}",
            "outputFormat": "application/json",
            "maxFeatures": str(max_features),
            "startIndex": str(start_index),
        }
        if ":" in lid:
            params["typeName"] = lid

        if cql:
            params["cql_filter"] = cql
        if sort_by:
            params["sortBy"] = sort_by
        session = await self._get_session()
        try:
            r = await session.get(self.endpoints["wfs_base"], params=params)
            r.raise_for_status()

            # Per user feedback, handle non-JSON responses gracefully
            if "application/json" not in r.headers.get("content-type", "").lower():
                return [
                    TextContent(
                        type="text",
                        text=f"Received non-JSON response (likely a service exception):\n{r.text}",
                    )
                ]

            data = r.json()
            feats = data.get("features", [])

            native_bbox = None
            if "bbox" in data:
                native_bbox = data["bbox"]
            elif feats and "bbox" in feats[0]:
                native_bbox = feats[0]["bbox"]

            if native_bbox and self.last_municipality:
                # The context that sets last_municipality (e.g., intersect_with_town)
                # implies the query was against a layer in EPSG:26986.
                # We transform this native bbox to EPSG:4326 before caching.
                try:
                    minx, miny, maxx, maxy = map(float, native_bbox)
                    tr = Transformer.from_crs(26986, 4326, always_xy=True)
                    minlon, minlat = tr.transform(minx, miny)
                    maxlon, maxlat = tr.transform(maxx, maxy)

                    bbox4326 = f"{minlon:.6f},{minlat:.6f},{maxlon:.6f},{maxlat:.6f}"
                    self._bbox_by_municipality[self.last_municipality] = bbox4326
                    logger.debug(
                        f"Cached EPSG:4326 bbox for {self.last_municipality}: {bbox4326}"
                    )
                except Exception as e:
                    logger.warning(f"Could not reproject or cache bbox: {e}")

        except httpx.HTTPStatusError as e:
            # If the server returned an error, its response body might contain a useful message
            return [
                TextContent(
                    type="text",
                    text=f"WFS HTTP error: {e}\nServer response:\n{e.response.text}",
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"WFS query failed: {e}")]
        
        if not feats:
            return [TextContent(type="text", text="No features returned.")]
        
        # Handle large result sets
        if len(feats) > 20 and not use_notepad:
            # Provide information about the large result set
            title = self.layer_catalog[lid].get("title") or self.layer_catalog[lid].get(
                "document_title", lid
            )
            return [TextContent(
                type="text",
                text=(
                    f"⚠️ **Large Result Set: {len(feats)} features found**\n\n"
                    f"This query returned many results from '{title}'.\n\n"
                    f"**Options:**\n"
                    f"1. View a sample by running the query again with `max_features=10`\n"
                    f"2. Save all results to notepad by running with `use_notepad=True`\n"
                    f"3. Refine your query to return fewer results\n\n"
                    f"If you want to save all {len(feats)} features for export, "
                    f"re-run the query with `use_notepad=True`."
                )
            )]
        
        # If using notepad, record all features
        if use_notepad:
            title = self.layer_catalog[lid].get("title") or self.layer_catalog[lid].get(
                "document_title", lid
            )
            notepad_entry = {
                "timestamp": datetime.now().isoformat(),
                "layer": lid,
                "title": title,
                "query": cql,
                "feature_count": len(feats),
                "features": []
            }
            
            for f in feats:
                notepad_entry["features"].append(f["properties"])
            
            self.notepad["entries"].append(notepad_entry)
            self.notepad["metadata"]["last_modified"] = datetime.now().isoformat()
            self.notepad["metadata"]["entry_count"] = len(self.notepad["entries"])
            
            return [TextContent(
                type="text",
                text=(
                    f"✅ **Saved {len(feats)} features to notepad**\n\n"
                    f"**Layer**: {title}\n"
                    f"**Query**: `{cql}`\n\n"
                    f"Use `notepad_read()` to view summary or `notepad_export()` to download as CSV/JSON."
                )
            )]
        
        # Normal display for reasonable result sets
        sample = feats[0]["properties"]
        keys = list(sample)[:8]
        title = self.layer_catalog[lid].get("title") or self.layer_catalog[lid].get(
            "document_title", lid
        )
        out = [
            f"# {title} – {len(feats)} feature(s)\n\n",
            "**Fields shown**: " + ", ".join(keys) + "\n\n",
        ]
        for i, f in enumerate(feats[: min(10, len(feats))], 1):
            out.append(f"### Feature {i}\n")
            for k in keys:
                out.append(f"* {k}: {f['properties'].get(k)}\n")
            out.append("\n")
        if len(feats) > 10:
            out.append(
                f"… plus {len(feats)-10} more (increase 'max_features' to fetch).\n"
            )
        return [TextContent(type="text", text="".join(out))]

    async def _get_native_srid(self, lid: str) -> str:
        """
        Parse WFS -> GetCapabilities once and cache the first <DefaultCRS>
        (or fallback to EPSG:26986 which is MassGIS' default).
        """
        if lid in self._srid_cache:
            return self._srid_cache[lid]

        caps_url = (
            f"{self.endpoints['wfs_base']}?service=WFS&version=1.1.0"
            f"&request=GetCapabilities"
        )
        session = await self._get_session()
        try:
            r = await session.get(caps_url)
            r.raise_for_status()
            txt = r.text
        except Exception as e:
            logger.warning("GetCapabilities failed for SRID lookup, falling back: %s", e)
            self._srid_cache[lid] = "26986"
            return "26986"

        # very light-weight parse – no external libs
        import html

        pat = rf"<Name>{re.escape(lid)}</Name>.*?<DefaultCRS>(.*?)</DefaultCRS>"
        m = re.search(pat, txt, flags=re.S)
        srid = m.group(1).split("::")[-1] if m else "26986"
        self._srid_cache[lid] = srid
        return srid

    async def _intersect_with_town(
        self, layer_name: str, municipality: str, max_features: int, use_notepad: bool = False
    ):
        """Return <layer_name> features whose geometry intersects the named municipality."""
        # Track this layer and municipality
        if layer_name not in self.used_layers:
            self.used_layers.append(layer_name)
        self.last_municipality = municipality.upper()
        
        geom_col = await self._get_default_geom(layer_name)
        native_srid = (
            await self._get_native_srid(layer_name)
        )  # "26986", "4326", …

        # Use doubled single quotes for proper CQL escaping
        town_filter = f'"town" = \'\'{municipality.upper()}\'\''
        sub_query = (
            f"collectGeometries("
            f"queryCollection('massgis:GISDATA.TOWNSSURVEY_POLYM','shape','{town_filter}')"
            f")"
        )

        # only reproject if needed
        if native_srid != "26986":
            sub_query = f"reproject({sub_query},'EPSG:26986','EPSG:{native_srid}')"

        cql = f'INTERSECTS("{geom_col}", {sub_query})'
        
        logger.debug(f"intersect_with_town CQL: {cql}")

        return await self._query_spatial(layer_name, cql, max_features, 0, None, use_notepad)

    async def _find_nearby(self, lid: str, lat: float, lon: float, radius_meters: int, use_notepad: bool = False):
        # Track this layer
        if lid not in self.used_layers:
            self.used_layers.append(lid)
            
        geom_col = await self._get_default_geom(lid)
        srid = await self._get_native_srid(lid)  # e.g. "26986"
        
        # transform lon/lat (EPSG:4326) → native CRS metres
        tr = Transformer.from_crs(4326, int(srid), always_xy=True)
        x, y = tr.transform(lon, lat)
        
        # Create plain WKT without SRID prefix
        point_wkt = f"POINT({x} {y})"
        
        # Use geomFromWKT to parse the WKT string
        cql = (
            f'DWITHIN("{geom_col}", '
            f"geomFromWKT('{point_wkt}'), "
            f"{radius_meters}, meters)"
        )
        
        res = await self._query_spatial(lid, cql, 20, 0, None, use_notepad)
        if res and isinstance(res[0], TextContent) and "No features" not in res[0].text:
            res[0].text = (
                f"# Features within {radius_meters} m of ({lat:.4f}, {lon:.4f})\n\n"
                + res[0].text
            )
        return res

    def _union_bboxes(self, boxes: list[str]) -> str:
        """
        boxes: ["minx,miny,maxx,maxy", ...]
        returns the smallest box containing them all
        """
        mins_x, mins_y, maxs_x, maxs_y = [], [], [], []
        for b in boxes:
            try:
                minx, miny, maxx, maxy = map(float, b.split(","))
                mins_x.append(minx)
                mins_y.append(miny)
                maxs_x.append(maxx)
                maxs_y.append(maxy)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse bbox '{b}': {e}")
                continue
        if not mins_x:
            return ""  # Or a default
        return f"{min(mins_x)},{min(mins_y)},{max(maxs_x)},{max(maxs_y)}"
        
    # ------------------------------------------------------------------
    # MassMapper link generation
    # ------------------------------------------------------------------
    async def _generate_massmapper_link(
        self,
        municipalities: Optional[List[str]],
        use_union_bbox: bool,
        bbox: Optional[str],
        include_all_used: bool,
        specific_layers: Optional[List[str]],
    ):
        """Generate a MassMapper web application link with previously used layers."""

        # 1. Determine which layers to include
        if specific_layers:
            layers_to_include = specific_layers
        elif include_all_used:
            layers_to_include = self.used_layers.copy()
        else:
            layers_to_include = []

        if not layers_to_include:
            return [
                TextContent(
                    type="text",
                    text="No layers have been queried yet. Please run some spatial queries first.",
                )
            ]

        # 2. Determine which municipalities and bounding boxes to use
        target_towns = []
        if municipalities:
            target_towns = [m.upper() for m in municipalities]
        elif include_all_used and self._bbox_by_municipality:
            target_towns = list(self._bbox_by_municipality.keys())
        elif self.last_municipality:
            target_towns = [self.last_municipality.upper()]
        
        # 3. Collect bounding boxes for the target towns
        bboxes_to_process = []
        if bbox: # Manual override
             bboxes_to_process.append(bbox)
        else:
            for town in target_towns:
                if town in self._bbox_by_municipality:
                    bboxes_to_process.append(self._bbox_by_municipality[town])
                else:
                    # Fetch and cache the bbox if we don't have it
                    fetched_bbox = await self._get_municipality_bbox(town)
                    bboxes_to_process.append(fetched_bbox)

        # 4. Generate the final report
        base_url = "https://maps.massgis.digital.mass.gov/MassMapper/MassMapper.html"
        layer_params = self._format_layers_for_massmapper(layers_to_include)
        
        # Handle different bbox scenarios
        if not bboxes_to_process:
            final_bbox = "-71.3,42.2,-70.9,42.5" # Fallback
            final_towns_str = "Greater Boston (Default)"
            report_text = self._build_massmapper_report(base_url, layer_params, layers_to_include, final_bbox, final_towns_str)
        elif len(bboxes_to_process) > 1 and not use_union_bbox:
            # Generate a report with multiple links, one for each town
            report_text = self._build_multi_link_report(base_url, layer_params, layers_to_include, target_towns)
        else:
            # Generate a single link, unioning bboxes if necessary
            final_bbox = self._union_bboxes(bboxes_to_process) if len(bboxes_to_process) > 1 else bboxes_to_process[0]
            final_towns_str = ", ".join(target_towns) if target_towns else "Custom Area"
            report_text = self._build_massmapper_report(base_url, layer_params, layers_to_include, final_bbox, final_towns_str)

        return [TextContent(type="text", text=report_text)]

    def _format_layers_for_massmapper(self, layers: List[str]) -> str:
        # Special layer mappings
        layer_mappings = {
            "GISDATA.L3_TAXPAR_POLY_ASSESS": "Basemaps_L3Parcels____ON__100",
            "GISDATA.TOWNSSURVEY_POLYM": "massgis:GISDATA.TOWNSSURVEY_POLYM__GISDATA.TOWNSSURVEY_POLYM::Default__ON__100",
        }
        formatted_layers = []
        for layer in layers:
            clean_layer = layer.replace("massgis:", "")
            if clean_layer in layer_mappings:
                formatted_layers.append(layer_mappings[clean_layer])
            else:
                formatted_layers.append(f"massgis:{clean_layer}__{clean_layer}::Default__ON__100")
        return ",".join(formatted_layers)

    def _build_massmapper_report(self, base_url: str, layer_params: str, layers_to_include: list, bbox: str, area_name: str) -> str:
        params = {"bl": "MassGIS+Basemap__100", "l": layer_params, "b": bbox}
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{base_url}?{query_string}"
        
        layer_list_md = []
        for layer in layers_to_include:
            info = self.layer_catalog.get(layer, {})
            title = info.get("title") or info.get("document_title", layer)
            layer_list_md.append(f"• {title} (`{layer}`)")
        
        return "\n".join([
            "# 🗺️ MassMapper Visualization Link",
            f"\n**Layers included** ({len(layers_to_include)}):",
            "\n".join(layer_list_md),
            f"\n**Area**: {area_name}",
            f"**Bounding box**: `{bbox}`\n",
            f"## 🔗 [Open Interactive Map in MassMapper]({full_url})\n",
            "Click the link above to view your analyzed layers. You can toggle layers, view details, export data, and more.",
            f"\nDirect URL: `{full_url}`"
        ])

    def _build_multi_link_report(self, base_url: str, layer_params: str, layers_to_include: list, towns: list[str]) -> str:
        report_lines = [
            "# 🗺️ MassMapper Visualization Links",
            "\nHere are individual map links for each requested municipality:"
        ]
        
        layer_list_md = []
        for layer in layers_to_include:
            info = self.layer_catalog.get(layer, {})
            title = info.get("title") or info.get("document_title", layer)
            layer_list_md.append(f"• {title} (`{layer}`)")
        
        report_lines.append(f"\n**Layers included** ({len(layers_to_include)}):")
        report_lines.append("\n".join(layer_list_md))
        report_lines.append("\n---")

        for town in towns:
            bbox = self._bbox_by_municipality.get(town, "")
            if not bbox:
                continue
            params = {"bl": "MassGIS+Basemap__100", "l": layer_params, "b": bbox}
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{base_url}?{query_string}"
            report_lines.append(f"### 📍 {town.title()}")
            report_lines.append(f"🔗 [Open Map for {town.title()}]({full_url})")

        return "\n".join(report_lines)

    async def _get_municipality_bbox(self, municipality: str) -> str:
        """Get bounding box for a municipality in format 'minx,miny,maxx,maxy' (EPSG:4326)"""
        layer = "massgis:GISDATA.TOWNSSURVEY_POLYM"
        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": layer,
            "outputFormat": "application/json",
            "cql_filter": f'upper("town") = \'{municipality.upper()}\'',
            "maxFeatures": "1",
        }

        session = await self._get_session()
        try:
            r = await session.get(self.endpoints["wfs_base"], params=params)
            r.raise_for_status()
            data = r.json()

            native_bbox = None
            if data.get("features"):
                feature = data["features"][0]
                if "bbox" in data:
                    native_bbox = data["bbox"]
                elif "bbox" in feature:
                    native_bbox = feature["bbox"]
                elif "geometry" in feature and feature["geometry"]:
                    geom = feature["geometry"]
                    if geom["type"] in ["Polygon", "MultiPolygon"]:
                        coords = []
                        if geom["type"] == "Polygon":
                            for ring in geom["coordinates"]:
                                coords.extend(ring)
                        else:  # MultiPolygon
                            for polygon in geom["coordinates"]:
                                for ring in polygon:
                                    coords.extend(ring)
                        if coords:
                            xs = [c[0] for c in coords]
                            ys = [c[1] for c in coords]
                            native_bbox = [min(xs), min(ys), max(xs), max(ys)]

            if native_bbox:
                minx, miny, maxx, maxy = map(float, native_bbox)
                # The towns layer is in EPSG:26986, so we reproject to 4326 for MassMapper
                tr = Transformer.from_crs(26986, 4326, always_xy=True)
                minlon, minlat = tr.transform(minx, miny)
                maxlon, maxlat = tr.transform(maxx, maxy)

                # Add a small buffer in degree space for better map framing
                buffer = 0.01
                bbox4326 = f"{minlon - buffer:.6f},{minlat - buffer:.6f},{maxlon + buffer:.6f},{maxlat + buffer:.6f}"

                self._bbox_by_municipality[municipality.upper()] = bbox4326
                logger.info(f"Fetched and cached EPSG:4326 bbox for {municipality}")
                return bbox4326

        except Exception as e:
            logger.error(f"Failed to get bbox for {municipality}: {e}", exc_info=True)

        # Default fallback with warning
        logger.warning(f"Using default EPSG:4326 bbox for {municipality}")
        bbox_str = "-71.3,42.2,-70.9,42.5"  # This is already 4326
        self._bbox_by_municipality[municipality.upper()] = bbox_str
        return bbox_str

    # ------------------------------------------------------------------
    # Notepad methods
    # ------------------------------------------------------------------
    async def _notepad_write(
        self,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Write arbitrary content to the notepad."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "title": title or "Manual Entry",
            "content": content,
            "metadata": metadata or {}
        }
        
        self.notepad["entries"].append(entry)
        self.notepad["metadata"]["last_modified"] = datetime.now().isoformat()
        self.notepad["metadata"]["entry_count"] = len(self.notepad["entries"])
        
        return [TextContent(
            type="text",
            text=f"✅ Added entry '{title or 'Manual Entry'}' to notepad. Total entries: {len(self.notepad['entries'])}"
        )]

    async def _notepad_read(
        self,
        last_n: Optional[int] = None,
        summary_only: bool = False
    ):
        """Read notepad contents."""
        entries = self.notepad["entries"]
        
        if not entries:
            return [TextContent(type="text", text="📝 Notepad is empty.")]
        
        if summary_only:
            out = ["# Notepad Summary\n\n"]
            out.append(f"**Total entries**: {len(entries)}\n")
            out.append(f"**Created**: {self.notepad['metadata']['created']}\n")
            out.append(f"**Last modified**: {self.notepad['metadata']['last_modified']}\n\n")
            
            for i, entry in enumerate(entries, 1):
                if "layer" in entry:
                    out.append(f"{i}. **{entry['title']}** ({entry['layer']}) - {entry['feature_count']} features\n")
                else:
                    out.append(f"{i}. **{entry['title']}**\n")
            
            return [TextContent(type="text", text="".join(out))]
        
        # Full read
        entries_to_show = entries if last_n is None else entries[-last_n:]
        out = [f"# Notepad Contents ({len(entries_to_show)} of {len(entries)} entries)\n\n"]
        
        for i, entry in enumerate(entries_to_show, 1):
            out.append(f"## Entry {i}: {entry.get('title', 'Untitled')}\n")
            out.append(f"**Timestamp**: {entry['timestamp']}\n")
            
            if "layer" in entry:
                # This is a spatial query result
                out.append(f"**Layer**: {entry['layer']}\n")
                out.append(f"**Query**: {entry.get('query', 'N/A')}\n")
                out.append(f"**Features**: {entry['feature_count']}\n")
                
                if entry['feature_count'] > 0 and entry['features']:
                    # Show first few features as sample
                    out.append("\n### Sample features:\n")
                    for j, feat in enumerate(entry['features'][:3], 1):
                        out.append(f"#### Feature {j}:\n")
                        for k, v in list(feat.items())[:5]:
                            out.append(f"* {k}: {v}\n")
                        out.append("\n")
                    if entry['feature_count'] > 3:
                        out.append(f"... and {entry['feature_count'] - 3} more features\n")
            else:
                # Regular text entry
                out.append(f"\n{entry.get('content', '')}\n")
            
            out.append("\n---\n\n")
        
        return [TextContent(type="text", text="".join(out))]

    async def _notepad_clear(self, confirm: bool = False):
        """Clear all notepad entries."""
        if not confirm:
            return [TextContent(
                type="text",
                text="⚠️ Set confirm=True to clear notepad. This will delete all entries!"
            )]
        
        entry_count = len(self.notepad["entries"])
        self.notepad["entries"] = []
        self.notepad["metadata"]["last_modified"] = datetime.now().isoformat()
        self.notepad["metadata"]["entry_count"] = 0
        
        return [TextContent(
            type="text",
            text=f"🗑️ Cleared {entry_count} entries from notepad."
        )]

    async def _notepad_export(
        self,
        format: str = "csv",
        filename_prefix: str = "massgis_export"
    ):
        """Export notepad contents directly to file system."""
        if not self.notepad["entries"]:
            return [TextContent(type="text", text="📝 Notepad is empty, nothing to export.")]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            # Export as JSON
            filename = f"{filename_prefix}_{timestamp}.json"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.notepad, f, indent=2)
            
            return [TextContent(
                type="text",
                text=(
                    f"✅ **JSON Export Saved**\n\n"
                    f"**File**: `{filename}`\n"
                    f"**Location**: `{str(filepath)}`\n"
                    f"**Entries**: {len(self.notepad['entries'])}\n\n"
                    f"The file has been saved to your exports folder."
                )
            )]
        
        else:  # CSV format
            # Collect all unique fields across all features
            all_fields = set()
            csv_data = []
            
            for entry in self.notepad["entries"]:
                if "features" in entry and entry["features"]:
                    # This is a spatial query result
                    for feat in entry["features"]:
                        all_fields.update(feat.keys())
                        # Add metadata fields
                        feat_with_meta = {
                            "_layer": entry["layer"],
                            "_query": entry.get("query", ""),
                            "_timestamp": entry["timestamp"],
                            **feat
                        }
                        csv_data.append(feat_with_meta)
                else:
                    # Regular text entry - add as a single row
                    csv_data.append({
                        "_type": "note",
                        "_timestamp": entry["timestamp"],
                        "_title": entry.get("title", ""),
                        "_content": entry.get("content", "")
                    })
            
            if not csv_data:
                return [TextContent(type="text", text="📝 No data to export as CSV.")]
            
            # Write CSV directly to file
            filename = f"{filename_prefix}_{timestamp}.csv"
            filepath = self.export_dir / filename
            
            # Get all fields including metadata fields
            fieldnames = sorted(list(all_fields | {k for row in csv_data for k in row.keys()}))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            return [TextContent(
                type="text",
                text=(
                    f"✅ **CSV Export Saved**\n\n"
                    f"**File**: `{filename}`\n"
                    f"**Location**: `{str(filepath)}`\n"
                    f"**Rows**: {len(csv_data)}\n"
                    f"**Columns**: {len(fieldnames)}\n\n"
                    f"The file has been saved to your exports folder:\n"
                    f"`{self.export_dir}`"
                )
            )]

    async def cleanup(self):
        if self.session:
            await self.session.aclose()


# ---------------------------------------------------------------------
async def main():
    srv = MassGISCatalogServer()
    try:
        async with mcp.server.stdio.stdio_server() as (reader, writer):
            caps = ServerCapabilities(tools={"listChanged": False})
            await srv.server.run(
                reader,
                writer,
                InitializationOptions(
                    server_name="massgis-vector",
                    server_version="1.1.0",
                    capabilities=caps,          # prompts & resources are auto-detected
                ),
            )
    finally:
        await srv.cleanup()


if __name__ == "__main__":
    asyncio.run(main())