# MassGIS MCP Server

This repository contains a Model Context Protocol (MCP) server that enables AI assistants to interface with Massachusetts statewide data layers (MassGIS) to answer complex spatial queries. The server is built using the MCP framework (created by Anthropic) and provides a comprehensive set of tools for searching, querying, and analyzing geospatial data relevant to environmental assessment and planning in Massachusetts.

## Overview

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between AI assistants and external data sources or tools. This implementation, `massgis_mcp.py`, creates an MCP server that exposes a suite of tools for interacting with the MassGIS Web Feature Service (WFS).

The server allows AI assistants (like Claude) to perform tasks such as:

- Searching for available data layers based on keywords
- Inspecting the data schema of specific layers to understand their attributes
- Performing complex spatial queries using ECQL (Extended Common Query Language)
- Finding features within specific municipal boundaries
- Generating interactive map links to visualize results in MassMapper
- Using a "notepad" system to store, manage, and export large result sets from queries

The MCP server is designed to enable interactive geospatial analysis, where users can chain commands together through their AI assistant. For example, a user might first search for "parks" layers, then inspect the schema of a chosen layer, and finally perform a spatial query to find all parks within a specific town.

## Getting Started

Follow these instructions to run the MassGIS MCP server and connect it to an MCP client like Claude Desktop.

### 1. Prerequisites

- [Python 3.8](https://www.python.org/downloads/) or newer must be installed on your system.

### 2. Installation

First, clone this repository to your local machine and install the required Python dependencies. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/mmulqu/MassMapper-MCP.git
cd MassMapper-MCP

# Create and activate a virtual environment
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Configure Your MCP Client

To connect this server to your MCP client (e.g., Claude Desktop, Cursor), you need to add its configuration to your client's settings file. The configuration tells your client how to launch the MassGIS server.

Add the following JSON object to your client's `mcpServers` configuration. **Remember to replace `/path/to/...` with the absolute paths on your machine.**

```json
{
  "mcpServers": {
    "massgis": {
      "command": "/path/to/your/venv/bin/python",
      "args": [
        "/path/to/your/cloned/repo/massmapper_MCP/massgis_mcp.py"
      ]
    }
  }
}
```

-   `"command"`: The absolute path to the Python executable inside the virtual environment you created.
    -   On macOS/Linux, this will be `/path/to/massmapper_MCP/venv/bin/python`.
    -   On Windows, this will be `C:\path\to\massmapper_MCP\venv\Scripts\python.exe`.
-   `"args"`: The absolute path to the `massgis_mcp.py` script within the cloned repository.

Once configured, restart your MCP client. The MassGIS tools should now be available.

## Available Tools

The MCP server exposes the following tools through the Model Context Protocol:

| Tool | Description |
|------|-------------|
| `search_layers` | Searches for MassGIS data layers using keywords and optional categories |
| `list_categories` | Lists all available data categories to help narrow down searches |
| `get_layer_details` | Retrieves detailed metadata and a summary for a specific data layer |
| `describe_layer_schema` | **CRITICAL**: Displays the exact column names and data types for a layer's attributes. Essential for building correct queries |
| `query_spatial` | The primary tool for executing complex spatial queries (e.g., INTERSECTS, DWITHIN) using ECQL filters |
| `intersect_with_town` | A simplified tool for finding features from a layer that are located within a specific Massachusetts town |
| `find_nearby` | A last-resort tool to find features within a given radius of a specific latitude/longitude point |
| `massmapper_link` | **VISUALIZATION**: Generates a MassMapper URL to create an interactive map showing the layers and features from the analysis |
| `notepad_write` | Appends text or data (like query results) to an internal notepad |
| `notepad_read` | Displays the contents of the notepad |
| `notepad_clear` | Clears all entries from the notepad |
| `notepad_export` | Exports the collected notepad entries into a file (e.g., CSV or JSON) |

## Working with Large Datasets: The Notepad Workflow

A key feature of this MCP server is its ability to handle queries that return hundreds or thousands of results—far more than can fit in an AI assistant's context window. The "notepad" system is designed specifically for this purpose. Here's how to work with large datasets effectively.

The AI assistant can only see a small sample of results from a large query. It **cannot** search or iterate through the full dataset stored in the server's notepad. The `notepad_export` function is the bridge for this limitation.

### Recommended Workflow

1.  **Perform a Broad Query & Save to Notepad**: When performing a query that you expect to return many results, set `use_notepad=True`. This saves the complete result set on the server.
    ```
    # Find all bike trails within 500m of any commuter rail station
    query_spatial(
        layer_name="GISDATA.BIKETRAILS_ARC",
        cql="DWITHIN(shape, collectGeometries(queryCollection('GISDATA.MBTA_NODE', 'shape', \"LINE='COMMUTER RAIL'\")), 500, meters)",
        use_notepad=True
    )
    ```
    The assistant will only see a few sample results, but all results are now stored server-side.

2.  **Export the Full Dataset**: Use the `notepad_export` tool to write the entire stored result set to a CSV file.
    ```
    notepad_export(format="csv", filename_prefix="bike_trails_near_stations")
    ```
    This creates a file in an `exports/` directory within the project folder.

3.  **Analyze the Data Externally**: As the user, you now have the complete dataset. Open the exported CSV file in a tool like Excel, Google Sheets, or a GIS application. Here, you can sort, filter, and analyze the data to identify specific features of interest.

4.  **Ask Targeted Follow-up Questions**: Once you've identified specific items (e.g., a specific parcel ID, a trail name), you can ask the AI assistant for more details about them.
    > "I see a trail named 'Minuteman Bikeway' in the exported CSV. Can you show me what schools are within 250 meters of that specific trail?"

5.  **The Assistant Performs New, Targeted Queries**: The assistant will now perform a *fresh query* based on your specific request. It is not searching the notepad; it is querying the live data source with the new, precise information you provided.

This workflow leverages the strengths of both systems: the MCP server's ability for bulk data extraction and precise querying, and your ability to analyze large datasets with familiar tools.

## Example Usage

Here are examples of how users can interact with MassGIS data through an AI assistant using this MCP server:

### Example 1: Find all public libraries in the town of "CAMBRIDGE"

1. Search for relevant layers:
   ```
   search_layers(query="libraries")
   ```

2. The search reveals a layer with library locations. Now, find libraries in Cambridge:
   ```
   intersect_with_town(layer_name="GISDATA.LIBRARIES_PT", municipality="CAMBRIDGE")
   ```

3. Visualize the results:
   ```
   massmapper_link(municipalities=["CAMBRIDGE"], specific_layers=["GISDATA.LIBRARIES_PT"])
   ```

### Example 2: Find all conservation areas within 1 kilometer of schools in "LEXINGTON"

1. First, search for conservation area layers:
   ```
   search_layers(query="conservation open space")
   ```
   This finds the layer `GISDATA.OPENSPACE_POLY`.

2. Next, search for schools:
   ```
   search_layers(query='schools')
   ```
   This finds the layer `GISDATA.SCHOOLS_PT`.

3. Run `describe_layer_schema` on both layers to get the geometry column names.

4. Find conservation areas within 1 kilometer of any school in Lexington:
   ```
   query_spatial(
       layer_name="GISDATA.OPENSPACE_POLY",
       cql="DWITHIN(shape, collectGeometries(queryCollection('GISDATA.SCHOOLS_PT', 'shape', \"town='LEXINGTON'\")), 1000, meters)"
   )
   ```

5. Visualize the result:
   ```
   massmapper_link(
       municipalities=["LEXINGTON"], 
       specific_layers=["GISDATA.OPENSPACE_POLY", "GISDATA.SCHOOLS_PT"]
   )
   ```

### Example 3: Find bike trails near commuter rail stations

1. Search for bike trail layers:
   ```
   search_layers(query="bike trails")
   ```

2. Search for commuter rail stations:
   ```
   search_layers(query="commuter rail mbta")
   ```

3. Find all bike trails within 500 meters of commuter rail stations:
   ```
   query_spatial(
       layer_name="GISDATA.BIKETRAILS_ARC",
       cql="DWITHIN(shape, collectGeometries(queryCollection('GISDATA.MBTA_NODE', 'shape', \"LINE='COMMUTER RAIL'\")), 500, meters)",
       use_notepad=True
   )
   ```

4. Export the results for further analysis:
   ```
   notepad_export(format="csv", filename_prefix="bike_trails_near_stations")
   ```

## Integration with AI Assistants

This MCP server is designed to work with AI assistants that support the Model Context Protocol. When connected, the AI assistant gains the ability to perform sophisticated geospatial analysis on Massachusetts data, making it a powerful tool for urban planners, researchers, educators, and anyone interested in exploring Massachusetts geographic data.

The Model Context Protocol ensures that the AI assistant can discover available tools, understand their parameters, and use them effectively to answer complex spatial questions about Massachusetts geography, infrastructure, recreation areas, and community resources. 