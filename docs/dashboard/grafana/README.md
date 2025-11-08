# Grafana Integration

This directory contains all Grafana-related documentation and dashboard templates for visualizing tracking data.

## Contents

- **[grafana_integration.md](grafana_integration.md)** - Complete setup and usage guide
  - PostgreSQL setup and configuration
  - Grafana installation and data source setup
  - Dashboard creation and panel configuration
  - Troubleshooting common issues

- **[grafana_queries.md](grafana_queries.md)** - Comprehensive SQL query library
  - 30+ tested SQL queries for all visualizations
  - Time-series queries (velocity, speed, distances)
  - Spatial queries (heatmaps, histograms)
  - Statistical queries (aggregations, correlations)

- **[grafana_dashboard_template.json](grafana_dashboard_template.json)** - Full-featured dashboard template
  - 10+ panels covering all visualization types
  - Time-series analysis
  - Spatial heatmaps
  - Statistical summaries

- **[grafana_template_simple.json](grafana_template_simple.json)** - Simplified dashboard template
  - Basic visualization panels
  - Good starting point for customization

## Quick Start

1. **Setup PostgreSQL and Grafana** - See [grafana_integration.md](grafana_integration.md#setup)
2. **Import Dashboard** - Use one of the JSON templates
3. **Browse Queries** - See [grafana_queries.md](grafana_queries.md) for all available queries

## Related Documentation

- [Database Schema](../../data/db/README.md) - Database design and structure
- [Spatial Analysis Dashboard](../spatial_analysis.md) - Panel-based dashboard for spatial analysis
