
Metadata YML File Format
=======================

This document describes the format for the metadata YML file used to describe data sources and notes for fieldwork sessions. These are stored in each session folder.  

YML Structure
-------------

The metadata YML file should contain the following fields:

- `notes`: (string, optional) Freeform notes about the session, site, or observations. These are recorded by the researchers on site.
- `data_sources`: (list, required) A list of data source entries, each describing a file used in the session. The thermal folders should be populated with a single .csq file, and the rgb folders with an .MP4

Each entry in `data_sources` should be a dictionary with the following fields:

  - `description`: (string, required) A short description of the data source (e.g., type of camera, sensor, etc.). Camera 1 (thermal or rgb) refers to the leftmost camera in the field. This may sometimes not align with the actual camera number (e.g., thermal_1 may be FLIR2).
  - `original_path`: (string, required) The original location of the file (e.g., on a local drive or cloud storage). Useful for provenance.
  - `path`: (string, required) The relative or final path to the file within the project or data repository. Based on the project structure, this should be a path to a thermal_1, thermal_2, rgb_1, or rgb_2 directory, assuming 2 camera set up.

Example
-------

.. code-block:: yaml

   notes: Marshlands conservancy. Seed feeders depleted, not much action, though I saw a titmouse check a feeder.

   data_sources:
     - description: raw data for thermal camera 1
       original_path: /Google Drive/fieldwork_to_sync/fieldwork_data/2024-02-06/FLIR2/20240206071808444.csq
       path: thermal_1/20240206071808444.csq
     - description: raw data for thermal camera 2
       original_path: /Google Drive/fieldwork_to_sync/fieldwork_data/2024-02-06/FLIR1/20240206071804298.csq
       path: thermal_2/20240206071804298.csq
     - description: raw data for rgb camera 1
       original_path: /Google Drive/fieldwork_to_sync/fieldwork_data/2024-02-06/GOPROB/GX010119.MP4
       path: rgb_1/GX010119.MP4
     - description: raw data for rgb camera 2 
       original_path: /Google Drive/fieldwork_to_sync/fieldwork_data/2024-02-06/GOPROA/GX010119.MP4
       path: rgb_2/GX010119.MP4

Notes
-----
- All paths should use forward slashes (`/`).

