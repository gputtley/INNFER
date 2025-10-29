---
layout: page
title: "Step: LoadData"
---

This step is typically the first step when running **INNFER** (assuming you are not working from a benchmark). It creates the 
base datasets, defined using the "files" input in the configuration file, from which all variations (both weight and feature changing) are calculated later. 

It loads in the input datasets and adds the extra columns specified in the configuration file. There are additional options to add summed columns and remove negative weights. It will also reduce the number of stored columns to the minimum, so that the next PreProcess steps runs more efficiently. It will output one parquet dataset per "files" key in the configuration file.