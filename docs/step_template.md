---
layout: page
title: "Adding and documenting a step"
---

A step is a runner class dispatched by `scripts/innfer.py`. Its input/output declarations are also used to construct the Snakemake graph. `python/runner/template.py` provides the basic interface.

## Runner contract

```python
class Example:
    def __init__(self):
        self.data_input = None
        self.data_output = None

    def Configure(self, options):
        for key, value in options.items():
            setattr(self, key, value)

    def Run(self):
        # Read the configured inputs and create every declared output.
        pass

    def Inputs(self):
        return [self.data_input]

    def Outputs(self):
        return [self.data_output]
```

Use the existing module's naming, indentation and configuration conventions when implementing a concrete runner. Inputs/Outputs must work during workflow generation before upstream producers run. Do not read an unavailable upstream fit simply to decide its dependency path; load its values during Run instead.

Declare the source architecture, metadata, weights, datasets and any external files required at runtime. Outputs must match the names written by Run and identify unique artifacts for each independent loop iteration. Intermediate shards are not completion outputs.

Add a dispatch branch with `module.Run`, configuration and meaningful loop keys. Add parser options only when required, and describe the step in `configs/other/step_description.yaml` so CLI descriptions remain in sync.

## Documentation page

Create `docs/{lowercase_step_name}.md` with this structure:

```markdown
---
layout: page
title: "Step: Example"
---

Explain the result and when this step is used.

## Run

Provide a command with the required options and mark example placeholders.

## Inputs and outputs

Name prerequisite producer steps and the artifacts read/written.

## Step options

Describe relevant CLI defaults, separately from architecture/configuration keys.

## Behaviour and checks

Describe important selection, weighting and indexing semantics.

## Implementation

Link the dispatch and runner source, then link back to steps.md.
```

Add the page to [steps.md](steps.md). Use relative links, fenced commands and blank lines around lists/tables. Verify that every documented selector exists in the dispatch and every referenced option is parsed. New docs should describe actual behavior rather than promise an unimplemented algorithm.

## Custom runners

[Custom](custom.md) uses the supplied name as both module and class name. The current dispatcher splits `--custom-options` entries on a colon, despite the parser help describing equals signs:

```bash
innfer --cfg="configs/run/your_analysis.py" --step="Custom" \
  --custom-module="Example" --custom-options="name:value;other:value"
```

Custom runners declare their own inputs/outputs and receive the run configuration path plus the parsed options dictionary.
