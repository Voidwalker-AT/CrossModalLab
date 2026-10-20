"""CrossModalLab — JSON Schema Validator for Experiment Artifact Schemas.

Validates that all JSON schemas documented in docs/artifact_schemas.md are syntactically
valid Draft 2020-12 schemas, and that the illustrative synthetic demo record adheres
strictly to Schema 1 (RunMetadata).
"""

import sys
import re
import json
from pathlib import Path

def validate_schemas():
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
    except ImportError:
        print("ERROR: jsonschema package is not installed.")
        sys.exit(1)

    schemas_file = Path("docs/artifact_schemas.md")
    if not schemas_file.exists():
        print(f"ERROR: {schemas_file} does not exist.")
        sys.exit(1)

    content = schemas_file.read_text(encoding="utf-8")

    # Extract json blocks
    json_blocks = re.findall(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    print(f"Discovered {len(json_blocks)} JSON blocks in {schemas_file}")

    validated_schemas = {}
    mock_records = []

    for idx, block in enumerate(json_blocks, 1):
        try:
            data = json.loads(block)
        except json.JSONDecodeError as e:
            print(f"[-] Block {idx}: JSON Decode Error: {e}")
            continue

        if isinstance(data, dict) and "$schema" in data:
            schema_title = data.get("title", f"Schema_{idx}")
            try:
                Draft202012Validator.check_schema(data)
                validated_schemas[schema_title] = data
                print(f"[+] Schema '{schema_title}' is valid Draft 2020-12 schema.")
            except Exception as e:
                print(f"[-] Schema '{schema_title}' FAILED validation: {e}")
                sys.exit(1)
        elif isinstance(data, dict) and "run_id" in data:
            mock_records.append((idx, data))

    print(f"\nSuccessfully validated {len(validated_schemas)} distinct schemas.")
    assert len(validated_schemas) == 8, f"Expected 8 schemas, found {len(validated_schemas)}"

    # Validate mock demo record against RunMetadata schema
    run_meta_schema = validated_schemas.get("RunMetadata")
    assert run_meta_schema is not None, "RunMetadata schema not found"

    for idx, record in mock_records:
        validator = Draft202012Validator(run_meta_schema)
        errors = list(validator.iter_errors(record))
        if errors:
            print(f"[-] Mock record at block {idx} FAILED RunMetadata schema:")
            for err in errors:
                print(f"    - {err.message}")
            sys.exit(1)
        else:
            print(f"[+] Mock demo record '{record.get('run_id')}' conforms to RunMetadata schema.")

    print("\nALL 8 SCHEMAS AND DEMO RECORD SUCCESSFULLY VALIDATED.")

if __name__ == "__main__":
    validate_schemas()
