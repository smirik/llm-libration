#!/usr/bin/env python3
"""
Patch transformers library to fix MLX compatibility issue.

This script patches a bug in transformers 4.57.x where `_config.model_type`
is accessed on a dict object instead of using `_config.get("model_type")`.

Issue: https://github.com/huggingface/transformers/issues/42369
Fix PR: https://github.com/huggingface/transformers/pull/42389

Once PR #42389 is merged and released (likely transformers 4.57.3+),
this patch will no longer be needed.
"""

import re
import sys
from pathlib import Path


def find_transformers_file() -> Path | None:
    """Find the tokenization_utils_base.py file in installed transformers."""
    try:
        import transformers
        base_path = Path(transformers.__file__).parent
        target_file = base_path / "tokenization_utils_base.py"
        if target_file.exists():
            return target_file
    except ImportError:
        pass
    return None


def check_if_patched(content: str) -> bool:
    """Check if the file is already patched."""
    # If we find .get("model_type") pattern, it's already patched
    return '_config.get("model_type")' in content


def check_if_needs_patch(content: str) -> bool:
    """Check if the file has the buggy pattern."""
    return '_config.model_type' in content


def apply_patch(file_path: Path) -> bool:
    """Apply the patch to fix the bug."""
    content = file_path.read_text()

    if check_if_patched(content):
        print(f"✅ Already patched: {file_path}")
        return True

    if not check_if_needs_patch(content):
        print(f"⚠️  Pattern not found, may be a different version: {file_path}")
        return False

    # Apply the patch
    patched_content = content.replace('_config.model_type', '_config.get("model_type")')

    # Verify the patch was applied
    if patched_content == content:
        print(f"❌ Patch failed - no changes made: {file_path}")
        return False

    # Write the patched content
    file_path.write_text(patched_content)

    # Count replacements
    count = content.count('_config.model_type')
    print(f"✅ Patched {count} occurrence(s) in: {file_path}")
    return True


def main():
    print("=" * 60)
    print("Transformers MLX Compatibility Patcher")
    print("=" * 60)
    print()
    print("Issue: https://github.com/huggingface/transformers/issues/42369")
    print("Fix:   https://github.com/huggingface/transformers/pull/42389")
    print()

    # Find transformers
    target_file = find_transformers_file()

    if target_file is None:
        print("❌ Could not find transformers installation.")
        print("   Make sure transformers is installed: pip install transformers")
        sys.exit(1)

    print(f"Found: {target_file}")
    print()

    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")

        # Parse version to check if patch might be included
        version_parts = version.replace('.dev0', '').split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        patch_num = int(version_parts[2]) if len(version_parts) > 2 else 0

        if major >= 5 or (major == 4 and minor == 57 and patch_num >= 3):
            print(f"⚠️  Version {version} may already include the fix.")
            print("   Checking anyway...")
    except Exception:
        pass

    print()

    # Apply patch
    success = apply_patch(target_file)

    print()
    if success:
        print("🎉 Patch complete! MLX models should now load correctly.")
        print()
        print("Note: This patch will be lost if you reinstall transformers.")
        print("      Re-run this script after: uv sync, pip install, etc.")
    else:
        print("❌ Patch failed. You may need to manually edit the file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
