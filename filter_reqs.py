import importlib.metadata
import re
import sys


def get_installed_packages():
    return {
        dist.metadata["Name"].lower().replace("-", "_")
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }


def filter_requirements(req_file, out_file):
    installed = get_installed_packages()
    to_install = []

    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            clean_line = line.split(";")[0].strip()
            match = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9_\-.]*)", clean_line)

            if match:
                pkg_name = match.group(1).lower().replace("-", "_")
                if pkg_name in installed:
                    print(f"Skipping {line} (already installed)")
                else:
                    to_install.append(line)
            else:
                print(f"Keeping complex requirement: {line}")
                to_install.append(line)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(to_install))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 filter_reqs.py <input_requirements> <output_requirements>")
        sys.exit(1)

    filter_requirements(sys.argv[1], sys.argv[2])
