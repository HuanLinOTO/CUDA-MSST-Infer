#!/usr/bin/env python3
"""Convert HTML file to C++ header with embedded content."""

import sys
import os

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.html> <output.h>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read HTML file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Generate C++ header
    header_content = f'''#pragma once
// Auto-generated from {os.path.basename(input_file)}
// DO NOT EDIT MANUALLY

namespace cudasep::app::webui {{

inline const char* INDEX_HTML = R"HTML({content})HTML";

}} // namespace cudasep::app::webui
'''
    
    # Write header file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
    
    print(f"Generated {output_file} ({len(content)} bytes)")

if __name__ == '__main__':
    main()
