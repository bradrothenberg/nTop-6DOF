"""
AVL Output Parsers

Parse AVL (Athena Vortex Lattice) output files for aerodynamic data.
Supports parsing .st (stability derivatives) and .ft (force/moment) files.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def parse_avl_stability_file(filepath: str) -> Dict[str, float]:
    """
    Parse AVL stability derivatives output file (.sb or .st).

    Parameters
    ----------
    filepath : str
        Path to AVL stability derivatives file

    Returns
    -------
    dict
        Dictionary of stability derivatives

    Examples
    --------
    >>> derivs = parse_avl_stability_file('results.st')
    >>> print(f"CL_alpha = {derivs['CLa']}")
    """
    derivatives = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse key stability derivatives using regex
        # Format in AVL: CLa   =   5.234
        patterns = {
            'CLa': r'CLa\s*=\s*([-+]?\d+\.?\d*)',
            'CYb': r'CYb\s*=\s*([-+]?\d+\.?\d*)',
            'Clb': r'Clb\s*=\s*([-+]?\d+\.?\d*)',
            'Cmb': r'Cmb\s*=\s*([-+]?\d+\.?\d*)',
            'Cnb': r'Cnb\s*=\s*([-+]?\d+\.?\d*)',
            'CLq': r'CLq\s*=\s*([-+]?\d+\.?\d*)',
            'CMq': r'CMq\s*=\s*([-+]?\d+\.?\d*)',
            'CLp': r'CLp\s*=\s*([-+]?\d+\.?\d*)',
            'CYp': r'CYp\s*=\s*([-+]?\d+\.?\d*)',
            'Clp': r'Clp\s*=\s*([-+]?\d+\.?\d*)',
            'Cmp': r'Cmp\s*=\s*([-+]?\d+\.?\d*)',
            'Cnp': r'Cnp\s*=\s*([-+]?\d+\.?\d*)',
            'CLr': r'CLr\s*=\s*([-+]?\d+\.?\d*)',
            'CYr': r'CYr\s*=\s*([-+]?\d+\.?\d*)',
            'Clr': r'Clr\s*=\s*([-+]?\d+\.?\d*)',
            'Cmr': r'Cmr\s*=\s*([-+]?\d+\.?\d*)',
            'Cnr': r'Cnr\s*=\s*([-+]?\d+\.?\d*)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                derivatives[key] = float(match.group(1))

    except Exception as e:
        print(f"Warning: Error parsing AVL stability file: {e}")

    return derivatives


def parse_avl_forces_file(filepath: str) -> Dict[str, float]:
    """
    Parse AVL forces and moments output file (.ft).

    Parameters
    ----------
    filepath : str
        Path to AVL forces file

    Returns
    -------
    dict
        Dictionary of forces and moments

    Examples
    --------
    >>> results = parse_avl_forces_file('results.ft')
    >>> print(f"CL = {results['CL']}, CD = {results['CD']}")
    """
    results = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse key coefficients
        patterns = {
            'CL': r'CLtot\s*=\s*([-+]?\d+\.?\d*)',
            'CD': r'CDtot\s*=\s*([-+]?\d+\.?\d*)',
            'CY': r'CYtot\s*=\s*([-+]?\d+\.?\d*)',
            'Cl': r'Cltot\s*=\s*([-+]?\d+\.?\d*)',
            'Cm': r'Cmtot\s*=\s*([-+]?\d+\.?\d*)',
            'Cn': r'Cntot\s*=\s*([-+]?\d+\.?\d*)',
            'L/D': r'CLtot/CDtot\s*=\s*([-+]?\d+\.?\d*)',
            'e': r'e\s*=\s*([-+]?\d+\.?\d*)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                results[key] = float(match.group(1))

    except Exception as e:
        print(f"Warning: Error parsing AVL forces file: {e}")

    return results


def parse_avl_run_file(filepath: str) -> Dict[str, Any]:
    """
    Parse AVL run case file (.run).

    Parameters
    ----------
    filepath : str
        Path to AVL run file

    Returns
    -------
    dict
        Dictionary of run case parameters
    """
    run_data = {}

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse run case format (simplified)
        for line in lines:
            # Look for key parameters
            if 'alpha' in line.lower() and '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    try:
                        run_data['alpha'] = float(parts[1].strip().split()[0])
                    except:
                        pass

            if 'beta' in line.lower() and '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    try:
                        run_data['beta'] = float(parts[1].strip().split()[0])
                    except:
                        pass

    except Exception as e:
        print(f"Warning: Error parsing AVL run file: {e}")

    return run_data


def parse_avl_mass_file(filepath: str) -> Dict[str, Any]:
    """
    Parse AVL mass file (.mass).

    Parameters
    ----------
    filepath : str
        Path to AVL mass file

    Returns
    -------
    dict
        Dictionary of mass properties
    """
    mass_data = {}

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse mass file format
        # First line: aircraft name
        # Then: mass, CG location, inertias
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try to parse numeric data
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Format: mass  Ixx  Iyy  Izz  Ixz
                    if 'mass' not in mass_data:
                        mass_data['mass'] = float(parts[0])
                        mass_data['CG'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif 'Ixx' not in mass_data:
                        mass_data['Ixx'] = float(parts[0])
                        mass_data['Iyy'] = float(parts[1])
                        mass_data['Izz'] = float(parts[2])
                        if len(parts) >= 4:
                            mass_data['Ixz'] = float(parts[3])
                except:
                    pass

    except Exception as e:
        print(f"Warning: Error parsing AVL mass file: {e}")

    return mass_data


def extract_stability_derivatives(avl_output: str) -> Dict[str, float]:
    """
    Extract stability derivatives from AVL console output.

    Parameters
    ----------
    avl_output : str
        Raw AVL output text

    Returns
    -------
    dict
        Stability derivatives
    """
    derivs = {}

    # Common patterns
    patterns = {
        'CL_alpha': r'CL\s*alpha\s*=\s*([-+]?\d+\.?\d*)',
        'Cm_alpha': r'Cm\s*alpha\s*=\s*([-+]?\d+\.?\d*)',
        'CL_beta': r'CL\s*beta\s*=\s*([-+]?\d+\.?\d*)',
        'Cm_q': r'Cm\s*q\s*=\s*([-+]?\d+\.?\d*)',
        'Cl_p': r'Cl\s*p\s*=\s*([-+]?\d+\.?\d*)',
        'Cn_r': r'Cn\s*r\s*=\s*([-+]?\d+\.?\d*)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, avl_output, re.IGNORECASE)
        if match:
            derivs[key] = float(match.group(1))

    return derivs


def test_parsers():
    """Test AVL parsers."""
    print("=" * 60)
    print("AVL Parser Test")
    print("=" * 60)
    print()
    print("Parser functions created:")
    print("  - parse_avl_stability_file()")
    print("  - parse_avl_forces_file()")
    print("  - parse_avl_run_file()")
    print("  - parse_avl_mass_file()")
    print()
    print("These parsers extract data from AVL output files.")
    print()


if __name__ == "__main__":
    test_parsers()
